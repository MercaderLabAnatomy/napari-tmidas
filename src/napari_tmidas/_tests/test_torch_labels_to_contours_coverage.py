"""Coverage tests for ``processing_functions.torch_labels_to_contours``.

PyTorch is not installed in this environment, so the module under test
short-circuits to its ``TORCH_AVAILABLE = False`` guard on a normal
import.  To exercise the real algorithmic body we load a *second copy*
of the very same source file with a minimal numpy-backed stand-in for
``torch`` / ``torch.nn.functional`` installed in ``sys.modules``.

The stand-in only supplies the array primitives (``zeros``, ``pad``,
``conv2d``, ...).  Every line of control flow, every boundary offset
loop and every zarr write below is executed by the real module source,
so the assertions pin real module behaviour rather than the shim's.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types

import numpy as np
import pytest
import zarr

from napari_tmidas.processing_functions import (
    torch_labels_to_contours as real_tlc,
)

MODULE_PATH = real_tlc.__file__
TORCH_REALLY_INSTALLED = real_tlc.TORCH_AVAILABLE


# ---------------------------------------------------------------------
# numpy-backed torch stand-in
# ---------------------------------------------------------------------
def _wrap(array):
    """Return ``array`` viewed as a fake tensor (never a copy)."""
    return np.asarray(array).view(_FakeTensor)


class _FakeTensor(np.ndarray):
    """Minimal numpy-backed stand-in for ``torch.Tensor``."""

    @property
    def device(self):
        return "cpu"

    def to(self, target, *args, **kwargs):
        if isinstance(target, str):
            return self
        return _wrap(np.asarray(self).astype(target))

    def float(self):  # noqa: A003
        return _wrap(np.asarray(self, dtype=np.float32))

    def long(self):
        return _wrap(np.asarray(self, dtype=np.int64))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, dim):
        return _wrap(np.expand_dims(np.asarray(self), dim))

    def view(self, *args, **kwargs):
        if args and all(isinstance(a, int) for a in args):
            return _wrap(np.reshape(np.asarray(self), args))
        return np.ndarray.view(self, *args, **kwargs)

    def zero_(self):
        self[...] = 0
        return self

    def copy_(self, src, non_blocking=False):
        self[...] = np.asarray(src)
        return self


def _fake_zeros(*size, dtype=None, device=None, pin_memory=False):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return _wrap(np.zeros(tuple(size), dtype=dtype))


def _fake_zeros_like(tensor, dtype=None):
    return _wrap(np.zeros_like(np.asarray(tensor), dtype=dtype))


def _fake_from_numpy(array):
    return _wrap(array)


def _fake_arange(stop, dtype=None, device=None):
    return _wrap(np.arange(stop, dtype=dtype))


def _fake_exp(tensor):
    return _wrap(np.exp(np.asarray(tensor)))


def _fake_pad(inp, pad, mode="constant", value=0):
    """torch-style padding: pairs apply to the trailing dims, reversed."""
    array = np.asarray(inp)
    widths = [(0, 0)] * array.ndim
    for i in range(len(pad) // 2):
        widths[array.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    if mode == "replicate":
        out = np.pad(array, widths, mode="edge")
    else:
        out = np.pad(array, widths, mode="constant", constant_values=value)
    return _wrap(out)


def _fake_conv2d(inp, weight):
    x = np.asarray(inp)
    w = np.asarray(weight)
    n, _, h, width = x.shape
    cout, cin, kh, kw = w.shape
    oh = h - kh + 1
    ow = width - kw + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.result_type(x, w))
    for co in range(cout):
        for ci in range(cin):
            for i in range(kh):
                for j in range(kw):
                    out[:, co] += (
                        x[:, ci, i : i + oh, j : j + ow] * w[co, ci, i, j]
                    )
    return _wrap(out)


def _fake_conv3d(inp, weight):
    x = np.asarray(inp)
    w = np.asarray(weight)
    n, _, d, h, width = x.shape
    cout, cin, kd, kh, kw = w.shape
    od = d - kd + 1
    oh = h - kh + 1
    ow = width - kw + 1
    out = np.zeros((n, cout, od, oh, ow), dtype=np.result_type(x, w))
    for co in range(cout):
        for ci in range(cin):
            for k in range(kd):
                for i in range(kh):
                    for j in range(kw):
                        out[:, co] += (
                            x[
                                :,
                                ci,
                                k : k + od,
                                i : i + oh,
                                j : j + ow,
                            ]
                            * w[co, ci, k, i, j]
                        )
    return _wrap(out)


class _FakeDeviceProperties:
    name = "FakeGPU"
    major = 12
    minor = 0
    total_memory = 8 * 1024**3


def _build_fake_torch():
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _FakeTensor
    torch_mod.bool = np.dtype(np.bool_)
    torch_mod.uint8 = np.dtype(np.uint8)
    torch_mod.uint16 = np.dtype(np.uint16)
    torch_mod.uint32 = np.dtype(np.uint32)
    torch_mod.int32 = np.dtype(np.int32)
    torch_mod.int64 = np.dtype(np.int64)
    torch_mod.float32 = np.dtype(np.float32)
    torch_mod.zeros = _fake_zeros
    torch_mod.zeros_like = _fake_zeros_like
    torch_mod.from_numpy = _fake_from_numpy
    torch_mod.arange = _fake_arange
    torch_mod.exp = _fake_exp
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
        get_device_properties=lambda device: _FakeDeviceProperties(),
    )

    nn_mod = types.ModuleType("torch.nn")
    fn_mod = types.ModuleType("torch.nn.functional")
    fn_mod.pad = _fake_pad
    fn_mod.conv2d = _fake_conv2d
    fn_mod.conv3d = _fake_conv3d
    nn_mod.functional = fn_mod
    torch_mod.nn = nn_mod
    return torch_mod, nn_mod, fn_mod


def _load_with_fake_torch():
    """Exec the module source again with the numpy-backed torch shim."""
    fake_torch, fake_nn, fake_fn = _build_fake_torch()
    keys = ("torch", "torch.nn", "torch.nn.functional")
    saved = {key: sys.modules.get(key) for key in keys}
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.nn.functional"] = fake_fn
    try:
        spec = importlib.util.spec_from_file_location(
            "_tlc_with_fake_torch", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


tlc = _load_with_fake_torch()


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _block_2d(dtype=np.uint16):
    """5x5 field holding a solid 3x3 block of label 1 at [1:4, 1:4]."""
    frame = np.zeros((5, 5), dtype=dtype)
    frame[1:4, 1:4] = 1
    return frame


def _block_3d(dtype=np.uint16):
    """5x5x5 field holding a solid 3x3x3 block of label 1."""
    vol = np.zeros((5, 5, 5), dtype=dtype)
    vol[1:4, 1:4, 1:4] = 1
    return vol


def _tensor(array):
    return tlc.torch.from_numpy(np.ascontiguousarray(array))


# ---------------------------------------------------------------------
class TestFindBoundariesTorch:
    """Pin the 2D/3D boundary detector and its dtype + ndim guards."""

    def test_solid_block_ring_is_boundary_center_is_not(self):
        out = tlc._find_boundaries_torch(_tensor(_block_2d()))
        arr = np.asarray(out)
        assert arr.dtype == np.bool_
        assert arr.shape == (5, 5)
        # the 8 ring pixels of the 3x3 block touch background
        assert arr.sum() == 8
        assert not arr[2, 2]
        assert arr[1, 1] and arr[3, 3] and arr[1, 3] and arr[3, 1]
        # background never becomes a boundary in this implementation
        assert arr[0].sum() == 0
        assert arr[4].sum() == 0

    def test_single_pixel_label_is_entirely_boundary(self):
        frame = np.zeros((3, 3), dtype=np.uint16)
        frame[1, 1] = 7
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(frame)))
        assert arr.sum() == 1
        assert arr[1, 1]

    def test_touching_labels_are_boundaries_on_both_sides(self):
        frame = np.zeros((4, 4), dtype=np.int32)
        frame[:, :2] = 1
        frame[:, 2:] = 2
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(frame)))
        # everything touches either the image edge or the other label
        assert arr.all()

    def test_uniform_interior_of_large_label_is_not_boundary(self):
        frame = np.ones((5, 5), dtype=np.int32)
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(frame)))
        # padding is 0 so the outer ring differs from its neighbours
        assert arr[1:4, 1:4].sum() == 0
        assert arr[0].all()

    def test_empty_frame_has_no_boundaries(self):
        frame = np.zeros((4, 4), dtype=np.uint8)
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(frame)))
        assert arr.sum() == 0

    @pytest.mark.parametrize(
        "dtype", [np.uint8, np.uint16, np.uint32, np.int32, np.int64]
    )
    def test_integer_dtypes_all_give_the_same_boundary(self, dtype):
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(_block_2d(dtype))))
        assert arr.dtype == np.bool_
        assert arr.sum() == 8

    def test_float_labels_are_cast_through_long(self):
        # exercises the ``labels.long()`` fallback branch
        arr = np.asarray(
            tlc._find_boundaries_torch(_tensor(_block_2d(np.float32)))
        )
        assert arr.sum() == 8

    def test_mode_argument_is_accepted_but_ignored(self):
        ref = np.asarray(tlc._find_boundaries_torch(_tensor(_block_2d())))
        for mode in ("outer", "inner", "thick", "not-a-mode"):
            other = np.asarray(
                tlc._find_boundaries_torch(_tensor(_block_2d()), mode=mode)
            )
            assert np.array_equal(ref, other)

    def test_3d_block_ring_is_boundary_center_is_not(self):
        arr = np.asarray(tlc._find_boundaries_torch(_tensor(_block_3d())))
        assert arr.shape == (5, 5, 5)
        # 27 voxels in the block, the single centre voxel is interior
        assert arr.sum() == 26
        assert not arr[2, 2, 2]

    def test_1d_input_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2D"):
            tlc._find_boundaries_torch(_tensor(np.arange(4, dtype=np.int32)))

    def test_4d_input_raises_value_error(self):
        vol = np.zeros((2, 2, 2, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported number of dim"):
            tlc._find_boundaries_torch(_tensor(vol))


# ---------------------------------------------------------------------
class TestGaussianFilterTorch:
    """Pin the separable Gaussian helper, incl. its short-circuits."""

    @pytest.mark.parametrize("sigma", [0.0, -1.0, -0.001])
    def test_non_positive_sigma_returns_input_object(self, sigma):
        src = _tensor(np.ones((4, 4), dtype=np.float32))
        assert tlc._gaussian_filter_torch(src, sigma) is src

    def test_constant_2d_field_is_preserved(self):
        src = _tensor(np.full((6, 6), 3.0, dtype=np.float32))
        out = np.asarray(tlc._gaussian_filter_torch(src, 1.0))
        assert out.shape == (6, 6)
        assert np.allclose(out, 3.0, atol=1e-5)

    def test_even_kernel_size_is_bumped_to_odd(self):
        # int(4 * 0.25 + 1) == 2 -> the "+= 1" branch, then max(3, 3)
        src = _tensor(np.full((5, 5), 2.0, dtype=np.float32))
        out = np.asarray(tlc._gaussian_filter_torch(src, 0.25))
        assert out.shape == (5, 5)
        assert np.allclose(out, 2.0, atol=1e-5)

    def test_delta_is_spread_and_mass_is_conserved(self):
        frame = np.zeros((11, 11), dtype=np.float32)
        frame[5, 5] = 1.0
        out = np.asarray(tlc._gaussian_filter_torch(_tensor(frame), 1.0))
        assert out[5, 5] < 1.0
        assert out[5, 4] > 0.0
        assert out.sum() == pytest.approx(1.0, abs=1e-4)
        # separable kernel is symmetric
        assert out[5, 4] == pytest.approx(out[5, 6], abs=1e-6)
        assert out[4, 5] == pytest.approx(out[5, 4], abs=1e-6)

    def test_constant_3d_field_is_preserved(self):
        src = _tensor(np.full((5, 6, 7), 1.5, dtype=np.float32))
        out = np.asarray(tlc._gaussian_filter_torch(src, 0.5))
        assert out.shape == (5, 6, 7)
        assert np.allclose(out, 1.5, atol=1e-5)

    def test_3d_delta_is_spread(self):
        vol = np.zeros((7, 7, 7), dtype=np.float32)
        vol[3, 3, 3] = 1.0
        out = np.asarray(tlc._gaussian_filter_torch(_tensor(vol), 0.5))
        assert out[3, 3, 3] < 1.0
        assert out[3, 3, 2] > 0.0
        assert out.sum() == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("shape", [(6,), (2, 2, 2, 2)])
    def test_unsupported_ndim_is_rejected(self, shape):
        # Only the 2D and 3D convolution paths are implemented below;
        # any other rank used to come back completely unfiltered with no
        # error, silently skipping the smoothing the caller asked for.
        src = _tensor(np.ones(shape, dtype=np.float32))
        with pytest.raises(ValueError, match="2D or 3D"):
            tlc._gaussian_filter_torch(src, 1.0)


# ---------------------------------------------------------------------
class TestLabelsToContoursGuards:
    """Pin the guard clauses of the public entry point."""

    @pytest.mark.skipif(
        TORCH_REALLY_INSTALLED, reason="torch is installed in this env"
    )
    def test_runtime_error_without_torch(self):
        assert real_tlc.TORCH_AVAILABLE is False
        with pytest.raises(RuntimeError, match="PyTorch is not installed"):
            real_tlc.labels_to_contours_torch(
                np.zeros((1, 2, 2), dtype=np.uint8)
            )

    def test_mismatched_ensemble_shapes_raise(self, tmp_path):
        first = np.zeros((2, 4, 4), dtype=np.uint8)
        second = np.zeros((2, 5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"Label 1 has shape"):
            tlc.labels_to_contours_torch(
                [first, second],
                foreground_store_or_path=tmp_path / "fg.zarr",
                contours_store_or_path=tmp_path / "ct.zarr",
            )

    def test_single_array_is_wrapped_in_a_list(self, tmp_path, capsys):
        labels = np.zeros((3, 4, 4), dtype=np.uint8)
        tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert "Processing 1 label image(s)" in capsys.readouterr().out

    def test_list_input_is_treated_as_an_ensemble(self, tmp_path, capsys):
        labels = np.zeros((2, 4, 4), dtype=np.uint8)
        tlc.labels_to_contours_torch(
            [labels, labels.copy()],
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert "Processing 2 label image(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------
class TestLabelsToContoursOutputs:
    """Pin the zarr plumbing and the numeric contents it receives."""

    def test_zarr_layout_dtypes_and_files(self, tmp_path):
        labels = np.stack([_block_2d(), _block_2d()])
        fg_path = tmp_path / "fg.zarr"
        ct_path = tmp_path / "ct.zarr"
        foreground, contours = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=fg_path,
            contours_store_or_path=ct_path,
        )
        assert isinstance(foreground, zarr.Array)
        assert isinstance(contours, zarr.Array)
        assert foreground.shape == (2, 5, 5)
        assert contours.shape == (2, 5, 5)
        assert foreground.dtype == np.dtype(bool)
        assert contours.dtype == np.dtype(np.float32)
        assert foreground.chunks == (1, 5, 5)
        assert contours.chunks == (1, 5, 5)
        assert fg_path.exists()
        assert ct_path.exists()

    def test_known_foreground_and_contours_for_a_block(self, tmp_path):
        labels = _block_2d()[None, ...]
        foreground, contours = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        fg = foreground[:]
        ct = contours[:]
        assert fg.sum() == 9
        assert fg[0, 2, 2]
        assert not fg[0, 0, 0]
        # single member ensemble -> contour values are exactly 1.0
        assert ct.sum() == pytest.approx(8.0)
        assert ct[0, 1, 1] == pytest.approx(1.0)
        assert ct[0, 2, 2] == pytest.approx(0.0)

    def test_ensemble_average_halves_a_lone_vote(self, tmp_path):
        voting = _block_2d()[None, ...]
        empty = np.zeros_like(voting)
        foreground, contours = tlc.labels_to_contours_torch(
            [voting, empty],
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        ct = contours[:]
        assert ct[0, 1, 1] == pytest.approx(0.5)
        assert ct[0, 2, 2] == pytest.approx(0.0)
        assert ct.sum() == pytest.approx(4.0)
        # foreground is the union over the ensemble
        assert foreground[:].sum() == 9

    def test_foreground_is_a_union_of_disjoint_members(self, tmp_path):
        left = np.zeros((1, 4, 4), dtype=np.uint8)
        left[0, 0, 0] = 1
        right = np.zeros((1, 4, 4), dtype=np.uint8)
        right[0, 3, 3] = 2
        foreground, _ = tlc.labels_to_contours_torch(
            [left, right],
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        fg = foreground[:]
        assert fg[0, 0, 0]
        assert fg[0, 3, 3]
        assert fg.sum() == 2

    def test_timepoints_are_processed_independently(self, tmp_path):
        labels = np.zeros((3, 5, 5), dtype=np.uint16)
        labels[1] = _block_2d()
        foreground, contours = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        fg = foreground[:]
        ct = contours[:]
        assert fg[0].sum() == 0
        assert fg[1].sum() == 9
        assert fg[2].sum() == 0
        assert ct[0].sum() == pytest.approx(0.0)
        assert ct[1].sum() == pytest.approx(8.0)
        # buffers are reset between timepoints, not carried over
        assert ct[2].sum() == pytest.approx(0.0)

    def test_3d_frames_are_supported(self, tmp_path):
        labels = _block_3d()[None, ...]
        foreground, contours = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert foreground.shape == (1, 5, 5, 5)
        assert foreground.chunks == (1, 5, 5, 5)
        assert foreground[:].sum() == 27
        assert contours[:].sum() == pytest.approx(26.0)
        assert contours[0, 2, 2, 2] == pytest.approx(0.0)

    def test_sigma_smooths_and_normalises_to_one(self, tmp_path):
        labels = _block_2d()[None, ...]
        _, contours = tlc.labels_to_contours_torch(
            labels,
            sigma=1.0,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        ct = contours[:]
        assert ct.max() == pytest.approx(1.0, abs=1e-5)
        # smoothing leaks contour signal into the hole and the background
        assert ct[0, 2, 2] > 0.0
        assert ct[0, 0, 0] > 0.0

    def test_sigma_on_an_empty_frame_skips_normalisation(self, tmp_path):
        labels = np.zeros((1, 6, 6), dtype=np.uint8)
        _, contours = tlc.labels_to_contours_torch(
            labels,
            sigma=2.0,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert contours[:].max() == pytest.approx(0.0)

    def test_zero_sigma_leaves_contours_unsmoothed(self, tmp_path):
        labels = _block_2d()[None, ...]
        _, contours = tlc.labels_to_contours_torch(
            labels,
            sigma=0.0,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert contours[:].sum() == pytest.approx(8.0)

    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.uint16, np.uint32, np.int32, np.int64, np.float32],
    )
    def test_every_label_dtype_reaches_the_same_result(self, tmp_path, dtype):
        # unsigned dtypes take the int32 cast, signed ones pass through
        # and float labels fall back to ``.long()``
        labels = _block_2d(dtype)[None, ...]
        foreground, contours = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert foreground.dtype == np.dtype(bool)
        assert contours.dtype == np.dtype(np.float32)
        assert foreground[:].sum() == 9
        assert contours[:].sum() == pytest.approx(8.0)

    def test_none_paths_produce_in_memory_zarr_arrays(self):
        labels = _block_2d()[None, ...]
        foreground, contours = tlc.labels_to_contours_torch(labels)
        assert isinstance(foreground, zarr.Array)
        assert foreground.shape == (1, 5, 5)
        assert foreground[:].sum() == 9
        assert contours[:].sum() == pytest.approx(8.0)

    def test_string_paths_are_accepted(self, tmp_path):
        labels = _block_2d()[None, ...]
        foreground, _ = tlc.labels_to_contours_torch(
            labels,
            foreground_store_or_path=str(tmp_path / "fg.zarr"),
            contours_store_or_path=str(tmp_path / "ct.zarr"),
        )
        assert (tmp_path / "fg.zarr").exists()
        assert foreground[:].sum() == 9

    def test_zarr_array_input_is_read_frame_by_frame(self, tmp_path):
        source = zarr.open(
            tmp_path / "src.zarr",
            mode="w",
            shape=(2, 5, 5),
            dtype=np.uint16,
            chunks=(1, 5, 5),
        )
        source[0] = _block_2d()
        source[1] = np.zeros((5, 5), dtype=np.uint16)
        foreground, contours = tlc.labels_to_contours_torch(
            source,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert foreground[0].sum() == 9
        assert foreground[1].sum() == 0
        assert contours[0].sum() == pytest.approx(8.0)


# ---------------------------------------------------------------------
class TestDeviceHandling:
    """Pin device selection and the pinned-memory transfer branches."""

    def test_cpu_device_disables_pinned_memory(self, tmp_path):
        calls = []
        original = tlc.torch.zeros

        def recording_zeros(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        tlc.torch.zeros = recording_zeros
        try:
            tlc.labels_to_contours_torch(
                _block_2d()[None, ...],
                device="cpu",
                use_pinned_memory=True,
                foreground_store_or_path=tmp_path / "fg.zarr",
                contours_store_or_path=tmp_path / "ct.zarr",
            )
        finally:
            tlc.torch.zeros = original
        assert calls, "torch.zeros was never called"
        assert not any(kw.get("pin_memory") for kw in calls)

    def test_device_defaults_to_cpu_when_cuda_is_absent(
        self, tmp_path, capsys
    ):
        tlc.labels_to_contours_torch(
            _block_2d()[None, ...],
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        out = capsys.readouterr().out
        assert "Using PyTorch device: cpu" in out
        assert "GPU:" not in out

    def test_cuda_autodetected_and_properties_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(tlc.torch.cuda, "is_available", lambda: True)
        foreground, contours = tlc.labels_to_contours_torch(
            _block_2d()[None, ...],
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        out = capsys.readouterr().out
        assert "Using PyTorch device: cuda" in out
        assert "GPU: FakeGPU" in out
        assert "Compute capability: 12.0" in out
        assert "VRAM: 8.0 GB" in out
        assert "Pinned memory enabled" not in out
        assert foreground[:].sum() == 9
        assert contours[:].sum() == pytest.approx(8.0)

    def test_pinned_memory_path_allocates_buffers_and_syncs(
        self, tmp_path, monkeypatch, capsys
    ):
        syncs = []
        monkeypatch.setattr(tlc.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            tlc.torch.cuda, "synchronize", lambda: syncs.append(1)
        )
        pinned = []
        original = tlc.torch.zeros

        def recording_zeros(*args, **kwargs):
            if kwargs.get("pin_memory"):
                pinned.append(kwargs.get("dtype"))
            return original(*args, **kwargs)

        monkeypatch.setattr(tlc.torch, "zeros", recording_zeros)
        foreground, contours = tlc.labels_to_contours_torch(
            np.stack([_block_2d(), _block_2d()]),
            device="cuda",
            use_pinned_memory=True,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert "Pinned memory enabled" in capsys.readouterr().out
        assert pinned == [np.dtype(bool), np.dtype(np.float32)]
        # one synchronize per timepoint
        assert len(syncs) == 2
        # results are identical to the non-pinned path
        assert foreground[:].sum() == 18
        assert contours[:].sum() == pytest.approx(16.0)

    def test_pinned_memory_ignored_for_explicit_cpu_string(self, tmp_path):
        # ``use_pinned_memory`` is forced off before the buffers are made
        foreground, _ = tlc.labels_to_contours_torch(
            _block_2d()[None, ...],
            device="cpu",
            use_pinned_memory=True,
            foreground_store_or_path=tmp_path / "fg.zarr",
            contours_store_or_path=tmp_path / "ct.zarr",
        )
        assert foreground[:].sum() == 9


# ---------------------------------------------------------------------
class TestModuleSelfTest:
    """Pin the module's own ``test_torch_labels_to_contours`` helper."""

    def test_self_test_is_a_noop_without_torch(self, capsys):
        if TORCH_REALLY_INSTALLED:
            pytest.skip("torch is installed in this env")
        assert real_tlc.test_torch_labels_to_contours() is None
        assert "PyTorch not available" in capsys.readouterr().out

    def test_self_test_runs_end_to_end(self, tmp_path, monkeypatch, capsys):
        # keep the module's tempfile usage inside the pytest tmp dir
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        tlc.test_torch_labels_to_contours()
        out = capsys.readouterr().out
        assert "All checks passed" in out
        assert "Foreground: (5, 100, 100)" in out
