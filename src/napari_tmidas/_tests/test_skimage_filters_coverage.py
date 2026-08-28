# napari_tmidas/_tests/test_skimage_filters_coverage.py
"""
Coverage-focused tests for ``processing_functions/skimage_filters.py``.

The sibling ``test_skimage_filters.py`` pins the headline behaviour of a
handful of filters.  This file walks the branches that file leaves cold:
the dimension-order block iterator every labelling filter is built on, the
per-dimension-order branches of the Otsu filters, the Dask code paths
(CLAHE, adaptive threshold, resize), the guard clauses that turn bad input
into a clear error, and the optional-import fallbacks at the top of the
module.
"""
import ast
import builtins
import json
import os
import sys

import numpy as np
import pytest
import tifffile

import napari_tmidas.processing_functions.skimage_filters as sf


def _blob_image(shape, seed=0, dtype=np.uint16, high=1000):
    """A reproducible non-degenerate image of the requested shape."""
    rng = np.random.default_rng(seed)
    image = (rng.random(shape) * high).astype(dtype)
    # Guarantee at least two intensity levels so threshold_otsu is defined.
    image[..., 0, 0] = 0
    image[..., -1, -1] = high
    return image


class TestSkimageOptionalImportFallback:
    """
    ``SKIMAGE_AVAILABLE`` and the sibling ``_HAS_PANDAS`` flag each come
    from a try/except ImportError wrapped around the module's top-level
    imports (lines ~10-30). Nothing else in this file forces that except
    branch to run, so it was previously untested despite this file's own
    module docstring claiming to cover "the optional-import fallbacks at
    the top of the module".

    Rather than ``importlib.reload`` the shared ``sf`` module (which would
    re-run every ``@BatchProcessingRegistry.register(...)`` decorator
    further down the file and mutate that process-wide registry for every
    later test in the session -- exactly the kind of unrestored global
    state this audit was told to hunt for), the module's own preamble
    source text is compiled and executed standalone in a scratch
    namespace. That exercises the real except branch -- proving the flag
    actually flips and the warning actually prints, not just that the
    module *would* raise if skimage were absent -- without touching the
    shared module or the registry at all.
    """

    @staticmethod
    def _preamble_code(index):
        """Compile the module's Nth top-level try/except block standalone."""
        with open(sf.__file__) as fh:
            source = fh.read()
        tree = ast.parse(source)
        try_nodes = [n for n in tree.body if isinstance(n, ast.Try)]
        preamble = ast.Module(body=[try_nodes[index]], type_ignores=[])
        ast.fix_missing_locations(preamble)
        return compile(preamble, sf.__file__, "exec")

    def test_missing_skimage_flips_the_flag_and_warns(
        self, monkeypatch, capsys
    ):
        code = self._preamble_code(0)  # the SKIMAGE_AVAILABLE try/except
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.startswith("skimage"):
                raise ImportError("simulated: scikit-image not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        scratch = {"__builtins__": builtins}
        exec(code, scratch)

        assert scratch["SKIMAGE_AVAILABLE"] is False
        assert (
            "scikit-image not available, some processing functions will "
            "be disabled" in capsys.readouterr().out
        )
        # The shared module (every other test in this file imports and
        # calls it directly) must be completely unaffected.
        assert sf.SKIMAGE_AVAILABLE is True

    def test_missing_pandas_flips_its_flag_without_raising(self, monkeypatch):
        code = self._preamble_code(1)  # the _HAS_PANDAS try/except
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("simulated: pandas not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        scratch = {"__builtins__": builtins}
        exec(code, scratch)

        assert scratch["_HAS_PANDAS"] is False
        assert scratch["pd"] is None
        assert sf._HAS_PANDAS is True

    def test_missing_skimage_stub_functions_raise_import_error(self):
        """
        When ``SKIMAGE_AVAILABLE`` is False, the module defines three stub
        replacements (``invert_image``, ``equalize_histogram``,
        ``otsu_thresholding``) that just raise ImportError instead of the
        real, skimage-backed versions. Extract exactly that ``else:``
        branch -- the module's own source text -- and execute its
        (undecorated, registry-free) function defs standalone to prove the
        stubs actually raise, without reloading the shared module.
        """
        with open(sf.__file__) as fh:
            source = fh.read()
        tree = ast.parse(source)
        skimage_available_if = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "SKIMAGE_AVAILABLE"
            and node.orelse
        )
        stub_module = ast.Module(body=skimage_available_if.orelse, type_ignores=[])
        ast.fix_missing_locations(stub_module)
        code = compile(stub_module, sf.__file__, "exec")

        scratch = {"__builtins__": builtins}
        exec(code, scratch)

        for name in ("invert_image", "equalize_histogram", "otsu_thresholding"):
            with pytest.raises(ImportError, match="scikit-image is not available"):
                scratch[name]()
            # Must be the disabled-mode stub, not the real registered
            # function still live on the shared module.
            assert scratch[name] is not getattr(sf, name)


class TestIterDimensionBlocks:
    """
    ``_iter_dimension_blocks`` decides what a "block" is for every labelling
    filter in the module: T and C are looped over independently, Z stays
    glued to Y/X so a 3D object spanning slices keeps one label.  Getting
    this wrong silently mislabels data, so each branch is pinned here.
    """

    def test_2d_yields_the_whole_image_regardless_of_hint(self):
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        blocks = list(sf._iter_dimension_blocks(image, "nonsense"))
        assert len(blocks) == 1
        index, block = blocks[0]
        assert index == ()
        assert block is image

    def test_zyx_keeps_the_volume_in_one_block(self):
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        blocks = list(sf._iter_dimension_blocks(image, "ZYX"))
        assert [b.shape for _, b in blocks] == [(3, 4, 5)]
        assert blocks[0][0] == ()

    def test_tyx_yields_one_block_per_timepoint(self):
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        blocks = list(sf._iter_dimension_blocks(image, "TYX"))
        assert len(blocks) == 3
        assert [b.shape for _, b in blocks] == [(4, 5)] * 3
        assert [i[0] for i, _ in blocks] == [0, 1, 2]

    def test_tczyx_loops_t_and_c_but_not_z(self):
        image = np.zeros((2, 3, 4, 5, 6), dtype=np.uint8)
        blocks = list(sf._iter_dimension_blocks(image, "TCZYX"))
        assert len(blocks) == 2 * 3
        assert {b.shape for _, b in blocks} == {(4, 5, 6)}

    def test_lowercase_hint_is_accepted(self):
        image = np.zeros((2, 4, 5), dtype=np.uint8)
        assert len(list(sf._iter_dimension_blocks(image, "tyx"))) == 2

    def test_stale_channel_in_hint_is_stripped(self):
        """
        A "TCZYX" hint survives upstream channel extraction; the helper must
        retry without the C rather than refuse a now 4D image.
        """
        image = np.zeros((2, 3, 4, 5), dtype=np.uint8)
        blocks = list(sf._iter_dimension_blocks(image, "TCZYX"))
        # Resolved as TZYX: T alone is independent, Z stays with Y/X.
        assert len(blocks) == 2
        assert {b.shape for _, b in blocks} == {(3, 4, 5)}

    @pytest.mark.parametrize("hint", ["Auto", "", None, "QQQ", "TZYX"])
    def test_unresolvable_hint_on_3d_raises(self, hint):
        image = np.zeros((2, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Cannot determine how to label"):
            list(sf._iter_dimension_blocks(image, hint))


class TestOtsuThresholdingDimensionOrders:
    """
    "Otsu Thresholding (semantic)" branches on the dimension-order hint to
    pick a per-frame, per-channel or global threshold.  Every branch must
    return a uint32 0/1 label image of the input shape.
    """

    @pytest.mark.parametrize(
        "hint,shape",
        [
            ("TYX", (3, 8, 8)),
            ("ZYX", (3, 8, 8)),
            ("TCYX", (2, 2, 8, 8)),
            ("TZYX", (2, 2, 8, 8)),
            ("ZCYX", (2, 2, 8, 8)),
            ("TZCYX", (2, 2, 2, 8, 8)),
            ("TCZYX", (2, 2, 2, 8, 8)),
        ],
    )
    def test_per_slice_branches(self, hint, shape):
        image = _blob_image(shape, seed=1)
        result = sf.otsu_thresholding(image, dimension_order=hint)
        assert result.shape == image.shape
        assert result.dtype == np.uint32
        assert set(np.unique(result)) <= {0, 1}
        assert result.max() == 1

    def test_per_slice_hint_thresholds_each_slice_separately(self):
        """A dim slice and a bright slice must not share one threshold."""
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        image[0, :4] = 10
        image[1, :4] = 5000
        result = sf.otsu_thresholding(image, dimension_order="TYX")
        # Both slices are foreground in their own upper half.
        assert result[0, :4].all() and not result[0, 4:].any()
        assert result[1, :4].all() and not result[1, 4:].any()

    def test_per_slice_hint_falls_back_for_2d_input(self):
        """
        A 3-axis hint on 2D data hits the shape fallback, which is the same
        two lines as the "YX/Auto" global-threshold branch below -- pin the
        exact array against an independently computed reference instead of
        just shape/dtype, so a wrong comparison operator or a stray dtype
        cast in that branch would be caught.
        """
        image = _blob_image((8, 8), seed=2)
        expected = (
            image > sf.skimage.filters.threshold_otsu(image)
        ).astype(np.uint32)
        result = sf.otsu_thresholding(image, dimension_order="TYX")
        np.testing.assert_array_equal(result, expected)

    def test_cyx_branch_labels_each_channel(self):
        image = _blob_image((2, 8, 8), seed=3)
        result = sf.otsu_thresholding(image, dimension_order="CYX")
        assert result.shape == image.shape
        assert result.dtype == np.uint32
        assert result.max() == 1

    def test_cyx_branch_thresholds_each_channel_separately(self):
        """
        A dim channel and a bright channel must not share one pooled
        threshold -- the same property
        ``test_per_slice_hint_thresholds_each_slice_separately`` pins for
        frames, checked here for the CYX branch. A regression to a single
        global threshold across channels would zero out the dim channel's
        foreground entirely while leaving shape/dtype/max unchanged, so the
        weaker checks above would not catch it.
        """
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        image[0, :4] = 10
        image[1, :4] = 5000
        result = sf.otsu_thresholding(image, dimension_order="CYX")
        assert result[0, :4].all() and not result[0, 4:].any()
        assert result[1, :4].all() and not result[1, 4:].any()

    def test_cyx_branch_falls_back_for_2d_input(self):
        """
        2D input under a CYX hint hits the same fallback computation as the
        global YX/Auto branch -- pin the exact array, not just shape/dtype.
        """
        image = _blob_image((8, 8), seed=4)
        expected = (
            image > sf.skimage.filters.threshold_otsu(image)
        ).astype(np.uint32)
        result = sf.otsu_thresholding(image, dimension_order="CYX")
        np.testing.assert_array_equal(result, expected)

    def test_auto_thresholds_globally(self):
        image = _blob_image((8, 8), seed=5)
        expected = (
            image > sf.skimage.filters.threshold_otsu(image)
        ).astype(np.uint32)
        result = sf.otsu_thresholding(image, dimension_order="Auto")
        np.testing.assert_array_equal(result, expected)


class TestOtsuThresholdingInstance:
    """
    "Otsu Thresholding (instance)" thresholds and then labels connected
    components, restarting label numbering for every independent T/C block.
    """

    def test_2d_returns_uint32_instance_labels(self):
        image = np.zeros((8, 8), dtype=np.uint16)
        image[1:3, 1:3] = 500
        image[5:7, 5:7] = 500
        result = sf.otsu_thresholding_instance(image)
        assert result.dtype == np.uint32
        assert result.max() == 2
        assert result[1, 1] != result[5, 5]

    def test_labels_restart_per_timepoint(self):
        image = np.zeros((2, 8, 8), dtype=np.uint16)
        image[:, 1:3, 1:3] = 500
        image[:, 5:7, 5:7] = 500
        result = sf.otsu_thresholding_instance(
            image, dimension_order="TYX"
        )
        assert result[0].max() == 2
        assert result[1].max() == 2

    def test_z_spanning_object_keeps_one_label(self):
        image = np.zeros((3, 8, 8), dtype=np.uint16)
        image[:, 2:5, 2:5] = 500
        result = sf.otsu_thresholding_instance(
            image, dimension_order="ZYX"
        )
        assert result.max() == 1
        assert len(np.unique(result[result > 0])) == 1


class TestBinaryToLabels:
    """
    "Binary to Labels" is the plain connected-component filter; it shares
    the block iterator with the Otsu instance filter.
    """

    def test_2d_binary_mask_gets_one_label_per_component(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        mask[5:8, 5:8] = 1
        result = sf.binary_to_labels(mask)
        assert result.dtype == np.uint32
        assert result.max() == 2
        assert result[0, 0] != result[6, 6]

    def test_3d_volume_keeps_one_label_across_z(self):
        mask = np.zeros((3, 8, 8), dtype=np.uint8)
        mask[:, 2:4, 2:4] = 1
        result = sf.binary_to_labels(mask, dimension_order="ZYX")
        assert result.max() == 1

    def test_labels_restart_per_timepoint(self):
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[:, 0:2, 0:2] = 1
        mask[:, 5:8, 5:8] = 1
        result = sf.binary_to_labels(mask, dimension_order="TYX")
        assert result[0].max() == 2
        assert result[1].max() == 2

    def test_unresolvable_hint_is_reported(self):
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="Cannot determine how to label"):
            sf.binary_to_labels(mask, dimension_order="Auto")


class TestSemanticToInstance:
    """
    "Semantic to Instance Segmentation" splits a multi-class mask into
    per-class connected components with globally unique ids.
    """

    def test_binary_mask_uses_the_plain_component_path(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        mask[5:8, 5:8] = 1
        result = sf._semantic_to_instance_block(mask)
        assert result.dtype == np.uint32
        assert result.max() == 2

    def test_multiclass_labels_are_unique_across_classes(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # class 1, component A
        mask[0:2, 5:8] = 1  # class 1, component B
        mask[5:8, 0:2] = 2  # class 2, component C
        result = sf._semantic_to_instance_block(mask)
        assert result.dtype == np.uint32
        ids = {
            int(result[0, 0]),
            int(result[0, 6]),
            int(result[6, 0]),
        }
        assert len(ids) == 3
        assert 0 not in ids
        assert result.max() == 3
        assert result[3, 3] == 0

    def test_class_present_only_as_background_is_skipped(self):
        """
        A class value whose mask is empty after the equality test cannot
        happen, but a single-class mask must still not leave a gap in the
        numbering.
        """
        mask = np.full((4, 4), 3, dtype=np.uint8)
        result = sf._semantic_to_instance_block(mask)
        assert result.max() == 1
        assert (result == 1).all()

    def test_registered_function_labels_per_timepoint(self):
        mask = np.zeros((2, 8, 8), dtype=np.uint8)
        mask[:, 0:2, 0:2] = 1
        mask[:, 5:8, 5:8] = 2
        result = sf.semantic_to_instance(mask, dimension_order="TYX")
        assert result.shape == mask.shape
        assert result.dtype == np.uint32
        assert result[0].max() == 2
        assert result[1].max() == 2

    def test_3d_multiclass_block(self):
        mask = np.zeros((2, 6, 6), dtype=np.uint8)
        mask[:, 0:2, 0:2] = 1
        mask[:, 4:6, 4:6] = 2
        result = sf.semantic_to_instance(mask, dimension_order="ZYX")
        # Both objects span Z, so each keeps a single id.
        assert result.max() == 2


class TestRollingBallGuard:
    """Rolling ball needs a YX plane to roll over."""

    def test_1d_input_is_rejected(self):
        with pytest.raises(ValueError, match="Rolling ball needs a 2D"):
            sf.rolling_ball_background(np.zeros(5, dtype=np.uint8), radius=5)


class TestClaheInMemoryBranches:
    """
    The in-memory CLAHE path picks between a thread pool and a single
    ``equalize_adapthist`` call depending on rank and leading-axis size.
    """

    def test_small_3d_stack_uses_the_single_shot_path(self, capsys):
        image = _blob_image((3, 16, 16), seed=6)
        result = sf.equalize_histogram(image, clip_limit=0.02, kernel_size=8)
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        printed = capsys.readouterr().out
        assert "Processing..." in printed
        assert "CLAHE processing complete!" in printed
        assert "Parallelizing" not in printed

    def test_large_3d_stack_is_parallelised(self, capsys):
        image = _blob_image((6, 16, 16), seed=7)
        result = sf.equalize_histogram(
            image, clip_limit=0.02, kernel_size=8, max_workers=2
        )
        assert result.shape == image.shape
        assert "Parallelizing CLAHE across 6 slices" in capsys.readouterr().out

    def test_4d_stack_is_parallelised_over_the_first_axis(self, capsys):
        image = _blob_image((2, 2, 16, 16), seed=8)
        result = sf.equalize_histogram(
            image, clip_limit=0.02, kernel_size=8, max_workers=2
        )
        assert result.shape == image.shape
        out = capsys.readouterr().out
        assert "Parallelizing CLAHE across 2 timepoints/slices" in out

    def test_auto_kernel_size_is_derived_in_memory(self, capsys):
        """Mirrors ``TestClaheDaskPath``'s auto-kernel-size test, but for
        the plain in-memory path, which computes it separately."""
        image = _blob_image((3, 64, 64), seed=35)
        sf.equalize_histogram(image, clip_limit=0.02, kernel_size=0)
        # 64 // 8 = 8, floored at 16, then forced odd -> 17.
        assert "kernel_size=17" in capsys.readouterr().out

    def test_float_input_is_preserved_in_memory(self):
        """Mirrors ``TestClaheDaskPath``'s float-dtype test: the in-memory
        ``store`` closure has its own separate integer/float branch."""
        rng = np.random.default_rng(36)
        image = rng.random((16, 16)).astype(np.float32)
        result = sf.equalize_histogram(image, clip_limit=0.02, kernel_size=5)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_max_workers_is_clamped(self):
        """Out-of-range worker counts are clamped, not passed through."""
        image = _blob_image((6, 16, 16), seed=9)
        low = sf.equalize_histogram(
            image, clip_limit=0.02, kernel_size=8, max_workers=-5
        )
        high = sf.equalize_histogram(
            image, clip_limit=0.02, kernel_size=8, max_workers=999
        )
        np.testing.assert_array_equal(low, high)


class TestClaheDaskPath:
    """
    A Dask input must return a lazy array (never a computed one) and must
    treat every T/C volume independently -- CLAHE is histogram based, so
    blending across a channel boundary changes the numbers.
    """

    @pytest.fixture()
    def da(self):
        return pytest.importorskip("dask.array")

    @pytest.mark.parametrize(
        "shape,chunks",
        [
            ((24, 24), (24, 24)),
            ((3, 24, 24), (3, 24, 24)),
            ((2, 3, 24, 24), (1, 3, 24, 24)),
            ((1, 2, 2, 24, 24), (1, 1, 2, 24, 24)),
        ],
    )
    def test_every_rank_returns_a_lazy_array(self, da, shape, chunks):
        image = _blob_image(shape, seed=10)
        lazy = da.from_array(image, chunks=chunks)
        result = sf.equalize_histogram(lazy, clip_limit=0.02, kernel_size=8)
        assert isinstance(result, da.Array)
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        computed = result.compute()
        assert computed.shape == image.shape
        assert computed.dtype == image.dtype
        assert not np.array_equal(computed, image)

    def test_channels_are_processed_independently(self, da):
        """
        The (t, c) volume of a full-stack run must equal the same volume
        processed on its own; anything else means chunks spanned T or C.
        """
        image = _blob_image((1, 2, 2, 24, 24), seed=11)
        full = sf.equalize_histogram(
            da.from_array(image, chunks=(1, 1, 2, 24, 24)),
            clip_limit=0.02,
            kernel_size=8,
        ).compute()
        one = sf.equalize_histogram(
            da.from_array(image[:, 1:2], chunks=(1, 1, 2, 24, 24)),
            clip_limit=0.02,
            kernel_size=8,
        ).compute()
        np.testing.assert_array_equal(full[:, 1:2], one)

    def test_float_input_keeps_its_dtype(self, da):
        rng = np.random.default_rng(12)
        image = rng.random((3, 24, 24)).astype(np.float32)
        result = sf.equalize_histogram(
            da.from_array(image, chunks=(3, 24, 24)),
            clip_limit=0.02,
            kernel_size=8,
        ).compute()
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_auto_kernel_size_is_derived_from_the_yx_plane(self, da, capsys):
        image = _blob_image((3, 64, 64), seed=13)
        sf.equalize_histogram(
            da.from_array(image, chunks=(3, 64, 64)),
            clip_limit=0.02,
            kernel_size=0,
        )
        # 64 // 8 = 8, floored at 16, then forced odd -> 17.
        assert "kernel_size=17" in capsys.readouterr().out

    def test_even_kernel_size_is_made_odd(self, da, capsys):
        image = _blob_image((3, 24, 24), seed=14)
        sf.equalize_histogram(
            da.from_array(image, chunks=(3, 24, 24)),
            clip_limit=0.02,
            kernel_size=8,
        )
        assert "kernel_size=9" in capsys.readouterr().out

    def test_missing_dask_is_reported_clearly(self, monkeypatch):
        """Without dask the lazy path must explain how to install it."""
        monkeypatch.setitem(sys.modules, "dask.array", None)
        fake = type(
            "FakeLazy",
            (),
            {"chunks": ((1,),), "map_blocks": lambda self: None},
        )()
        with pytest.raises(ImportError, match="Dask is required"):
            sf._equalize_histogram_dask(fake, 0.01, 8, 1)


class TestSpatialWindowSizeTuple:
    """
    ``_spatial_window_size_tuple`` decides which axes a local filter is
    allowed to span.  A ``size`` on a T or C axis blends unrelated
    frames/channels together, so those axes must always come back as 1.
    """

    @pytest.mark.parametrize(
        "hint,ndim,expected",
        [
            ("TYX", 3, (1, 7, 7)),
            ("CYX", 3, (1, 7, 7)),
            ("TCYX", 4, (1, 1, 7, 7)),
            ("TZYX", 4, (1, 1, 7, 7)),
            ("ZCYX", 4, (1, 1, 7, 7)),
            ("TZCYX", 5, (1, 1, 1, 7, 7)),
            ("TCZYX", 5, (1, 1, 1, 7, 7)),
            ("ZYX", 3, (7, 7, 7)),
            ("Auto", 3, (1, 7, 7)),
            ("Auto", 5, (1, 1, 1, 7, 7)),
            ("nonsense", 4, (1, 1, 7, 7)),
            ("Auto", 2, (7, 7)),
            ("YX", 2, (7, 7)),
            ("TYX", 2, (7, 7)),
        ],
    )
    def test_window_per_hint(self, hint, ndim, expected):
        assert sf._spatial_window_size_tuple(hint, ndim, 7) == expected

    def test_zyx_is_the_only_hint_that_spans_the_leading_axis(self):
        """ZYX is a real volume, so the window is allowed to cross Z."""
        assert sf._spatial_window_size_tuple("ZYX", 3, 5)[0] == 5
        assert sf._spatial_window_size_tuple("TYX", 3, 5)[0] == 1


class TestClampOverlapDepth:
    """
    ``map_overlap`` raises when the halo is at least as wide as a chunk;
    the clamp turns that crash into a warning and a smaller halo.
    """

    @pytest.fixture()
    def da(self):
        return pytest.importorskip("dask.array")

    def test_depth_within_the_chunk_is_left_alone(self, da):
        image = da.zeros((20, 20), chunks=(10, 10))
        depth = {0: 3, 1: 3}
        assert sf._clamp_overlap_depth(image, depth, "test") == depth

    def test_zero_depth_axes_are_skipped(self, da):
        image = da.zeros((4, 20, 20), chunks=(1, 10, 10))
        clamped = sf._clamp_overlap_depth(image, {0: 0, 1: 2, 2: 2}, "test")
        assert clamped == {0: 0, 1: 2, 2: 2}

    def test_oversized_depth_is_clamped_and_announced(self, da, capsys):
        image = da.zeros((20, 20), chunks=(4, 20))
        clamped = sf._clamp_overlap_depth(image, {0: 9, 1: 2}, "adaptive")
        assert clamped == {0: 3, 1: 2}
        printed = capsys.readouterr().out
        assert "adaptive overlap depth 9 on axis 0" in printed
        assert "clamping to 3" in printed

    def test_the_input_dict_is_not_mutated(self, da):
        image = da.zeros((20, 20), chunks=(4, 20))
        depth = {0: 9, 1: 2}
        sf._clamp_overlap_depth(image, depth, "adaptive")
        assert depth == {0: 9, 1: 2}


class TestAdaptiveThresholdBright:
    """
    "Adaptive Threshold (Bright Bias)" thresholds a local neighbourhood.
    The window must stay inside a single frame for T/C hints, and the Dask
    input must stay lazy.
    """

    @pytest.fixture()
    def da(self):
        return pytest.importorskip("dask.array")

    def test_non_uint8_stack_is_converted_per_block(self):
        image = _blob_image((2, 24, 24), seed=15, dtype=np.uint16, high=60000)
        result = sf.adaptive_threshold_bright(
            image, block_size=7, offset=-5.0, dimension_order="TYX"
        )
        assert result.dtype == np.uint8
        assert result.shape == image.shape
        assert set(np.unique(result)) <= {0, 255}

    def test_frames_are_thresholded_independently(self):
        """
        A per-frame hint means frame 1 must come out the same whether or
        not frame 0 is in the stack.
        """
        image = _blob_image((2, 24, 24), seed=16)
        stacked = sf.adaptive_threshold_bright(
            image, block_size=7, dimension_order="TYX"
        )
        alone = sf.adaptive_threshold_bright(
            image[1:], block_size=7, dimension_order="TYX"
        )
        np.testing.assert_array_equal(stacked[1:], alone)

    def test_zyx_hint_spans_the_volume(self):
        """
        ZYX is a real volume, so the neighbourhood is a 3D cube -- the
        result must match a single 3D ``threshold_local`` on the stack.
        """
        image = _blob_image((3, 24, 24), seed=17)
        as_ubyte = sf.skimage.img_as_ubyte(image)
        reference = (
            as_ubyte
            > sf.skimage.filters.threshold_local(
                as_ubyte, block_size=(7, 7, 7), offset=-5.0
            )
        ).astype(np.uint8) * 255

        volume = sf.adaptive_threshold_bright(
            image, block_size=7, offset=-5.0, dimension_order="ZYX"
        )
        np.testing.assert_array_equal(volume, reference)

    def test_dask_input_stays_lazy(self, da):
        image = _blob_image((2, 24, 24), seed=18, dtype=np.uint8, high=255)
        lazy = da.from_array(image, chunks=(1, 24, 24))
        result = sf.adaptive_threshold_bright(
            lazy, block_size=7, offset=-5.0, dimension_order="TYX"
        )
        assert isinstance(result, da.Array)
        computed = result.compute()
        assert computed.dtype == np.uint8
        assert computed.shape == image.shape
        assert set(np.unique(computed)) <= {0, 255}

    def test_dask_non_uint8_input_is_converted_lazily(self, da):
        image = _blob_image((2, 24, 24), seed=19, dtype=np.uint16, high=60000)
        lazy = da.from_array(image, chunks=(1, 24, 24))
        computed = sf.adaptive_threshold_bright(
            lazy, block_size=7, offset=-5.0, dimension_order="Auto"
        ).compute()
        assert computed.dtype == np.uint8
        assert set(np.unique(computed)) <= {0, 255}

    def test_dask_halo_is_clamped_for_small_chunks(self, da, capsys):
        image = _blob_image((24, 24), seed=20, dtype=np.uint8, high=255)
        lazy = da.from_array(image, chunks=(6, 6))
        computed = sf.adaptive_threshold_bright(
            lazy, block_size=35, dimension_order="YX"
        ).compute()
        assert computed.shape == image.shape
        assert "clamping to 5" in capsys.readouterr().out

    def test_even_block_size_is_made_odd_on_the_dask_path(self, da, capsys):
        image = _blob_image((2, 24, 24), seed=21, dtype=np.uint8, high=255)
        lazy = da.from_array(image, chunks=(1, 24, 24))
        sf.adaptive_threshold_bright(
            lazy, block_size=8, dimension_order="TYX"
        )
        assert "window=(1, 9, 9)" in capsys.readouterr().out


class TestRemoveSmallObjectsFallbacks:
    """
    "Remove Small Labels" streams from disk when the widget gives it a path,
    and otherwise works in memory.  These are the paths around that choice.
    """

    def test_unloaded_input_is_read_from_the_source_path(self, tmp_path):
        """
        ``skip_load`` leaves ``image`` as None; without an output folder the
        function has to load the file itself instead of silently returning.
        """
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[0, 0] = 1  # 1 pixel  -> removed
        labels[3:7, 3:7] = 2  # 16 pixels -> kept
        path = tmp_path / "labels.tif"
        tifffile.imwrite(str(path), labels)

        result = sf.remove_small_objects(
            None, min_size=2, _source_filepath=str(path)
        )

        assert isinstance(result, np.ndarray)
        assert set(np.unique(result)) == {0, 2}

    def test_old_skimage_signature_is_retried_with_min_size(
        self, monkeypatch
    ):
        """
        scikit-image < 0.26 has no ``max_size``; the retry must bump the
        threshold by one so "<= min_size" keeps meaning the same thing.
        """
        calls = []

        def fake_remove(image, **kwargs):
            calls.append(kwargs)
            if "max_size" in kwargs:
                raise TypeError("unexpected keyword argument 'max_size'")
            return image

        monkeypatch.setattr(
            sf.skimage.morphology, "remove_small_objects", fake_remove
        )
        labels = np.zeros((4, 4), dtype=np.uint32)
        sf.remove_small_objects(labels, min_size=100)

        assert calls == [{"max_size": 100}, {"min_size": 101}]

    def test_4d_input_is_processed_volume_by_volume(self):
        labels = np.zeros((2, 3, 8, 8), dtype=np.uint32)
        labels[0, 0, 0, 0] = 1  # 1 voxel -> removed
        labels[1, :, 2:6, 2:6] = 2  # 48 voxels -> kept
        result = sf.remove_small_objects(labels, min_size=4)
        assert result.shape == labels.shape
        assert set(np.unique(result)) == {0, 2}

    def test_declares_the_streaming_contract(self):
        assert sf.remove_small_objects.skip_load is True

    def test_streaming_path_writes_a_correctly_filtered_stack(self, tmp_path):
        """
        The full disk-to-disk streaming path, invoked through the public
        ``remove_small_objects`` wrapper exactly as the batch widget calls
        it (``image=None`` under ``skip_load``, with an output folder and
        suffix so the streaming branch -- not the in-memory one -- fires).
        A label that appears only in the last Z slice forces
        ``_stream_remove_small_labels``'s running bincount to grow
        mid-scan, so this also covers that branch.
        """
        labels = np.zeros((4, 6, 6), dtype=np.uint32)
        labels[0, 0:1, 0:1] = 1  # 1 voxel total -> removed
        labels[:, 2:5, 2:5] = 2  # 36 voxels -> kept
        labels[3, 0:2, 0:2] = 5  # 4 voxels, only in the last Z slice -> kept
        source = tmp_path / "labels.tif"
        tifffile.imwrite(str(source), labels)
        out_dir = tmp_path / "out"

        result_path = sf.remove_small_objects(
            None,
            min_size=2,
            _source_filepath=str(source),
            _output_folder=str(out_dir),
            _output_suffix="_rm",
        )

        assert result_path == str(out_dir / "labels_rm.tif")
        written = tifffile.imread(result_path)
        assert written.shape == labels.shape
        assert written.dtype == labels.dtype
        assert set(np.unique(written)) == {0, 2, 5}
        np.testing.assert_array_equal(written[3, 0:2, 0:2], 5)
        np.testing.assert_array_equal(written[:, 2:5, 2:5], 2)
        assert written[0, 0, 0] == 0


class TestStreamRemoveSmallLabelsGuards:
    """
    The streaming writer refuses input it cannot interpret rather than
    writing a corrupt stack.
    """

    def test_one_dimensional_source_is_rejected(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        source = tmp_path / "one.zarr"
        array = zarr.open_array(
            str(source), mode="w", shape=(6,), dtype="uint32", chunks=(6,)
        )
        array[:] = np.arange(6)

        with pytest.raises(ValueError, match="expected a 2D\\+ label image"):
            sf._stream_remove_small_labels(
                str(source), str(tmp_path / "out.tif"), 1
            )

    def test_negative_label_values_are_rejected(self, tmp_path):
        """
        ``np.bincount`` cannot count negative ids, so a signed image with
        negative values has to be reported instead of crashing deep inside.
        """
        labels = np.zeros((4, 4), dtype=np.int32)
        labels[0, 0] = -1
        source = tmp_path / "neg.tif"
        tifffile.imwrite(str(source), labels)

        with pytest.raises(ValueError, match="negative values"):
            sf._stream_remove_small_labels(
                str(source), str(tmp_path / "out.tif"), 1
            )


class TestResizeImageFixedYXBranches:
    """
    "Resize Image by YX Scale" only ever touches the trailing YX plane.
    These are the guard clauses and the lazy path around that.
    """

    def test_one_dimensional_input_is_rejected(self):
        """A 1D array has no YX plane to resize and must fail descriptively."""
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            sf.resize_image_fixed_yx(np.zeros(4, dtype=np.uint8))

    def test_explicit_yx_order_resizes_the_plane(self):
        image = _blob_image((32, 32), seed=22)
        result = sf.resize_image_fixed_yx(
            image, scale_factor=0.5, dim_order="YX"
        )
        assert result.shape == (16, 16)
        assert result.dtype == image.dtype

    def test_lowercase_dim_order_is_accepted(self):
        image = _blob_image((2, 32, 32), seed=23)
        result = sf.resize_image_fixed_yx(
            image, scale_factor=0.5, dim_order="tyx"
        )
        assert result.shape == (2, 16, 16)

    def test_dask_input_stays_lazy_and_is_rechunked(self, capsys):
        """
        The body carries its own Dask branch.  ``@chunked`` intercepts lazy
        inputs before the body ever sees one, so it is reached only through
        ``__wrapped__`` -- but it is the branch that guarantees Y and X are
        never split across blocks, which is what keeps a resized plane from
        being stitched together out of independently scaled tiles.
        """
        da = pytest.importorskip("dask.array")
        image = _blob_image((2, 3, 32, 32), seed=24)
        lazy = da.from_array(image, chunks=(1, 1, 16, 16))

        result = sf.resize_image_fixed_yx.__wrapped__(lazy, scale_factor=0.5)

        assert isinstance(result, da.Array)
        assert result.shape == (2, 3, 16, 16)
        # Y and X must be whole planes in every block, or a resized block
        # would be stitched from independently-scaled tiles.
        assert result.chunks[-2:] == ((16,), (16,))
        assert "Resize (dask)" in capsys.readouterr().out

        computed = result.compute()
        assert computed.dtype == image.dtype
        np.testing.assert_array_equal(
            computed[0, 0],
            sf.resize_image_fixed_yx(
                image[0, 0], scale_factor=0.5, dim_order="YX"
            ),
        )

    def test_dask_input_through_the_chunked_wrapper(self):
        """The registered wrapper handles a lazy input itself."""
        da = pytest.importorskip("dask.array")
        image = _blob_image((2, 3, 32, 32), seed=26)
        lazy = da.from_array(image, chunks=(1, 1, 16, 16))
        result = sf.resize_image_fixed_yx(lazy, scale_factor=0.5)
        assert np.asarray(result).shape == (2, 3, 16, 16)

    def test_non_positive_scale_factor_is_rejected(self):
        with pytest.raises(ValueError, match="scale_factor must be > 0"):
            sf.resize_image_fixed_yx(
                _blob_image((8, 8), seed=27), scale_factor=0, dim_order="YX"
            )

    def test_dim_order_not_ending_in_yx_is_rejected(self):
        with pytest.raises(
            ValueError, match="last two axes must be Y and X"
        ):
            sf.resize_image_fixed_yx(
                _blob_image((8, 8), seed=28),
                scale_factor=0.5,
                dim_order="XY",
            )

    def test_dim_order_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="incompatible with image.ndim"):
            sf.resize_image_fixed_yx(
                _blob_image((8, 8), seed=29),
                scale_factor=0.5,
                dim_order="ZYX",
            )

    def test_upscaling_disables_anti_aliasing(self, monkeypatch):
        """
        ``_resize_2d`` sets ``anti_aliasing = target < source_size`` per
        axis, so upscaling must pass ``anti_aliasing=False`` to skimage's
        ``resize`` and downscaling must pass ``True``. The function imports
        ``resize`` locally from ``skimage.transform`` on every call, so
        patching that module attribute is what the real call picks up.
        Only checking output shape (as this test used to) would still pass
        if ``anti_aliasing`` were hardcoded to either value, since shape is
        unaffected by it -- spy on the real call instead so the assertion
        is on what was actually passed, while the computation stays real.
        """
        import skimage.transform as sk_transform

        real_resize = sk_transform.resize
        seen_anti_aliasing = []

        def spy_resize(*args, **kwargs):
            seen_anti_aliasing.append(kwargs.get("anti_aliasing"))
            return real_resize(*args, **kwargs)

        monkeypatch.setattr(sk_transform, "resize", spy_resize)

        image = _blob_image((16, 16), seed=25)
        result = sf.resize_image_fixed_yx(
            image, scale_factor=2.0, dim_order="YX"
        )
        assert result.shape == (32, 32)
        assert seen_anti_aliasing == [False]

        seen_anti_aliasing.clear()
        downscaled = sf.resize_image_fixed_yx(
            image, scale_factor=0.5, dim_order="YX"
        )
        assert downscaled.shape == (8, 8)
        assert seen_anti_aliasing == [True]


def _write_v2_source(
    path,
    data,
    axes=("t", "c", "z", "y", "x"),
    scale=(1.0, 1.0, 2.0, 0.325, 0.325),
    array_path="0",
    omero_channels=None,
    n_levels=1,
):
    """
    Write a zarr-v2 OME layout (a ``.zattrs`` next to the array) -- the
    layout ``resize_zarr_native`` reads its metadata from.
    """
    zarr = pytest.importorskip("zarr")
    path = str(path)
    group = zarr.open_group(path, mode="w", zarr_format=2)
    parent = group
    name = array_path
    if "/" in array_path:
        head, name = array_path.rsplit("/", 1)
        parent = group.create_group(head)
    array = parent.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=(1,) * (data.ndim - 2) + data.shape[-2:],
    )
    array[:] = data

    types = {"t": "time", "c": "channel", "z": "space",
             "y": "space", "x": "space"}
    datasets = []
    for level in range(n_levels):
        datasets.append(
            {
                "path": array_path if level == 0 else f"{level}",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            s * (2**level if a in ("y", "x") else 1)
                            for a, s in zip(axes, scale)
                        ],
                    }
                ],
            }
        )
    attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": a, "type": types[a]} for a in axes
                ],
                "datasets": datasets,
            }
        ]
    }
    if omero_channels is not None:
        attrs["omero"] = {"channels": omero_channels}
    with open(os.path.join(path, ".zattrs"), "w") as handle:
        json.dump(attrs, handle)
    return path


class TestResizeZarrNativeMetadataReading:
    """
    Before it writes anything, "Resize Zarr by YX Scale (OME-Zarr native)"
    reads the source's OME metadata: it locates the channel axis, extracts a
    single channel when asked, remembers the level-0 pixel size and derives
    the output name.  Those steps run to completion on the installed stack
    (the write itself does not -- see the xfails in test_skimage_filters).
    """

    def test_channel_axis_is_found_and_extracted(self, tmp_path, capsys):
        data = _blob_image((1, 2, 2, 16, 16), seed=30)
        source = _write_v2_source(tmp_path / "src.zarr", data)

        sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            channel="1",
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        printed = capsys.readouterr().out
        assert "Extracting channel 1 (axis 1)" in printed
        # The C axis is gone from the shape the resize graph is built on.
        assert "(1, 2, 16, 16) → (1, 2, 8, 8)" in printed

    def test_double_zarr_extension_is_stripped_once(self, tmp_path, capsys):
        data = _blob_image((1, 2, 16, 16), seed=31)
        source = _write_v2_source(
            tmp_path / "stack.zarr.zarr",
            data,
            axes=("t", "z", "y", "x"),
            scale=(1.0, 2.0, 0.325, 0.325),
        )

        sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        printed = capsys.readouterr().out
        assert "stack_rz.zarr" in printed
        assert "stack.zarr_rz.zarr" not in printed

    def test_group_without_arrays_uses_the_dataset_path(
        self, tmp_path, capsys
    ):
        """
        A multiscale group whose levels live in a subgroup exposes no arrays
        at the root; the level path has to come from the metadata instead.
        """
        data = _blob_image((1, 2, 16, 16), seed=32)
        source = _write_v2_source(
            tmp_path / "nested.zarr",
            data,
            axes=("t", "z", "y", "x"),
            scale=(1.0, 2.0, 0.325, 0.325),
            array_path="lvl/0",
        )

        sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        assert "(1, 2, 16, 16) → (1, 2, 8, 8)" in capsys.readouterr().out

    def test_directory_without_zarr_suffix_is_still_recognised(
        self, tmp_path, capsys
    ):
        """A directory holding a ``.zattrs`` counts as zarr input."""
        data = _blob_image((1, 2, 16, 16), seed=33)
        source = _write_v2_source(
            tmp_path / "plain_dir",
            data,
            axes=("t", "z", "y", "x"),
            scale=(1.0, 2.0, 0.325, 0.325),
        )

        sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
        )

        assert "Resize (OME-Zarr native)" in capsys.readouterr().out

    def test_write_failure_falls_back_to_the_in_memory_resize(
        self, tmp_path, capsys
    ):
        """
        Every failure in the native path is swallowed; the contract is that
        the caller still gets a correctly resized array.
        """
        data = _blob_image((1, 2, 16, 16), seed=34)
        source = _write_v2_source(
            tmp_path / "src.zarr",
            data,
            axes=("t", "z", "y", "x"),
            scale=(1.0, 2.0, 0.325, 0.325),
        )

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
        )

        if isinstance(result, str):
            pytest.skip("native OME-Zarr write succeeded on this stack")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2, 8, 8)
        assert "falling back to skimage path" in capsys.readouterr().out


@pytest.fixture()
def zarr_v2_shims(monkeypatch):
    """
    Put back the two zarr-v2 entry points ``resize_zarr_native`` was written
    against but zarr 3 no longer provides in the same shape, so the write
    half of the function can be exercised at all:

    * ``zarr.storage.FSStore`` -- removed in zarr 3, mapped to ``LocalStore``;
    * ``zarr.open_group`` -- defaulted back to ``zarr_format=2`` so the
      output carries a ``.zattrs``, which the function edits afterwards;
    * ``zarr.open_array`` -- ome-zarr names its levels ``s0``/``s1`` while
      the function re-opens level ``0``, so the lookup is redirected.

    Only library entry points are replaced; every line under test is the
    module's own.  On the installed stack the unshimmed call bails out at
    the FSStore import -- that is what the xfails in ``test_skimage_filters``
    record.
    """
    zarr = pytest.importorskip("zarr")

    monkeypatch.setattr(
        zarr.storage,
        "FSStore",
        lambda path, **kwargs: zarr.storage.LocalStore(path),
        raising=False,
    )

    original_open_group = zarr.open_group

    def open_group(store=None, **kwargs):
        kwargs.setdefault("zarr_format", 2)
        return original_open_group(store, **kwargs)

    monkeypatch.setattr(zarr, "open_group", open_group)

    original_open_array = zarr.open_array

    def open_array(store=None, **kwargs):
        if isinstance(store, str) and not os.path.exists(store):
            alternative = os.path.join(
                os.path.dirname(store), "s" + os.path.basename(store)
            )
            if os.path.exists(alternative):
                store = alternative
        return original_open_array(store, **kwargs)

    monkeypatch.setattr(zarr, "open_array", open_array)
    return zarr


class TestResizeZarrNativeWriting:
    """
    The write half of "Resize Zarr by YX Scale (OME-Zarr native)": it writes
    an OME-Zarr, rewrites the level scales so a halved image reports twice
    the physical pixel size, and builds omero display windows by sampling
    the output.  See ``zarr_v2_shims`` for why the shims are needed.
    """

    def test_returns_the_written_path(self, tmp_path, zarr_v2_shims, capsys):
        data = _blob_image((1, 2, 2, 16, 16), seed=40)
        source = _write_v2_source(
            tmp_path / "src.zarr",
            data,
            omero_channels=[
                {"color": "00FF00", "label": "green"},
                {"color": "FF0000", "label": "red"},
            ],
        )

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        printed = capsys.readouterr().out
        assert "falling back to skimage path" not in printed
        assert result == str(tmp_path / "src_rz.zarr")
        assert os.path.isdir(result)

    def test_pixel_size_is_doubled_when_yx_is_halved(
        self, tmp_path, zarr_v2_shims, capsys
    ):
        """0.325 um pixels at half resolution are 0.65 um pixels."""
        data = _blob_image((1, 2, 2, 16, 16), seed=41)
        source = _write_v2_source(tmp_path / "src.zarr", data)

        sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        assert (
            "Updated coordinate transforms: level 0 Y/X scale = 0.6500"
            in capsys.readouterr().out
        )

    def test_omero_windows_are_built_from_the_written_data(
        self, tmp_path, zarr_v2_shims, capsys
    ):
        data = _blob_image((1, 2, 2, 16, 16), seed=42)
        source = _write_v2_source(
            tmp_path / "src.zarr",
            data,
            omero_channels=[
                {"color": "00FF00", "label": "green"},
                {"color": "FF0000", "label": "red"},
            ],
        )

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )
        assert "Wrote omero window metadata for 2 channel(s)" in (
            capsys.readouterr().out
        )

        written = zarr_v2_shims.open_group(result, mode="r")
        omero = dict(written.attrs)["omero"]
        channels = omero["channels"]
        assert len(channels) == 2
        # Colour and label are inherited from the source, per channel.
        assert [c["color"] for c in channels] == ["00FF00", "FF0000"]
        assert [c["label"] for c in channels] == ["green", "red"]
        assert all(c["active"] is True for c in channels)
        assert omero["version"] == "0.3"

        # The window must describe the data that was actually written.
        level0 = np.asarray(
            zarr_v2_shims.open_array(
                os.path.join(result, "s0"), mode="r"
            )[:]
        )
        for index, channel in enumerate(channels):
            plane = level0[:, index]
            assert channel["window"]["min"] == int(plane.min())
            assert channel["window"]["max"] == int(plane.max())
            assert channel["window"]["end"] <= int(plane.max())
            assert (
                channel["window"]["start"]
                >= channel["window"]["min"]
            )

    def test_single_channel_extraction_drops_the_channel_axis(
        self, tmp_path, zarr_v2_shims, capsys
    ):
        """
        Extracting one channel removes C from the data, from the axes list
        and from the per-level scale vector, and the omero entry inherits
        that channel's colour rather than the first one's.
        """
        data = _blob_image((1, 2, 2, 16, 16), seed=43)
        source = _write_v2_source(
            tmp_path / "src.zarr",
            data,
            omero_channels=[
                {"color": "00FF00", "label": "green"},
                {"color": "FF0000", "label": "red"},
            ],
        )

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            channel="1",
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        printed = capsys.readouterr().out
        assert "Extracting channel 1 (axis 1)" in printed
        assert "Wrote omero window metadata for 1 channel(s)" in printed

        written = zarr_v2_shims.open_group(result, mode="r")
        channels = dict(written.attrs)["omero"]["channels"]
        assert len(channels) == 1
        assert channels[0]["color"] == "FF0000"
        assert channels[0]["label"] == "red"

        level0 = zarr_v2_shims.open_array(
            os.path.join(result, "s0"), mode="r"
        )
        assert level0.shape == (1, 2, 8, 8)

    def test_default_colours_are_supplied_when_the_source_has_none(
        self, tmp_path, zarr_v2_shims
    ):
        data = _blob_image((1, 2, 2, 16, 16), seed=44)
        source = _write_v2_source(tmp_path / "src.zarr", data)

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        written = zarr_v2_shims.open_group(result, mode="r")
        channels = dict(written.attrs)["omero"]["channels"]
        assert [c["color"] for c in channels] == ["FFFFFF", "00FF00"]
        assert [c["label"] for c in channels] == ["Channel 0", "Channel 1"]

    def test_source_pyramid_depth_is_preserved(
        self, tmp_path, zarr_v2_shims
    ):
        data = _blob_image((1, 2, 2, 16, 16), seed=45)
        source = _write_v2_source(tmp_path / "src.zarr", data, n_levels=3)

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        written = zarr_v2_shims.open_group(result, mode="r")
        datasets = dict(written.attrs)["multiscales"][0]["datasets"]
        assert len(datasets) == 3

    def test_missing_ome_metadata_falls_back_to_axes_by_rank(
        self, tmp_path, zarr_v2_shims, capsys
    ):
        """
        A plain zarr with no multiscales still has to be written with a
        sensible axis list, inferred from the rank.
        """
        zarr = zarr_v2_shims
        data = _blob_image((1, 2, 16, 16), seed=46)
        source = str(tmp_path / "bare.zarr")
        group = zarr.open_group(source, mode="w", zarr_format=2)
        array = group.create_array(
            "0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 16, 16)
        )
        array[:] = data
        with open(os.path.join(source, ".zattrs"), "w") as handle:
            json.dump({}, handle)

        result = sf.resize_zarr_native(
            data,
            scale_factor=0.5,
            _source_filepath=source,
            _output_folder=str(tmp_path),
            _output_suffix="_rz",
        )

        assert "falling back to skimage path" not in capsys.readouterr().out
        written = zarr.open_group(result, mode="r")
        axes = dict(written.attrs)["multiscales"][0]["axes"]
        assert [a["name"] for a in axes] == ["t", "z", "y", "x"]
