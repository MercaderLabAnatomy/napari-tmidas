import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from napari_tmidas.processing_functions.ome_output_utils import (
    write_labels_with_source_metadata,
)


def _write_source_zattrs(source_path, n_levels=3):
    datasets = []
    for level in range(n_levels):
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, float(2**level), float(2**level)],
                    }
                ],
            }
        )

    attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ],
        "omero": {
            "version": "0.3",
            "channels": [
                {
                    "label": "labels",
                    "color": "FFFFFF",
                    "window": {
                        "start": 0,
                        "end": 10,
                        "min": 0,
                        "max": 10,
                    },
                }
            ],
        },
    }

    with open(source_path / ".zattrs", "w", encoding="utf-8") as f:
        json.dump(attrs, f)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_write_labels_with_source_metadata_preserves_pyramid_and_omero(tmp_path):
    pytest.importorskip("ome_zarr")

    source_path = tmp_path / "source.zarr"
    source_path.mkdir()
    _write_source_zattrs(source_path, n_levels=3)

    labels = np.zeros((8, 64, 64), dtype=np.uint32)
    labels[:, 10:20, 10:20] = 3

    output_path = tmp_path / "out_labels.zarr"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=str(source_path),
        output_path=str(output_path),
        output_format="zarr",
        dim_order="ZYX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    out_zattrs = output_path / ".zattrs"
    out_zarr_json = output_path / "zarr.json"
    if out_zattrs.exists():
        with open(out_zattrs, encoding="utf-8") as f:
            out_attrs = json.load(f)
    else:
        with open(out_zarr_json, encoding="utf-8") as f:
            out_doc = json.load(f)
        out_attrs = out_doc.get("attributes", {})

    out_multiscales = out_attrs.get("multiscales", [])
    if not out_multiscales and isinstance(out_attrs.get("ome"), dict):
        out_multiscales = out_attrs["ome"].get("multiscales", [])

    assert out_multiscales
    out_datasets = out_multiscales[0].get("datasets", [])
    assert len(out_datasets) == 3
    assert out_attrs.get("omero", {}).get("version") == "0.3"


def test_write_labels_with_source_metadata_writes_ome_tiff(tmp_path):
    labels = np.zeros((32, 32), dtype=np.uint32)
    labels[5:10, 5:10] = 7

    output_path = tmp_path / "labels.ome.tif"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=None,
        output_path=str(output_path),
        output_format="tiff",
        dim_order="YX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    with tifffile.TiffFile(output_path) as tif:
        assert tif.is_ome
        arr = tif.asarray()
        assert arr.dtype == np.uint32
        assert arr.shape == labels.shape


def test_write_labels_with_source_metadata_tiff_failure_is_atomic(
    tmp_path, monkeypatch
):
    labels = np.ones((4, 8, 8), dtype=np.uint32)
    output_path = tmp_path / "labels.ome.tif"

    def _failing_imwrite(path, *args, **kwargs):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(tifffile, "imwrite", _failing_imwrite)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        write_labels_with_source_metadata(
            labels=labels,
            source_path=None,
            output_path=str(output_path),
            output_format="tiff",
            dim_order="ZYX",
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_write_labels_with_source_metadata_streams_zarr_array_to_ome_tiff(
    tmp_path,
):
    labels_path = tmp_path / "labels_cache.zarr"
    labels = zarr.open_array(
        str(labels_path),
        mode="w",
        shape=(2, 3, 16, 16),
        chunks=(1, 1, 16, 16),
        dtype=np.uint32,
    )
    labels[:] = np.arange(2 * 3 * 16 * 16, dtype=np.uint32).reshape(
        2, 3, 16, 16
    )

    output_path = tmp_path / "labels_streamed.ome.tif"
    returned = write_labels_with_source_metadata(
        labels=labels,
        source_path=None,
        output_path=str(output_path),
        output_format="tiff",
        dim_order="TZYX",
    )

    assert returned == str(output_path)
    assert output_path.exists()

    with tifffile.TiffFile(output_path) as tif:
        assert tif.is_ome
        arr = tif.asarray()
        assert arr.dtype == np.uint32
        assert arr.shape == (2, 3, 16, 16)


class TestZarrOutputStreaming:
    """
    Writing a store-backed array as OME-Zarr must not read it all in.

    ome_zarr's write_image() calls da.from_array() on anything that is not
    already a Dask array, and that auto-chunks to ~128 MiB with no regard for
    the array's own on-disk layout.  Peak for the write is
    (dask threads) x (chunk size), so the chunking is the whole ballgame:
    left to auto it is a fixed ~4 GB on a large stack, and on a buffer chunked
    one timepoint deep (57x2720x2720 uint32 = 1.7 GB per chunk) it would be
    32 x 1.7 GB.  Both are bounded, and both are too big.
    """

    @staticmethod
    def _source(tmp_path, shape):
        import json

        import zarr

        path = tmp_path / "src.zarr"
        # Deliberately v2: a legacy source whose metadata still has to be
        # readable and copyable onto the v3 output.
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        group.create_array(
            "0", shape=shape, chunks=(1, 1) + shape[2:], dtype="uint16"
        )
        (path / ".zattrs").write_text(
            json.dumps(
                {
                    "multiscales": [
                        {
                            "version": "0.4",
                            "axes": [
                                {"name": n, "type": "space"} for n in "tzyx"
                            ],
                            "datasets": [{"path": "0"}],
                        }
                    ]
                }
            )
        )
        return str(path)

    @staticmethod
    def _buffer(tmp_path, shape, chunks):
        import zarr

        # v3: mirrors the scratch buffer convpaint streams into.
        buf = zarr.open_array(
            str(tmp_path / "buf.zarr"),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="uint32",
            zarr_format=3,
        )
        rng = np.random.default_rng(0)
        for t in range(shape[0]):
            buf[t] = rng.integers(0, 3, shape[1:], dtype=np.uint32)
        return buf

    def test_peak_does_not_grow_with_the_stack(self, tmp_path):
        """Doubling the stack at a fixed chunk size must not cost more RAM."""
        pytest.importorskip("zarr")
        pytest.importorskip("dask")
        import tracemalloc

        peaks, dense = {}, {}
        for n_timepoints in (10, 40):
            case = tmp_path / f"t{n_timepoints}"
            case.mkdir()
            shape = (n_timepoints, 4, 128, 128)
            dense[n_timepoints] = int(np.prod(shape)) * 4  # uint32
            buf = self._buffer(case, shape, (1, 1) + shape[2:])
            source = self._source(case, shape)

            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                write_labels_with_source_metadata(
                    buf, source, str(case / "out.zarr"), "zarr", "TZYX"
                )
                peaks[n_timepoints] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        # What "does not grow with the stack" has to mean here is that the
        # *marginal* peak per extra timepoint is ~0.  Comparing the two peaks
        # directly is the obvious formulation and a bad one: the constant
        # overhead of the write dominates the measurement and varies with the
        # dask/zarr build, so one healthy streaming write measured a flat
        # 5.3 MB on the dev machine and 0.9 -> 1.4 MB on macOS CI -- and a
        # `peaks[40] < peaks[10] * 1.5` bound failed on the latter purely from
        # that noise.  Bounding peak as a fraction of the stack is no better,
        # for the same reason: 0.5x passes on CI and fails on the dev machine.
        #
        # Marginal cost is immune to both.  A write that materialised the
        # stack spends ~1 byte of peak per extra byte of data; streaming
        # spends ~0 (measured -0.03 dev, 0.06 macOS CI), so 0.25 keeps ~4x
        # headroom over the worst seen while still failing loudly if this
        # ever starts holding the stack.
        marginal = (peaks[40] - peaks[10]) / (dense[40] - dense[10])
        assert marginal < 0.25, (
            f"peak grew {marginal:.2f} bytes per extra byte of stack "
            f"({peaks[10]/1e6:.1f} -> {peaks[40]/1e6:.1f} MB as the stack "
            f"went {dense[10]/1e6:.1f} -> {dense[40]/1e6:.1f} MB); a "
            f"streaming write should spend ~0"
        )

    def test_peak_tracks_the_arrays_own_chunking(self, tmp_path):
        """
        A smaller on-disk chunk must actually buy a smaller peak.

        This is what fails if the Dask wrap is dropped: write_image's auto
        chunking ignores the array's layout, so both cases cost the same and
        the caller has no way to bound the write.
        """
        pytest.importorskip("zarr")
        pytest.importorskip("dask")
        import tracemalloc

        shape = (24, 8, 128, 128)
        peaks = {}
        for label, chunks in (
            ("thin", (1, 1) + shape[2:]),
            ("deep", (1, 8) + shape[2:]),
        ):
            case = tmp_path / label
            case.mkdir()
            buf = self._buffer(case, shape, chunks)
            source = self._source(case, shape)

            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                write_labels_with_source_metadata(
                    buf, source, str(case / "out.zarr"), "zarr", "TZYX"
                )
                peaks[label] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        assert peaks["thin"] < peaks["deep"] * 0.7, (
            f"chunking made no difference: thin {peaks['thin']/1e6:.1f} MB "
            f"vs deep {peaks['deep']/1e6:.1f} MB"
        )

    def test_output_is_ome_zarr_regardless_of_buffer_shape(self, tmp_path):
        """The written store is a conformant group, not a bare array."""
        pytest.importorskip("zarr")
        import json

        import zarr

        shape = (6, 4, 64, 64)
        buf = self._buffer(tmp_path, shape, (1, 1) + shape[2:])
        source = self._source(tmp_path, shape)
        out = str(tmp_path / "out.zarr")
        write_labels_with_source_metadata(buf, source, out, "zarr", "TZYX")

        doc = json.loads((Path(out) / "zarr.json").read_text())
        attrs = doc.get("attributes", {})
        multiscales = (attrs.get("ome") or {}).get(
            "multiscales"
        ) or attrs.get("multiscales")
        assert multiscales, "output carries no multiscales metadata"
        assert [a["name"] for a in multiscales[0]["axes"]] == list("tzyx")

        group = zarr.open_group(out, mode="r")
        key = list(group.array_keys())[0]
        np.testing.assert_array_equal(np.asarray(group[key]), np.asarray(buf))
