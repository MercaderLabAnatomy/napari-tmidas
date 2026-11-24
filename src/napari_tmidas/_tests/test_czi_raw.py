"""
Tests for reading CZI files whose subblocks libCZI cannot decode.

ZEN camera streaming writes uncompressed pixels but tags each subblock with
compression value 1000, and libCZI then fails every read with "The method
or operation is not implemented." (ZEISS/libczi#187). The fixtures here are
real CZI files written by pylibCZIrw and then patched to compression 1000
the same way, so the fallback is checked against libCZI's own output for
the identical pixels.
"""

import shutil
import struct
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import pytest
from pylibCZIrw import czi as pyczi

from napari_tmidas import _czi_raw
from napari_tmidas import _file_conversion as fc

T, Z, H, W = 2, 3, 5, 7


def _subblock_positions(path):
    """(directory entry offset, subblock segment offset) of every subblock."""
    with open(path, "rb") as f:
        f.seek(32 + 52)
        (directory,) = struct.unpack("<q", f.read(8))
        f.seek(directory + 32)
        (count,) = struct.unpack("<i", f.read(4))
        entry = directory + 32 + 128
        positions = []
        for _ in range(count):
            f.seek(entry)
            raw = f.read(32)
            (segment,) = struct.unpack_from("<q", raw, 6)
            (ndim,) = struct.unpack_from("<i", raw, 28)
            positions.append((entry, segment))
            entry += 32 + 20 * ndim
    return positions


def _with_compression(src, dst, compression):
    """Copy ``src`` and retag every subblock with ``compression``, as ZEN does."""
    shutil.copy(src, dst)
    with open(dst, "r+b") as f:
        for entry, segment in _subblock_positions(dst):
            for offset in (entry + 18, segment + 32 + 16 + 18):
                f.seek(offset)
                f.write(struct.pack("<i", compression))
    return str(dst)


def _write_czi(path, planes):
    """Write ``{(t, z, scene, x, y): 2-D array}`` with pylibCZIrw."""
    with pyczi.create_czi(str(path), exist_ok=True) as writer:
        for (t, z, scene, x, y), data in planes.items():
            writer.write(
                data,
                plane={"T": t, "Z": z, "C": 0},
                location=(x, y),
                scene=scene,
            )
    return str(path)


def _rng_plane(rng, shape=(H, W)):
    return rng.integers(0, 65535, shape, dtype=np.uint16)


@pytest.fixture
def timelapse(tmp_path):
    """A single-scene T x Z timelapse at a stage offset, plus its mode-1000 copy."""
    rng = np.random.default_rng(0)
    planes = {
        (t, z, 0, 100, 200): _rng_plane(rng)
        for t in range(T)
        for z in range(Z)
    }
    original = _write_czi(tmp_path / "original.czi", planes)
    streamed = _with_compression(original, tmp_path / "streamed.czi", 1000)
    expected = np.stack(
        [
            np.stack([planes[(t, z, 0, 100, 200)] for z in range(Z)])
            for t in range(T)
        ]
    )[:, None]  # T, C, Z, Y, X
    return original, streamed, expected


def test_libczi_rejects_compression_1000(timelapse):
    """Guards the premise: if this starts passing upstream, the fallback is idle."""
    _, streamed, _ = timelapse
    with pyczi.open_czi(streamed) as doc, pytest.raises(RuntimeError) as err:
        doc.read(plane={"T": 0, "Z": 0, "C": 0})
    assert _czi_raw.is_libczi_not_implemented(err.value)


def test_fallback_returns_the_written_pixels(timelapse):
    original, streamed, expected = timelapse

    from_libczi = fc.CZILoader.load_series(original, 0)
    from_fallback = fc.CZILoader.load_series(streamed, 0)

    assert from_fallback.dtype == np.uint16
    np.testing.assert_array_equal(from_libczi, expected)
    np.testing.assert_array_equal(from_fallback, expected)


def test_fallback_dask_path(timelapse, monkeypatch):
    _, streamed, expected = timelapse
    monkeypatch.setattr(fc.CZILoader, "DASK_THRESHOLD_GB", 0.0)

    data = fc.CZILoader.load_series(streamed, 0)

    assert isinstance(data, da.Array)
    np.testing.assert_array_equal(data.compute(), expected)


def test_metadata_still_available_for_streamed_file(timelapse):
    original, streamed, _ = timelapse
    assert fc.CZILoader.get_metadata(streamed, 0) == fc.CZILoader.get_metadata(
        original, 0
    )


def test_raw_reader_matches_libczi_on_scenes_mosaics_and_gaps(tmp_path):
    """Two scenes, a two-tile mosaic plane and a missing plane, read both ways."""
    rng = np.random.default_rng(1)
    planes = {
        # scene 0: two side-by-side tiles in every plane
        **{
            (t, z, 0, 0, 0): _rng_plane(rng)
            for t in range(T)
            for z in range(Z)
        },
        **{
            (t, z, 0, W, 0): _rng_plane(rng)
            for t in range(T)
            for z in range(Z)
        },
        # scene 1: offset tile, and plane (t=1, z=2) never acquired
        **{
            (t, z, 1, 50, 40): _rng_plane(rng)
            for t in range(T)
            for z in range(Z)
            if (t, z) != (1, 2)
        },
    }
    path = _write_czi(tmp_path / "mosaic.czi", planes)
    reader = _czi_raw.CziRawReader(path)

    with pyczi.open_czi(path) as doc:
        for scene, rect in doc.scenes_bounding_rectangle.items():
            roi = (rect.x, rect.y, rect.w, rect.h)
            for t in range(T):
                for z in range(Z):
                    plane = {"T": t, "Z": z, "C": 0}
                    np.testing.assert_array_equal(
                        reader.read_plane(plane, roi, scene),
                        doc.read(plane=plane, scene=scene).squeeze(),
                        err_msg=f"scene={scene} t={t} z={z}",
                    )
    assert (
        reader.read_plane({"T": 1, "Z": 2, "C": 0}, (50, 40, W, H), 1).max()
        == 0
    )


def test_payload_size_mismatch_is_refused(timelapse, tmp_path):
    """A mode-1000 payload that is not exactly w*h*bytes is never guessed at."""
    _, streamed, _ = timelapse
    broken = tmp_path / "broken.czi"
    shutil.copy(streamed, broken)
    _, segment = _subblock_positions(broken)[0]
    with open(broken, "r+b") as f:
        f.seek(segment + 32 + 8)  # DataSize
        f.write(struct.pack("<q", H * W * 2 - 2))

    with pytest.raises(fc.FileFormatError, match="expected 70"):
        fc.CZILoader.load_series(str(broken), 0)


def test_unknown_compression_is_refused(timelapse, tmp_path):
    original, _, _ = timelapse
    odd = _with_compression(original, tmp_path / "odd.czi", 1001)

    with pytest.raises(fc.FileFormatError, match=r"\[1001\]"):
        fc.CZILoader.load_series(odd, 0)


def test_other_libczi_errors_are_not_masked(monkeypatch, tmp_path):
    """Only the not-implemented error triggers the fallback."""

    class Doc:
        total_bounding_box = {"T": (0, 1), "Z": (0, 1), "C": (0, 1)}
        scenes_bounding_rectangle = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, **_):
            raise RuntimeError("Illegal data detected at offset 42")

    monkeypatch.setattr(fc.pyczi, "open_czi", lambda _path: Doc())
    monkeypatch.setattr(
        fc, "CziRawReader", lambda _path: pytest.fail("fallback must not run")
    )
    with pytest.raises(fc.FileFormatError, match="Illegal data"):
        fc.CZILoader.load_series(str(tmp_path / "x.czi"), 0)


# --------------------------------------------------------------------------
# Frame interval
# --------------------------------------------------------------------------


def _build_czi(path, subblocks):
    """
    Write a minimal CZI by hand: file header, subblocks, subblock directory.

    pylibCZIrw cannot write subblock metadata, which is where ZEN puts each
    plane's AcquisitionTime. Each item of ``subblocks`` is a dict with
    ``coords`` (non-spatial start indices), ``data`` (2-D uint16), and
    optional ``time`` (ISO string) and ``compression``.
    """

    def entry(sb, file_position):
        height, width = sb["data"].shape
        dims = [("X", 0, width), ("Y", 0, height)] + [
            (name, index, 1) for name, index in sb["coords"].items()
        ]
        out = b"DV" + struct.pack(
            "<iqiiBB4si",
            1,
            file_position,
            0,
            sb.get("compression", 1000),
            0,
            0,
            b"\0" * 4,
            len(dims),
        )
        for name, start, size in dims:
            out += struct.pack(
                "<4siifi",
                name.encode().ljust(4, b"\0"),
                start,
                size,
                0.0,
                size,
            )
        return out

    def segment(name, payload):
        return (
            name.ljust(16, b"\0")
            + struct.pack("<qq", len(payload), len(payload))
            + payload
        )

    blob = bytearray(segment(b"ZISRAWFILE", b"\0" * 512))
    entries = []
    for sb in subblocks:
        position = len(blob)
        dv = entry(sb, position)
        metadata = b""
        if sb.get("time"):
            metadata = (
                "<METADATA><Tags><AcquisitionTime>"
                f"{sb['time']}</AcquisitionTime></Tags></METADATA>"
            ).encode()
        pixels = sb["data"].astype("<u2").tobytes()
        header = struct.pack("<iiq", len(metadata), 0, len(pixels)) + dv
        header = header.ljust(max(256, 16 + len(dv)), b"\0")
        blob += segment(b"ZISRAWSUBBLOCK", header + metadata + pixels)
        entries.append(dv)

    directory_position = len(blob)
    directory = struct.pack("<i", len(entries)) + b"\0" * 124
    blob += segment(b"ZISRAWDIR", directory + b"".join(entries))
    struct.pack_into("<q", blob, 32 + 52, directory_position)
    path.write_bytes(bytes(blob))
    return str(path)


def _zen_time(seconds):
    """ZEN-style AcquisitionTime: seven fractional digits and a 'Z'."""
    stamp = datetime(2022, 12, 8, 12, 31, 45) + timedelta(seconds=seconds)
    fraction = f"{seconds % 1:.7f}"[1:]
    return stamp.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"


def _timelapse_subblocks(starts, z_planes=2, z_step=0.35):
    """One subblock per (t, z); plane z of frame t starts at starts[t]."""
    return [
        {
            "coords": {"Z": z, "C": 0, "T": t},
            "data": np.full((2, 3), 10 * t + z, dtype=np.uint16),
            "time": _zen_time(start + z * z_step),
        }
        for t, start in enumerate(starts)
        for z in range(z_planes)
    ]


def test_handcrafted_streaming_file_reads_back(tmp_path):
    path = _build_czi(tmp_path / "hand.czi", _timelapse_subblocks([0, 11]))
    reader = _czi_raw.CziRawReader(path)

    plane = reader.read_plane({"T": 1, "Z": 1, "C": 0}, (0, 0, 3, 2))

    np.testing.assert_array_equal(plane, np.full((2, 3), 11))


def test_frame_interval_is_measured_from_acquisition_times(tmp_path):
    # Stacks run back to back: frames every ~11.1 s, not a set 10 s.
    path = _build_czi(
        tmp_path / "timed.czi",
        _timelapse_subblocks([0.0, 11.1, 22.2, 33.4]),
    )

    timing = _czi_raw.measure_frame_interval(path)

    assert timing.frames == 4
    assert timing.median == pytest.approx(11.1, abs=1e-6)
    assert timing.minimum == pytest.approx(11.1, abs=1e-6)
    assert timing.maximum == pytest.approx(11.2, abs=1e-6)


def test_frame_interval_needs_two_timestamped_frames(tmp_path):
    single = _build_czi(tmp_path / "one.czi", _timelapse_subblocks([0.0]))
    untimed = _timelapse_subblocks([0.0, 10.0])
    for sb in untimed:
        sb.pop("time")
    untimed = _build_czi(tmp_path / "untimed.czi", untimed)

    assert _czi_raw.measure_frame_interval(single) is None
    assert _czi_raw.measure_frame_interval(untimed) is None


@pytest.mark.parametrize(
    ("value", "unit", "seconds"),
    [("10", "s", 10.0), ("500", "ms", 0.5), ("1.5", "min", 90.0)],
)
def test_set_interval_is_read_from_zen_metadata(value, unit, seconds):
    xml = (
        '<TimeSeriesSetup IsActivated="true"><Duration><Cycles>60</Cycles>'
        "</Duration><Interval><TimeSpan>"
        f"<Value>{value}</Value><DefaultUnitFormat>{unit}</DefaultUnitFormat>"
        "</TimeSpan></Interval></TimeSeriesSetup>"
    )
    assert fc.CZILoader._extract_time_interval_from_xml(xml) == pytest.approx(
        seconds
    )


def test_no_set_interval_without_a_time_series_setup():
    xml = "<Interval><Distance><Value>3.84E-06</Value></Distance></Interval>"
    assert fc.CZILoader._extract_time_interval_from_xml(xml) is None


class _XmlDoc:
    def __init__(self, set_seconds):
        self.raw_metadata = (
            "<TimeSeriesSetup><Interval><TimeSpan>"
            f"<Value>{set_seconds}</Value>"
            "<DefaultUnitFormat>s</DefaultUnitFormat>"
            "</TimeSpan></Interval></TimeSeriesSetup>"
        )


def test_measured_interval_wins_and_mismatch_is_reported(tmp_path, capsys):
    path = _build_czi(
        tmp_path / "slow.czi", _timelapse_subblocks([0.0, 14.5, 29.0])
    )

    meta = fc.CZILoader._time_interval_metadata(path, _XmlDoc(10))

    assert meta["time_interval"] == pytest.approx(14.5, abs=1e-6)
    assert meta["time_interval_set"] == 10.0
    assert "not every 10 s as set in ZEN" in capsys.readouterr().out


def test_matching_interval_is_not_reported_as_a_mismatch(tmp_path, capsys):
    path = _build_czi(
        tmp_path / "ok.czi", _timelapse_subblocks([0.0, 10.1, 20.2])
    )

    fc.CZILoader._time_interval_metadata(path, _XmlDoc(10))

    assert "not every" not in capsys.readouterr().out


def test_set_interval_is_used_without_timestamps(timelapse):
    original, _, _ = timelapse  # written by pylibCZIrw: no AcquisitionTime

    meta = fc.CZILoader._time_interval_metadata(original, _XmlDoc(7.5))

    assert meta == {"time_interval": 7.5, "time_interval_set": 7.5}


def test_metadata_without_timing_information_has_no_interval(timelapse):
    _, streamed, _ = timelapse

    meta = fc.CZILoader.get_metadata(streamed, 0)

    # pylibCZIrw writes neither a time series setup nor timestamps.
    assert "time_interval" not in meta
