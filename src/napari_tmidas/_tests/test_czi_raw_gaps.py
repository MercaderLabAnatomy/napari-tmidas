"""
The raw CZI reader refuses files it cannot read safely as uncompressed gray
pixels, and frame-interval measurement gives up rather than guessing.

Files are written by hand (header, subblocks, directory) so each field can
be set to a value ZEN would never write; test_czi_raw.py covers real files.
"""

import struct

import numpy as np
import pytest

from napari_tmidas import _czi_raw
from napari_tmidas._czi_raw import CziRawReader, CziRawReadError

_DIRECTORY_POSITION = 32 + 52  # file header segment data + offset


def _segment(name, payload):
    return (
        name.ljust(16, b"\0")
        + struct.pack("<qq", len(payload), len(payload))
        + payload
    )


def _entry(sb, file_position):
    height, width = sb["data"].shape
    stored = sb.get("stored")  # (stored_w, stored_h) for pyramid tiles
    stored_w, stored_h = stored if stored else (width, height)
    x, y = sb.get("xy", (0, 0))
    dims = [("X", x, width, stored_w), ("Y", y, height, stored_h)]
    if sb.get("drop_y"):
        dims = dims[:1]
    dims += [(name, index, 1, 1) for name, index in sb["coords"].items()]
    out = sb.get("schema", b"DV") + struct.pack(
        "<iqiiBB4si",
        sb.get("pixel_type", 1),
        file_position,
        sb.get("file_part", 0),
        sb.get("compression", 1000),
        0,
        0,
        b"\0" * 4,
        len(dims),
    )
    for name, start, size, stored_size in dims:
        out += struct.pack(
            "<4siifi", name.encode().ljust(4, b"\0"), start, size, 0.0, stored_size
        )
    return out


def _build(path, subblocks):
    """Write ``subblocks`` (dicts: coords, data, and optional overrides)."""
    blob = bytearray(_segment(b"ZISRAWFILE", b"\0" * 512))
    entries = []
    for sb in subblocks:
        position = len(blob)
        dv = _entry(sb, position + sb.get("position_shift", 0))
        metadata = b""
        if sb.get("time"):
            metadata = (
                "<METADATA><Tags><AcquisitionTime>"
                f"{sb['time']}</AcquisitionTime></Tags></METADATA>"
            ).encode()
        pixels = sb["data"].astype("<u2").tobytes()
        metadata_size = sb.get("metadata_size", len(metadata))
        header = struct.pack("<iiq", metadata_size, 0, len(pixels)) + dv
        header = header.ljust(max(256, 16 + len(dv)), b"\0")
        blob += _segment(b"ZISRAWSUBBLOCK", header + metadata + pixels)
        entries.append(dv)

    directory_position = len(blob)
    directory = struct.pack("<i", len(entries)) + b"\0" * 124
    blob += _segment(b"ZISRAWDIR", directory + b"".join(entries))
    struct.pack_into("<q", blob, _DIRECTORY_POSITION, directory_position)
    path.write_bytes(bytes(blob))
    return str(path)


def _plane(value=7, shape=(2, 3)):
    return np.full(shape, value, dtype=np.uint16)


def _sb(**overrides):
    sb = {"coords": {"T": 0, "Z": 0, "C": 0}, "data": _plane()}
    sb.update(overrides)
    return sb


def _time(seconds):
    return f"2022-12-08T12:31:{int(seconds):02d}.{int(seconds % 1 * 1e7):07d}Z"


# --------------------------------------------------------------------------
# Files that are not, or not safely, raw CZI
# --------------------------------------------------------------------------


def test_a_file_that_is_not_czi_is_refused(tmp_path):
    path = tmp_path / "fake.czi"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 200)

    with pytest.raises(CziRawReadError, match="not a CZI file"):
        CziRawReader(str(path))


def test_a_directory_pointer_to_the_wrong_segment_is_refused(tmp_path):
    path = tmp_path / "moved.czi"
    _build(path, [_sb()])
    blob = bytearray(path.read_bytes())
    struct.pack_into("<q", blob, _DIRECTORY_POSITION, 0)  # the file header
    path.write_bytes(bytes(blob))

    with pytest.raises(CziRawReadError, match="no subblock directory"):
        CziRawReader(str(path))


def test_a_directory_entry_of_an_unknown_schema_is_refused(tmp_path):
    path = _build(tmp_path / "schema.czi", [_sb(schema=b"DE")])

    with pytest.raises(CziRawReadError, match="directory entry schema"):
        CziRawReader(path)


def test_a_subblock_without_a_y_extent_is_refused(tmp_path):
    path = _build(tmp_path / "noy.czi", [_sb(drop_y=True)])

    with pytest.raises(CziRawReadError, match="without X/Y"):
        CziRawReader(path)


def test_a_file_holding_only_pyramid_tiles_is_refused(tmp_path):
    path = _build(tmp_path / "pyramid.czi", [_sb(stored=(1, 1))])

    with pytest.raises(CziRawReadError, match="no full-resolution"):
        CziRawReader(path)


@pytest.mark.parametrize(
    "pixel_types",
    [
        [8],  # Bgr24: three samples per pixel
        [0, 1],  # Gray8 and Gray16 in one file
    ],
)
def test_colour_or_mixed_pixel_types_are_refused(tmp_path, pixel_types):
    blocks = [
        _sb(coords={"T": 0, "Z": z, "C": 0}, pixel_type=pt)
        for z, pt in enumerate(pixel_types)
    ]
    path = _build(tmp_path / "pixels.czi", blocks)

    with pytest.raises(CziRawReadError, match="pixel type"):
        CziRawReader(path)


def test_a_multi_part_file_is_refused(tmp_path):
    path = _build(tmp_path / "part.czi", [_sb(file_part=1)])

    with pytest.raises(CziRawReadError, match="multi-part"):
        CziRawReader(path)


def test_a_subblock_pointer_to_the_wrong_segment_is_refused(tmp_path):
    # The directory says the tile starts 32 bytes into its own segment.
    path = _build(tmp_path / "offset.czi", [_sb(position_shift=32)])
    reader = CziRawReader(path)

    with pytest.raises(CziRawReadError, match="no subblock segment"):
        reader.read_plane({"T": 0, "Z": 0, "C": 0}, (0, 0, 3, 2))


def test_pixel_data_past_the_end_of_the_file_is_refused(tmp_path):
    # A metadata size larger than the file sends the pixel read past EOF.
    path = _build(tmp_path / "trunc.czi", [_sb(metadata_size=10**6)])
    reader = CziRawReader(path)

    with pytest.raises(CziRawReadError, match="truncated"):
        reader.read_plane({"T": 0, "Z": 0, "C": 0}, (0, 0, 3, 2))


# --------------------------------------------------------------------------
# Reading regions of a mosaic
# --------------------------------------------------------------------------


def test_a_region_reads_only_the_tiles_it_overlaps(tmp_path):
    left = _sb(data=_plane(1), xy=(0, 0))
    right = _sb(data=_plane(2), xy=(3, 0))
    # A broken tile far away must not be touched for this region.
    far = _sb(data=_plane(3), xy=(100, 100), position_shift=32)
    reader = CziRawReader(_build(tmp_path / "mosaic.czi", [left, right, far]))

    plane = reader.read_plane({"T": 0, "Z": 0, "C": 0}, (2, 0, 3, 2))

    np.testing.assert_array_equal(plane, [[1, 2, 2], [1, 2, 2]])


# --------------------------------------------------------------------------
# Frame interval
# --------------------------------------------------------------------------


def test_pyramid_tiles_do_not_count_as_frames(tmp_path):
    blocks = [
        _sb(coords={"T": 0, "Z": 0, "C": 0}, time=_time(0)),
        _sb(coords={"T": 1, "Z": 0, "C": 0}, time=_time(10.5)),
        # A downsampled copy of a third frame, as written for overviews.
        _sb(coords={"T": 2, "Z": 0, "C": 0}, time=_time(99), stored=(1, 1)),
    ]

    timing = _czi_raw.measure_frame_interval(_build(tmp_path / "p.czi", blocks))

    assert timing.frames == 2
    assert timing.median == pytest.approx(10.5, abs=1e-6)


def test_frames_without_a_common_plane_have_no_interval(tmp_path):
    # Frame 0 only has Z=0 and frame 1 only Z=1: no plane to compare.
    blocks = [
        _sb(coords={"T": 0, "Z": 0, "C": 0}, time=_time(0)),
        _sb(coords={"T": 1, "Z": 1, "C": 0}, time=_time(10)),
    ]

    assert _czi_raw.measure_frame_interval(_build(tmp_path / "z.czi", blocks)) is None


def test_a_timestamp_in_an_unreadable_subblock_is_not_guessed(tmp_path):
    blocks = [
        _sb(coords={"T": 0, "Z": 0, "C": 0}, time=_time(0)),
        _sb(coords={"T": 1, "Z": 0, "C": 0}, time=_time(10), position_shift=32),
    ]

    assert _czi_raw.measure_frame_interval(_build(tmp_path / "b.czi", blocks)) is None
