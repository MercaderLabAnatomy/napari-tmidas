"""
Direct reader for uncompressed CZI subblocks that libCZI refuses to decode.

ZEN camera streaming stores uncompressed pixel data but tags every subblock
with the raw compression value 1000, which lies in the "system-specific"
range of the CZI specification. libCZI maps that value to
``CompressionMode::Invalid`` and any plane read fails with
"The method or operation is not implemented." (ZEISS/libczi#187,
ZEISS/pylibczirw#128).

This module reads such files straight from the subblock directory. It only
accepts subblocks whose payload is provably raw pixels: compression 0 or
1000, a single-channel gray pixel type, a full-resolution (pyramid layer 0)
tile, and a data size of exactly width * height * bytes-per-pixel. Anything
else raises :class:`CziRawReadError` rather than guessing at pixel data.

The on-disk layout follows the ZISRAW (CZI) file format specification.
"""

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# Raw compression values whose payload is plain pixel data.
RAW_COMPRESSION_MODES = frozenset({0, 1000})

# CZI pixel types that map to a single numpy plane.
GRAY_PIXEL_DTYPES = {
    0: np.dtype("<u1"),  # Gray8
    1: np.dtype("<u2"),  # Gray16
    2: np.dtype("<f4"),  # Gray32Float
}

_SEGMENT_HEADER_SIZE = 32  # Id[16], AllocatedSize int64, UsedSize int64
# Offset of SubBlockDirectoryPosition inside the file header segment data:
# Major, Minor, Reserved1, Reserved2 (4 x int32), PrimaryFileGuid[16],
# FileGuid[16], FilePart int32.
_SUBBLOCK_DIRECTORY_POSITION_OFFSET = 52
_DIRECTORY_HEADER_SIZE = 128  # EntryCount int32 + Reserved[124]
_ENTRY_FIXED_SIZE = 32  # DirectoryEntryDV without its dimension entries
_DIMENSION_ENTRY_SIZE = 20
_SUBBLOCK_FIXED_SIZE = 16  # MetadataSize, AttachmentSize int32, DataSize int64
_SUBBLOCK_MIN_HEADER_SIZE = 256


class CziRawReadError(ValueError):
    """The file cannot be read safely as raw, uncompressed subblocks."""


@dataclass(frozen=True)
class RawSubBlock:
    """One entry of the CZI subblock directory."""

    file_position: int
    file_part: int
    pixel_type: int
    compression: int
    x: int
    y: int
    width: int
    height: int
    stored_width: int
    stored_height: int
    entry_size: int
    m_index: int = 0
    # Start index of every non-spatial dimension (T, Z, C, S, ...).
    coords: Dict[str, int] = field(default_factory=dict)

    @property
    def is_layer0(self) -> bool:
        """Full-resolution tile, as opposed to a pyramid level."""
        return (self.stored_width, self.stored_height) == (
            self.width,
            self.height,
        )

    def matches(self, plane: Dict[str, int], scene: Optional[int]) -> bool:
        # A subblock without a given dimension belongs to every index of it.
        for dim, index in plane.items():
            if self.coords.get(dim, index) != index:
                return False
        return scene is None or self.coords.get("S", scene) == scene


def is_libczi_not_implemented(error: BaseException) -> bool:
    """True for the libCZI error raised on an unsupported compression."""
    return isinstance(error, RuntimeError) and "not implemented" in str(error)


def read_subblock_directory(filepath: str) -> List[RawSubBlock]:
    """Parse the subblock directory of a CZI file."""
    with open(filepath, "rb") as f:
        header = f.read(_SEGMENT_HEADER_SIZE + 64)
        if not header.startswith(b"ZISRAWFILE"):
            raise CziRawReadError("not a CZI file (missing ZISRAWFILE)")
        (directory_position,) = struct.unpack_from(
            "<q",
            header,
            _SEGMENT_HEADER_SIZE + _SUBBLOCK_DIRECTORY_POSITION_OFFSET,
        )

        f.seek(directory_position)
        segment = f.read(_SEGMENT_HEADER_SIZE)
        if not segment.startswith(b"ZISRAWDIR"):
            raise CziRawReadError(
                f"no subblock directory at offset {directory_position}"
            )
        (used_size,) = struct.unpack_from("<q", segment, 24)
        data = f.read(used_size)

    (count,) = struct.unpack_from("<i", data, 0)
    position = _DIRECTORY_HEADER_SIZE
    subblocks = []
    for _ in range(count):
        if data[position : position + 2] != b"DV":
            raise CziRawReadError(
                f"unexpected directory entry schema at {position}"
            )
        (
            pixel_type,
            file_position,
            file_part,
            compression,
            _pyramid_type,
        ) = struct.unpack_from("<iqiiB", data, position + 2)
        (dimension_count,) = struct.unpack_from("<i", data, position + 28)

        spatial = {}
        coords = {}
        dim_position = position + _ENTRY_FIXED_SIZE
        for _ in range(dimension_count):
            name, start, size, _start_coordinate, stored_size = (
                struct.unpack_from("<4siifi", data, dim_position)
            )
            name = name.rstrip(b"\0").decode("ascii")
            if name in ("X", "Y"):
                spatial[name] = (start, size, stored_size)
            else:
                coords[name] = start
            dim_position += _DIMENSION_ENTRY_SIZE

        if "X" not in spatial or "Y" not in spatial:
            raise CziRawReadError("subblock without X/Y extent")
        entry_size = (
            _ENTRY_FIXED_SIZE + dimension_count * _DIMENSION_ENTRY_SIZE
        )
        subblocks.append(
            RawSubBlock(
                file_position=file_position,
                file_part=file_part,
                pixel_type=pixel_type,
                compression=compression,
                x=spatial["X"][0],
                y=spatial["Y"][0],
                width=spatial["X"][1],
                height=spatial["Y"][1],
                stored_width=spatial["X"][2],
                stored_height=spatial["Y"][2],
                entry_size=entry_size,
                m_index=coords.pop("M", 0),
                coords=coords,
            )
        )
        position += entry_size
    return subblocks


class CziRawReader:
    """
    Read planes of a CZI file whose subblocks are all raw pixel data.

    Construction parses the subblock directory and raises
    :class:`CziRawReadError` unless every full-resolution subblock can be
    read as raw pixels. Planes are composed like libCZI does without a
    zoom: tiles in M order onto a zero (black) background, clipped to the
    requested region.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.subblocks = [
            sb for sb in read_subblock_directory(filepath) if sb.is_layer0
        ]
        if not self.subblocks:
            raise CziRawReadError("no full-resolution subblocks")

        compressions = {sb.compression for sb in self.subblocks}
        unsupported = sorted(compressions - RAW_COMPRESSION_MODES)
        if unsupported:
            raise CziRawReadError(
                f"compression mode(s) {unsupported} are not raw pixel data"
            )
        pixel_types = {sb.pixel_type for sb in self.subblocks}
        if (
            len(pixel_types) != 1
            or not pixel_types <= GRAY_PIXEL_DTYPES.keys()
        ):
            raise CziRawReadError(
                f"pixel type(s) {sorted(pixel_types)} are not supported; "
                "only a single gray pixel type (Gray8, Gray16, Gray32Float) is"
            )
        if any(sb.file_part != 0 for sb in self.subblocks):
            raise CziRawReadError("multi-part CZI files are not supported")

        self.compression_modes = compressions
        self.dtype = GRAY_PIXEL_DTYPES[pixel_types.pop()]

    def read_plane(
        self,
        plane: Dict[str, int],
        roi: Tuple[int, int, int, int],
        scene: Optional[int] = None,
    ) -> np.ndarray:
        """Return the (height, width) plane of ``roi`` = (x, y, w, h)."""
        roi_x, roi_y, roi_w, roi_h = roi
        out = np.zeros((roi_h, roi_w), dtype=self.dtype)
        tiles = sorted(
            (sb for sb in self.subblocks if sb.matches(plane, scene)),
            key=lambda sb: sb.m_index,
        )
        if not tiles:
            return out

        with open(self.filepath, "rb") as f:
            for sb in tiles:
                x0 = max(sb.x, roi_x)
                y0 = max(sb.y, roi_y)
                x1 = min(sb.x + sb.width, roi_x + roi_w)
                y1 = min(sb.y + sb.height, roi_y + roi_h)
                if x0 >= x1 or y0 >= y1:
                    continue
                tile = self._read_tile(f, sb)
                out[y0 - roi_y : y1 - roi_y, x0 - roi_x : x1 - roi_x] = tile[
                    y0 - sb.y : y1 - sb.y, x0 - sb.x : x1 - sb.x
                ]
        return out

    def _read_tile(self, f, sb: RawSubBlock) -> np.ndarray:
        f.seek(sb.file_position)
        header = f.read(_SEGMENT_HEADER_SIZE + _SUBBLOCK_FIXED_SIZE)
        if not header.startswith(b"ZISRAWSUBBLOCK"):
            raise CziRawReadError(
                f"no subblock segment at offset {sb.file_position}"
            )
        metadata_size, _attachment_size, data_size = struct.unpack_from(
            "<iiq", header, _SEGMENT_HEADER_SIZE
        )
        expected = sb.width * sb.height * self.dtype.itemsize
        if data_size != expected:
            raise CziRawReadError(
                f"subblock at offset {sb.file_position} holds {data_size} "
                f"bytes, expected {expected} for an uncompressed "
                f"{sb.width}x{sb.height} {self.dtype} tile"
            )
        header_size = max(
            _SUBBLOCK_MIN_HEADER_SIZE, _SUBBLOCK_FIXED_SIZE + sb.entry_size
        )
        f.seek(
            sb.file_position
            + _SEGMENT_HEADER_SIZE
            + header_size
            + metadata_size
        )
        tile = np.fromfile(f, dtype=self.dtype, count=sb.width * sb.height)
        if tile.size != sb.width * sb.height:
            raise CziRawReadError(
                f"subblock at offset {sb.file_position} is truncated"
            )
        return tile.reshape(sb.height, sb.width)


# --------------------------------------------------------------------------
# Frame timing
# --------------------------------------------------------------------------

_ACQUISITION_TIME = re.compile(
    rb"<AcquisitionTime>\s*(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?"
)


@dataclass(frozen=True)
class FrameTiming:
    """Measured interval between consecutive timepoints, in seconds."""

    median: float
    minimum: float
    maximum: float
    frames: int


def _acquisition_time(f, sb: RawSubBlock) -> Optional[float]:
    """AcquisitionTime of one subblock as POSIX seconds, or None."""
    f.seek(sb.file_position)
    header = f.read(_SEGMENT_HEADER_SIZE + _SUBBLOCK_FIXED_SIZE)
    if not header.startswith(b"ZISRAWSUBBLOCK"):
        return None
    (metadata_size,) = struct.unpack_from("<i", header, _SEGMENT_HEADER_SIZE)
    if metadata_size <= 0:
        return None
    header_size = max(
        _SUBBLOCK_MIN_HEADER_SIZE, _SUBBLOCK_FIXED_SIZE + sb.entry_size
    )
    f.seek(sb.file_position + _SEGMENT_HEADER_SIZE + header_size)
    match = _ACQUISITION_TIME.search(f.read(metadata_size))
    if not match:
        return None
    # ZEN writes 7 fractional digits, more than datetime parses; the time
    # zone is the same for every subblock of a file, so it cancels out.
    whole = datetime.strptime(match.group(1).decode(), "%Y-%m-%dT%H:%M:%S")
    fraction = float(match.group(2) or 0.0)
    return whole.replace(tzinfo=timezone.utc).timestamp() + fraction


def measure_frame_interval(filepath: str) -> Optional[FrameTiming]:
    """
    Interval between timepoints from the subblocks' acquisition times.

    ZEN records the interval that was *set*, but when a Z stack takes
    longer than that interval the stacks simply run back to back, so the
    real interval can be much longer. Each subblock carries the time its
    plane was acquired; this compares the same plane position (lowest
    Z/C/scene/tile) across timepoints.

    Works for any CZI compression, since only subblock metadata is read.
    Returns None for fewer than two timepoints or missing timestamps.
    """
    first_per_t: Dict[int, RawSubBlock] = {}
    for sb in read_subblock_directory(filepath):
        if not sb.is_layer0:
            continue
        t = sb.coords.get("T", 0)
        key = (sorted(sb.coords.items()), sb.m_index)
        best = first_per_t.get(t)
        if best is None or key < (sorted(best.coords.items()), best.m_index):
            first_per_t[t] = sb
    if len(first_per_t) < 2:
        return None

    # Only planes at the same position within the stack are comparable.
    reference = {
        dim: index
        for dim, index in first_per_t[min(first_per_t)].coords.items()
        if dim != "T"
    }
    subblocks = [
        first_per_t[t]
        for t in sorted(first_per_t)
        if {d: i for d, i in first_per_t[t].coords.items() if d != "T"}
        == reference
    ]
    if len(subblocks) < 2:
        return None

    with open(filepath, "rb") as f:
        times = [_acquisition_time(f, sb) for sb in subblocks]
    if any(t is None for t in times):
        return None
    intervals = np.diff(np.asarray(times, dtype=float))
    return FrameTiming(
        median=float(np.median(intervals)),
        minimum=float(intervals.min()),
        maximum=float(intervals.max()),
        frames=len(times),
    )
