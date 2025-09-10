try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Conditional imports - these might fail on Windows without dependencies
try:
    from ._file_selector import file_selector
except ImportError:
    file_selector = None

try:
    from ._reader import napari_get_reader
except ImportError:
    napari_get_reader = None

try:
    from ._sample_data import make_sample_data
except ImportError:
    make_sample_data = None

try:
    from ._writer import write_multiple, write_single_image
except ImportError:
    write_multiple = None
    write_single_image = None

try:
    from ._label_inspection import label_inspector_widget
except ImportError:
    label_inspector_widget = None

try:
    from ._roi_colocalization import roi_colocalization_analyzer
except ImportError:
    roi_colocalization_analyzer = None

try:
    from ._crop_anything import batch_crop_anything_widget
except ImportError:
    batch_crop_anything_widget = None

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "file_selector",
    "label_inspector_widget",
    "batch_crop_anything_widget",
    "roi_colocalization_analyzer",
)
