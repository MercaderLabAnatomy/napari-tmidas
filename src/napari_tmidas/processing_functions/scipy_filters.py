# processing_functions/scipy_filters.py
"""
Processing functions that depend on SciPy.
"""
import numpy as np

try:
    from scipy import ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available, some processing functions will be disabled")

from napari_tmidas._registry import BatchProcessingRegistry

if SCIPY_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Resize Labels (Nearest, SciPy)",
        suffix="_scaled",
        description="Resize a label mask or label image by a scale factor using nearest-neighbor interpolation (scipy.ndimage.zoom, grid_mode=True) to preserve label integrity without shifting position.",
        parameters={
            "scale_factor": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 10.0,
                "description": "Factor by which to resize the label image (e.g., 0.8 for 80% size, 1.2 for 120% size). 1.0 means no resizing.",
            },
        },
    )
    def resize_labels(label_image: np.ndarray, scale_factor=1.0) -> np.ndarray:
        """
        Resize a label mask or label image by a scale factor using nearest-neighbor interpolation to preserve label integrity without shifting position.
        """
        scale_factor = float(scale_factor)
        if scale_factor == 1.0:
            return label_image
        from scipy.ndimage import zoom

        scaled_labels = zoom(
            label_image,
            zoom=scale_factor,
            order=0,  # Nearest-neighbor interpolation
            grid_mode=True,  # Prevents positional shift
            mode="nearest",
        ).astype(label_image.dtype)
        return scaled_labels

    @BatchProcessingRegistry.register(
        name="Gaussian Blur",
        suffix="_blurred",
        description="Apply Gaussian blur to the image",
        parameters={
            "sigma": {
                "type": float,
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Standard deviation for Gaussian kernel",
            }
        },
    )
    def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to the image
        """
        return ndimage.gaussian_filter(image, sigma=sigma)

    @BatchProcessingRegistry.register(
        name="Median Filter",
        suffix="_median",
        description="Apply median filter for noise reduction",
        parameters={
            "size": {
                "type": int,
                "default": 3,
                "min": 3,
                "max": 15,
                "description": "Size of the median filter window",
            }
        },
    )
    def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply median filter for noise reduction
        """
        return ndimage.median_filter(image, size=size)

else:
    # Export stub functions that raise ImportError when called
    def gaussian_blur(*args, **kwargs):
        raise ImportError(
            "SciPy is not available. Please install scipy to use this function."
        )

    def median_filter(*args, **kwargs):
        raise ImportError(
            "SciPy is not available. Please install scipy to use this function."
        )
