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
