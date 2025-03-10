# processing_functions/skimage_filters.py
"""
Processing functions that depend on scikit-image.
"""
import numpy as np

try:
    import skimage.exposure
    import skimage.filters

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print(
        "scikit-image not available, some processing functions will be disabled"
    )

from napari_tmidas._registry import BatchProcessingRegistry

if SKIMAGE_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Adaptive Histogram Equalization",
        suffix="_clahe",
        description="Enhance contrast using Contrast Limited Adaptive Histogram Equalization",
        parameters={
            "kernel_size": {
                "type": int,
                "default": 8,
                "min": 4,
                "max": 64,
                "description": "Size of local region for histogram equalization",
            },
            "clip_limit": {
                "type": float,
                "default": 0.01,
                "min": 0.001,
                "max": 0.1,
                "description": "Clipping limit for contrast enhancement",
            },
        },
    )
    def adaptive_hist_eq(
        image: np.ndarray, kernel_size: int = 8, clip_limit: float = 0.01
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        """
        # CLAHE expects image in [0, 1] range
        img_norm = skimage.exposure.rescale_intensity(image, out_range=(0, 1))
        return skimage.exposure.equalize_adapthist(
            img_norm, kernel_size=kernel_size, clip_limit=clip_limit
        )

    @BatchProcessingRegistry.register(
        name="Edge Detection",
        suffix="_edges",
        description="Detect edges using Sobel filter",
    )
    def edge_detection(image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Sobel filter
        """
        return skimage.filters.sobel(image)
