# processing_functions/skimage_filters.py
"""
Processing functions that depend on scikit-image.
"""
import numpy as np

try:
    import skimage.exposure
    import skimage.filters
    import skimage.morphology

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print(
        "scikit-image not available, some processing functions will be disabled"
    )

from napari_tmidas._registry import BatchProcessingRegistry

if SKIMAGE_AVAILABLE:

    # Equalize histogram
    @BatchProcessingRegistry.register(
        name="Equalize Histogram",
        suffix="_equalized",
        description="Equalize histogram of image",
    )
    def equalize_histogram(
        image: np.ndarray, clip_limit: float = 0.01
    ) -> np.ndarray:
        """
        Equalize histogram of image
        """

        return skimage.exposure.equalize_hist(image)

    # simple otsu thresholding
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (semantic)",
        suffix="_otsu_semantic",
        description="Threshold image using Otsu's method",
    )
    def otsu_thresholding(image: np.ndarray) -> np.ndarray:
        """
        Threshold image using Otsu's method
        """

        image = skimage.img_as_ubyte(image)  # convert to 8-bit
        thresh = skimage.filters.threshold_otsu(image)
        return (image > thresh).astype(np.uint32)

    # instance segmentation
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (instance)",
        suffix="_otsu_labels",
        description="Threshold image using Otsu's method",
    )
    def otsu_thresholding_instance(image: np.ndarray) -> np.ndarray:
        """
        Threshold image using Otsu's method
        """
        image = skimage.img_as_ubyte(image)  # convert to 8-bit
        thresh = skimage.filters.threshold_otsu(image)
        return skimage.measure.label(image > thresh).astype(np.uint32)

    # simple thresholding
    @BatchProcessingRegistry.register(
        name="Manual Thresholding (8-bit)",
        suffix="_thresh",
        description="Threshold image using a fixed threshold",
        parameters={
            "threshold": {
                "type": int,
                "default": 128,
                "min": 0,
                "max": 255,
                "description": "Threshold value",
            },
        },
    )
    def simple_thresholding(
        image: np.ndarray, threshold: int = 128
    ) -> np.ndarray:
        """
        Threshold image using a fixed threshold
        """
        # convert to 8-bit
        image = skimage.img_as_ubyte(image)
        return image > threshold

    # remove small objects
    @BatchProcessingRegistry.register(
        name="Remove Small Objects",
        suffix="_rm_small",
        description="Remove small objects from the binary image",
        parameters={
            "min_size": {
                "type": int,
                "default": 100,
                "min": 1,
                "max": 100000,
                "description": "Remove labels smaller than: ",
            },
        },
    )
    def remove_small_objects(
        image: np.ndarray, min_size: int = 100
    ) -> np.ndarray:
        """
        Remove small objects from the binary image
        """
        return skimage.morphology.remove_small_objects(
            image, min_size=min_size
        )
