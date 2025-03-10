# processing_functions/basic.py
"""
Basic image processing functions that don't require additional dependencies.
"""
import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry


@BatchProcessingRegistry.register(
    name="Min-Max Normalization",
    suffix="_normalized",
    description="Normalize image values to range [0, 1] using min-max scaling",
)
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization
    """
    if image.min() == image.max():
        return np.zeros_like(image, dtype=float)
    return (image - image.min()) / (image.max() - image.min())


@BatchProcessingRegistry.register(
    name="Contrast Stretch",
    suffix="_contrast",
    description="Stretch the contrast by clipping percentiles and rescaling",
    parameters={
        "low_percentile": {
            "type": float,
            "default": 2.0,
            "min": 0.0,
            "max": 49.0,
            "description": "Low percentile to clip",
        },
        "high_percentile": {
            "type": float,
            "default": 98.0,
            "min": 51.0,
            "max": 100.0,
            "description": "High percentile to clip",
        },
    },
)
def contrast_stretch(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """
    Stretch contrast by clipping percentiles
    """
    p_low = np.percentile(image, low_percentile)
    p_high = np.percentile(image, high_percentile)

    # Clip and normalize
    image_clipped = np.clip(image, p_low, p_high)
    if p_high == p_low:
        return np.zeros_like(image, dtype=float)
    return (image_clipped - p_low) / (p_high - p_low)
