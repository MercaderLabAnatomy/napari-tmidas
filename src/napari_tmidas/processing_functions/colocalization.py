"""
ROI Colocalization Processing Function

This module provides a function for batch processing to analyze colocalization
between multiple labeled regions in image stacks.

The function accepts a multi-channel input image with labeled regions and
returns statistics about their colocalization.
"""

import numpy as np
from skimage import measure


def get_nonzero_labels(image):
    """Get unique, non-zero labels from an image."""
    mask = image != 0
    labels = np.unique(image[mask])
    return [int(x) for x in labels]


def count_unique_nonzero(array, mask):
    """Count unique non-zero values in array where mask is True."""
    unique_vals = np.unique(array[mask])
    count = len(unique_vals)

    # Remove 0 from count if present
    if count > 0 and 0 in unique_vals:
        count -= 1

    return count


def calculate_coloc_size(
    image_c1, image_c2, label_id, mask_c2=None, image_c3=None
):
    """Calculate the size of colocalization between channels."""
    # Create mask for current ROI
    mask = image_c1 == int(label_id)

    # Handle mask_c2 parameter
    if mask_c2 is not None:
        if mask_c2:
            # sizes where c2 is present
            mask = mask & (image_c2 != 0)
            target_image = image_c3 if image_c3 is not None else image_c2
        else:
            # sizes where c2 is NOT present
            mask = mask & (image_c2 == 0)
            if image_c3 is None:
                # If no image_c3, just return count of mask pixels
                return np.count_nonzero(mask)
            target_image = image_c3
    else:
        target_image = image_c2

    # Calculate size of overlap
    masked_image = target_image * mask
    size = np.count_nonzero(masked_image)

    return int(size)


def process_single_roi(
    label_id,
    image_c1,
    image_c2,
    image_c3=None,
    get_sizes=False,
    roi_sizes=None,
):
    """Process a single ROI for colocalization analysis."""
    # Create masks once
    mask_roi = image_c1 == label_id
    mask_c2 = image_c2 != 0

    # Calculate counts
    c2_in_c1_count = count_unique_nonzero(image_c2, mask_roi & mask_c2)

    # Build the result dictionary
    result = {"label_id": int(label_id), "ch2_in_ch1_count": c2_in_c1_count}

    # Add size information if requested
    if get_sizes:
        if roi_sizes is None:
            roi_sizes = {}
            # Calculate sizes for current label only
            area = np.sum(mask_roi)
            roi_sizes[label_id] = area

        size = roi_sizes.get(int(label_id), 0)
        c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id)

        result.update({"ch1_size": size, "ch2_in_ch1_size": c2_in_c1_size})

    # Handle third channel if present
    if image_c3 is not None:
        mask_c3 = image_c3 != 0

        # Calculate third channel statistics
        c3_in_c2_in_c1_count = count_unique_nonzero(
            image_c3, mask_roi & mask_c2 & mask_c3
        )
        c3_not_in_c2_but_in_c1_count = count_unique_nonzero(
            image_c3, mask_roi & ~mask_c2 & mask_c3
        )

        result.update(
            {
                "ch3_in_ch2_in_ch1_count": c3_in_c2_in_c1_count,
                "ch3_not_in_ch2_but_in_ch1_count": c3_not_in_c2_but_in_c1_count,
            }
        )

        # Add size information for third channel if requested
        if get_sizes:
            c3_in_c2_in_c1_size = calculate_coloc_size(
                image_c1, image_c2, label_id, mask_c2=True, image_c3=image_c3
            )
            c3_not_in_c2_but_in_c1_size = calculate_coloc_size(
                image_c1, image_c2, label_id, mask_c2=False, image_c3=image_c3
            )

            result.update(
                {
                    "ch3_in_ch2_in_ch1_size": c3_in_c2_in_c1_size,
                    "ch3_not_in_ch2_but_in_ch1_size": c3_not_in_c2_but_in_c1_size,
                }
            )

    return result


# @BatchProcessingRegistry.register(
#     name="ROI Colocalization",
#     suffix="_coloc",
#     description="Analyze colocalization between ROIs in multiple channel label images",
#     parameters={
#         "get_sizes": {
#             "type": bool,
#             "default": False,
#             "description": "Calculate size statistics",
#         },
#         "size_method": {
#             "type": str,
#             "default": "median",
#             "description": "Method for size calculation (median or sum)",
#         },
#     },
# )
def roi_colocalization(image, get_sizes=False, size_method="median"):
    """
    Calculate colocalization between channels for a multi-channel label image.

    This function takes a multi-channel image where each channel contains
    labeled objects (segmentation masks). It analyzes how objects in one channel
    overlap with objects in the other channels, and returns detailed statistics
    about their colocalization relationships.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image array, should have shape corresponding to a multichannel
        label image (e.g., [n_channels, height, width]).
    get_sizes : bool, optional
        Whether to calculate size statistics for overlapping regions.
    size_method : str, optional
        Method for calculating size statistics ('median' or 'sum').

    Returns:
    --------
    numpy.ndarray
        Multi-channel array with colocalization results
    """
    # Ensure image is a stack of label images (assume first dimension is channels)
    if image.ndim < 3:
        # Handle single channel image - not enough for colocalization
        print("Input must have multiple channels for colocalization analysis")
        # Return a copy of the input with markings
        return image.copy()

    # Extract channels
    channels = [image[i] for i in range(min(3, image.shape[0]))]
    n_channels = len(channels)

    if n_channels < 2:
        print("Need at least 2 channels for colocalization analysis")
        return image.copy()

    # Assign channels
    image_c1, image_c2 = channels[:2]
    image_c3 = channels[2] if n_channels > 2 else None

    # Get unique label IDs in image_c1
    label_ids = get_nonzero_labels(image_c1)

    # Process each label
    results = []
    roi_sizes = {}

    # Pre-calculate sizes for image_c1 if needed
    if get_sizes:
        for prop in measure.regionprops(image_c1.astype(np.uint32)):
            label = int(prop.label)
            roi_sizes[label] = int(prop.area)

    for label_id in label_ids:
        result = process_single_roi(
            label_id, image_c1, image_c2, image_c3, get_sizes, roi_sizes
        )
        results.append(result)

    # Create a new multi-channel output image with colocalization results
    # Each channel will highlight different colocalization results
    out_shape = image_c1.shape

    # For 2 channels: [original ch1, ch2 overlap]
    # For 3 channels: [original ch1, ch2 overlap, ch3 overlap]
    output_channels = n_channels

    # Create output array
    output = np.zeros((output_channels,) + out_shape, dtype=np.uint32)

    # Fill first channel with original labels
    output[0] = image_c1

    # Fill second channel with ch1 labels where ch2 overlaps
    for label_id in label_ids:
        mask = (image_c1 == label_id) & (image_c2 != 0)
        if np.any(mask):
            output[1][mask] = label_id

    # Fill third channel with ch1 labels where ch3 overlaps (if applicable)
    if image_c3 is not None and output_channels > 2:
        for label_id in label_ids:
            mask = (image_c1 == label_id) & (image_c3 != 0)
            if np.any(mask):
                output[2][mask] = label_id

    return output
