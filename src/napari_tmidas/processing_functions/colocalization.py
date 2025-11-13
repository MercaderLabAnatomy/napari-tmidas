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


def convert_semantic_to_instance_labels(image, connectivity=None):
    """
    Convert semantic labels (where all objects have the same value) to instance labels.

    Args:
        image: Label image that may contain semantic labels
        connectivity: Connectivity for connected component analysis (1, 2, or None for full)

    Returns:
        Image with instance labels (each connected component gets unique label)
    """
    if image is None or np.all(image == 0):
        return image

    # Get unique non-zero values
    unique_labels = np.unique(image[image != 0])

    # If there's only one unique non-zero value, it's likely semantic
    # But even with multiple values, we should check if they're truly instance labels
    # For safety, we'll convert all non-zero regions to instance labels

    output = np.zeros_like(image)
    current_label = 1

    for semantic_label in unique_labels:
        # Get mask for this semantic label
        mask = image == semantic_label

        # Find connected components
        labeled_components = measure.label(mask, connectivity=connectivity)

        # Relabel to avoid conflicts
        unique_components = np.unique(
            labeled_components[labeled_components != 0]
        )
        for component_id in unique_components:
            component_mask = labeled_components == component_id
            output[component_mask] = current_label
            current_label += 1

    return output


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


def calculate_intensity_stats(intensity_image, mask):
    """
    Calculate intensity statistics for a masked region.

    Args:
        intensity_image: Raw intensity image
        mask: Boolean mask defining the region

    Returns:
        dict: Dictionary with mean, median, std, max, min intensity
    """
    # Get intensity values within the mask
    intensity_values = intensity_image[mask]

    if len(intensity_values) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

    stats = {
        "mean": float(np.mean(intensity_values)),
        "median": float(np.median(intensity_values)),
        "std": float(np.std(intensity_values)),
        "max": float(np.max(intensity_values)),
        "min": float(np.min(intensity_values)),
    }

    return stats


def count_positive_objects(
    image_c2,
    intensity_c3,
    mask_roi,
    threshold_method="percentile",
    threshold_value=75.0,
):
    """
    Count Channel 2 objects that are positive for Channel 3 signal.

    Args:
        image_c2: Label image of Channel 2 (e.g., nuclei)
        intensity_c3: Intensity image of Channel 3 (e.g., KI67)
        mask_roi: Boolean mask for the ROI from Channel 1
        threshold_method: 'percentile' or 'absolute'
        threshold_value: Threshold value (0-100 for percentile, or absolute intensity)

    Returns:
        dict: Dictionary with counts and threshold info
    """
    # Get all unique Channel 2 objects in the ROI
    c2_in_roi = image_c2 * mask_roi
    c2_labels = np.unique(c2_in_roi)
    c2_labels = c2_labels[c2_labels != 0]  # Remove background

    if len(c2_labels) == 0:
        return {
            "total_c2_objects": 0,
            "positive_c2_objects": 0,
            "negative_c2_objects": 0,
            "percent_positive": 0.0,
            "threshold_used": 0.0,
        }

    # Calculate threshold
    if threshold_method == "percentile":
        # Calculate threshold from all Channel 3 intensity values within ROI where Channel 2 exists
        mask_c2_in_roi = c2_in_roi > 0
        intensity_in_c2 = intensity_c3[mask_c2_in_roi]
        if len(intensity_in_c2) > 0:
            threshold = float(np.percentile(intensity_in_c2, threshold_value))
        else:
            threshold = 0.0
    else:  # absolute
        threshold = threshold_value

    # Count positive objects
    positive_count = 0
    for label_id in c2_labels:
        # Get mask for this specific Channel 2 object
        mask_c2_obj = (image_c2 == label_id) & mask_roi

        # Get mean intensity of Channel 3 in this Channel 2 object
        intensity_in_obj = intensity_c3[mask_c2_obj]
        if len(intensity_in_obj) > 0:
            mean_intensity = float(np.mean(intensity_in_obj))
            if mean_intensity >= threshold:
                positive_count += 1

    total_count = int(len(c2_labels))
    negative_count = total_count - positive_count
    percent_positive = (
        (positive_count / total_count * 100) if total_count > 0 else 0.0
    )

    return {
        "total_c2_objects": total_count,
        "positive_c2_objects": positive_count,
        "negative_c2_objects": negative_count,
        "percent_positive": percent_positive,
        "threshold_used": threshold,
    }


def process_single_roi(
    label_id,
    image_c1,
    image_c2,
    image_c3=None,
    get_sizes=False,
    roi_sizes=None,
    channel2_is_labels=True,
    channel3_is_labels=True,
    image_c2_intensity=None,
    image_c3_intensity=None,
    count_positive=False,
    threshold_method="percentile",
    threshold_value=75.0,
    convert_to_instances_c2=False,
    convert_to_instances_c3=False,
):
    """
    Process a single ROI for colocalization analysis.

    Args:
        label_id: Label ID to process
        image_c1: First channel image (ROI labels)
        image_c2: Second channel image (object labels or intensity)
        image_c3: Third channel image (labels or intensity, optional)
        get_sizes: Whether to calculate size statistics
        roi_sizes: Pre-calculated ROI sizes dict
        channel2_is_labels: If True, treat channel 2 as labels; if False, as intensity
        channel3_is_labels: If True, treat channel 3 as labels; if False, as intensity
        image_c2_intensity: Separate intensity image for channel 2 (optional)
        image_c3_intensity: Separate intensity image for channel 3 (optional)
        count_positive: Count positive objects (only applicable when one channel is labels and another is intensity)
        threshold_method: 'percentile' or 'absolute' for positive counting
        threshold_value: Threshold value for positive counting
        convert_to_instances_c2: If True, convert semantic labels to instance labels for channel 2
        convert_to_instances_c3: If True, convert semantic labels to instance labels for channel 3
    """
    # Convert semantic labels to instance labels if requested
    if convert_to_instances_c2 and channel2_is_labels and image_c2 is not None:
        image_c2 = convert_semantic_to_instance_labels(image_c2)

    if convert_to_instances_c3 and channel3_is_labels and image_c3 is not None:
        image_c3 = convert_semantic_to_instance_labels(image_c3)

    # Create masks once
    mask_roi = image_c1 == label_id

    # Build the result dictionary
    result = {"label_id": int(label_id)}

    # Handle Channel 2 based on whether it's labels or intensity
    if channel2_is_labels:
        mask_c2 = image_c2 != 0
        # Calculate counts
        c2_in_c1_count = count_unique_nonzero(image_c2, mask_roi & mask_c2)
        result["ch2_in_ch1_count"] = c2_in_c1_count
    else:
        # Channel 2 is intensity - calculate intensity statistics
        intensity_img = (
            image_c2_intensity if image_c2_intensity is not None else image_c2
        )
        stats_c2 = calculate_intensity_stats(intensity_img, mask_roi)
        result.update(
            {
                "ch2_in_ch1_mean": stats_c2["mean"],
                "ch2_in_ch1_median": stats_c2["median"],
                "ch2_in_ch1_std": stats_c2["std"],
                "ch2_in_ch1_max": stats_c2["max"],
            }
        )

    # Add size information if requested (only for label-based channels)
    if get_sizes:
        if roi_sizes is None:
            roi_sizes = {}
            # Calculate sizes for current label only
            area = np.sum(mask_roi)
            roi_sizes[label_id] = area

        size = roi_sizes.get(int(label_id), 0)
        result["ch1_size"] = size

        if channel2_is_labels:
            c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id)
            result["ch2_in_ch1_size"] = c2_in_c1_size

    # Handle third channel if present
    if image_c3 is not None:
        if channel3_is_labels:
            # Original behavior: count objects in channel 3
            mask_c3 = image_c3 != 0

            if channel2_is_labels:
                # Both ch2 and ch3 are labels - original 3-channel label mode
                mask_c2 = image_c2 != 0

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
                        image_c1,
                        image_c2,
                        label_id,
                        mask_c2=True,
                        image_c3=image_c3,
                    )
                    c3_not_in_c2_but_in_c1_size = calculate_coloc_size(
                        image_c1,
                        image_c2,
                        label_id,
                        mask_c2=False,
                        image_c3=image_c3,
                    )

                    result.update(
                        {
                            "ch3_in_ch2_in_ch1_size": c3_in_c2_in_c1_size,
                            "ch3_not_in_ch2_but_in_c1_size": c3_not_in_c2_but_in_c1_size,
                        }
                    )
            else:
                # Ch2 is intensity, Ch3 is labels - count Ch3 objects in Ch1
                c3_in_c1_count = count_unique_nonzero(
                    image_c3, mask_roi & mask_c3
                )
                result["ch3_in_ch1_count"] = c3_in_c1_count

                if get_sizes:
                    c3_in_c1_size = calculate_coloc_size(
                        image_c1, image_c3, label_id
                    )
                    result["ch3_in_ch1_size"] = c3_in_c1_size
        else:
            # Channel 3 is intensity
            intensity_img = (
                image_c3_intensity
                if image_c3_intensity is not None
                else image_c3
            )

            if channel2_is_labels:
                # Ch2 is labels, Ch3 is intensity - original intensity mode
                mask_c2 = image_c2 != 0

                # Calculate intensity where c2 is present in c1
                mask_c2_in_c1 = mask_roi & mask_c2
                stats_c2_in_c1 = calculate_intensity_stats(
                    intensity_img, mask_c2_in_c1
                )

                # Calculate intensity where c2 is NOT present in c1
                mask_not_c2_in_c1 = mask_roi & ~mask_c2
                stats_not_c2_in_c1 = calculate_intensity_stats(
                    intensity_img, mask_not_c2_in_c1
                )

                # Add intensity statistics to result
                result.update(
                    {
                        "ch3_in_ch2_in_ch1_mean": stats_c2_in_c1["mean"],
                        "ch3_in_ch2_in_ch1_median": stats_c2_in_c1["median"],
                        "ch3_in_ch2_in_ch1_std": stats_c2_in_c1["std"],
                        "ch3_in_ch2_in_ch1_max": stats_c2_in_c1["max"],
                        "ch3_not_in_ch2_but_in_ch1_mean": stats_not_c2_in_c1[
                            "mean"
                        ],
                        "ch3_not_in_ch2_but_in_ch1_median": stats_not_c2_in_c1[
                            "median"
                        ],
                        "ch3_not_in_ch2_but_in_ch1_std": stats_not_c2_in_c1[
                            "std"
                        ],
                        "ch3_not_in_ch2_but_in_ch1_max": stats_not_c2_in_c1[
                            "max"
                        ],
                    }
                )

                # Count positive Channel 2 objects if requested
                if count_positive:
                    positive_counts = count_positive_objects(
                        image_c2,
                        intensity_img,
                        mask_roi,
                        threshold_method,
                        threshold_value,
                    )
                    result.update(
                        {
                            "ch2_in_ch1_positive_for_ch3_count": positive_counts[
                                "positive_c2_objects"
                            ],
                            "ch2_in_ch1_negative_for_ch3_count": positive_counts[
                                "negative_c2_objects"
                            ],
                            "ch2_in_ch1_percent_positive_for_ch3": positive_counts[
                                "percent_positive"
                            ],
                            "ch3_threshold_used": positive_counts[
                                "threshold_used"
                            ],
                        }
                    )
            else:
                # Both Ch2 and Ch3 are intensity - just add Ch3 stats to Ch1 ROIs
                stats_c3 = calculate_intensity_stats(intensity_img, mask_roi)
                result.update(
                    {
                        "ch3_in_ch1_mean": stats_c3["mean"],
                        "ch3_in_ch1_median": stats_c3["median"],
                        "ch3_in_ch1_std": stats_c3["std"],
                        "ch3_in_ch1_max": stats_c3["max"],
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
#         "channel2_is_labels": {
#             "type": bool,
#             "default": True,
#             "description": "Treat channel 2 as labels (True) or intensity (False)",
#         },
#         "channel3_is_labels": {
#             "type": bool,
#             "default": True,
#             "description": "Treat channel 3 as labels (True) or intensity (False)",
#         },
#         "count_positive": {
#             "type": bool,
#             "default": False,
#             "description": "Count positive objects (when one channel is labels and another is intensity)",
#         },
#         "threshold_method": {
#             "type": str,
#             "default": "percentile",
#             "description": "Threshold method: 'percentile' or 'absolute'",
#         },
#         "threshold_value": {
#             "type": float,
#             "default": 75.0,
#             "description": "Threshold value for positive counting",
#         },
#     },
# )
def roi_colocalization(
    image,
    get_sizes=False,
    size_method="median",
    channel2_is_labels=True,
    channel3_is_labels=True,
    count_positive=False,
    threshold_method="percentile",
    threshold_value=75.0,
):
    """
    Calculate colocalization between channels for a multi-channel label/intensity image.

    This function takes a multi-channel image where each channel contains
    labeled objects (segmentation masks) or intensity values. It analyzes how
    objects in one channel overlap with objects in the other channels, and
    returns detailed statistics about their colocalization relationships.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image array, should have shape corresponding to a multichannel
        image (e.g., [n_channels, height, width]).
    get_sizes : bool, optional
        Whether to calculate size statistics for overlapping regions (only for label channels).
    size_method : str, optional
        Method for calculating size statistics ('median' or 'sum').
    channel2_is_labels : bool, optional
        If True, treat channel 2 as labeled objects. If False, treat as intensity image.
    channel3_is_labels : bool, optional
        If True, treat channel 3 as labeled objects. If False, treat as intensity image.
    count_positive : bool, optional
        Count Channel 2 objects positive for Channel 3 signal (only when channel3_is_labels=False).
    threshold_method : str, optional
        Method for positive threshold: 'percentile' or 'absolute'.
    threshold_value : float, optional
        Threshold value (0-100 for percentile, or absolute intensity value).

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

    # Handle intensity images for channel 2 and 3
    image_c2_intensity = None
    image_c3_intensity = None

    if not channel2_is_labels:
        image_c2_intensity = image_c2

    if image_c3 is not None and not channel3_is_labels:
        image_c3_intensity = image_c3

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
            label_id,
            image_c1,
            image_c2,
            image_c3,
            get_sizes,
            roi_sizes,
            channel2_is_labels,
            channel3_is_labels,
            image_c2_intensity,
            image_c3_intensity,
            count_positive,
            threshold_method,
            threshold_value,
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

    # Fill second channel based on whether it's labels or intensity
    if channel2_is_labels:
        # Fill with ch1 labels where ch2 overlaps
        for label_id in label_ids:
            mask = (image_c1 == label_id) & (image_c2 != 0)
            if np.any(mask):
                output[1][mask] = label_id
    else:
        # For intensity-based channel 2, show the intensity values within ch1 ROIs
        for label_id in label_ids:
            mask = image_c1 == label_id
            if np.any(mask):
                output[1][mask] = image_c2[mask]

    # Fill third channel with ch1 labels where ch3 overlaps (if applicable)
    if image_c3 is not None and output_channels > 2:
        if channel3_is_labels:
            # For label-based channel 3, show overlap
            if channel2_is_labels:
                # Ch2 is labels - show ch3 overlap with ch2 in ch1
                for label_id in label_ids:
                    mask = (image_c1 == label_id) & (image_c3 != 0)
                    if np.any(mask):
                        output[2][mask] = label_id
            else:
                # Ch2 is intensity - just show ch3 overlap with ch1
                for label_id in label_ids:
                    mask = (image_c1 == label_id) & (image_c3 != 0)
                    if np.any(mask):
                        output[2][mask] = label_id
        else:
            # For intensity-based channel 3, show the intensity values
            for label_id in label_ids:
                mask = image_c1 == label_id
                if np.any(mask):
                    output[2][mask] = image_c3[mask]

    return output
