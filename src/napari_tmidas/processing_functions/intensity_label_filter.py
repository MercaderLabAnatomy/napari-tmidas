# processing_functions/intensity_label_filter.py
"""
Processing functions for filtering labels based on intensity using k-medoids clustering.
"""
import inspect
from pathlib import Path
from typing import Dict

import numpy as np
from skimage import measure

from napari_tmidas._registry import BatchProcessingRegistry


def _convert_semantic_to_instance(image: np.ndarray) -> np.ndarray:
    """
    Convert semantic labels (where all objects have the same value) to instance labels.

    Parameters
    ----------
    image : np.ndarray
        Label image that may contain semantic labels

    Returns
    -------
    np.ndarray
        Image with instance labels (each connected component gets unique label)
    """
    if image is None or np.all(image == 0):
        return image

    # Get unique non-zero values
    unique_labels = np.unique(image[image != 0])

    # If there's only one unique non-zero value, it's definitely semantic
    if len(unique_labels) == 1:
        # Single semantic label - convert to instance labels
        mask = image > 0
        return measure.label(mask, connectivity=None)
    else:
        # Multiple labels - already instance labels
        return image


# Lazy imports for optional dependencies
try:
    from sklearn_extra.cluster import KMedoids

    _HAS_KMEDOIDS = True
except (ImportError, ValueError) as e:
    # ImportError: package not installed
    # ValueError: binary incompatibility (e.g., numpy version mismatch)
    KMedoids = None
    _HAS_KMEDOIDS = False
    error_msg = str(e)
    if "numpy.dtype size changed" in error_msg:
        print(
            "scikit-learn-extra has a NumPy binary incompatibility. "
            "This is typically resolved by reinstalling scikit-learn-extra. "
            "Run: pip install --force-reinstall --no-cache-dir scikit-learn-extra"
        )
    else:
        print(
            f"scikit-learn-extra not available ({type(e).__name__}). "
            "Install with: pip install scikit-learn-extra"
        )

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


def _calculate_label_mean_intensities(
    label_image: np.ndarray, intensity_image: np.ndarray
) -> Dict[int, float]:
    """
    Calculate mean intensity for each label.

    Parameters
    ----------
    label_image : np.ndarray
        Label image with integer labels
    intensity_image : np.ndarray
        Intensity image corresponding to the label image

    Returns
    -------
    Dict[int, float]
        Dictionary mapping label IDs to mean intensities
    """
    # Use regionprops to calculate mean intensity for each label
    props = measure.regionprops_table(
        label_image, intensity_image, properties=["label", "intensity_mean"]
    )

    return dict(zip(props["label"], props["intensity_mean"]))


def _cluster_intensities(
    intensities: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster intensities using k-medoids and determine threshold.

    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values to cluster
    n_clusters : int
        Number of clusters (2 or 3)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Cluster labels, cluster centers (medoids), and threshold value
    """
    if not _HAS_KMEDOIDS:
        raise ImportError(
            "scikit-learn-extra is required for k-medoids clustering. "
            "Install with: pip install scikit-learn-extra"
        )

    # Reshape for sklearn
    X = intensities.reshape(-1, 1)

    # Perform k-medoids clustering
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method="pam")
    cluster_labels = kmedoids.fit_predict(X)
    medoids = kmedoids.cluster_centers_.flatten()

    # Sort medoids to identify clusters from low to high intensity
    sorted_indices = np.argsort(medoids)
    sorted_medoids = medoids[sorted_indices]

    # Create mapping from old cluster labels to sorted cluster labels
    label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(sorted_indices)
    }
    sorted_labels = np.array(
        [label_mapping[label] for label in cluster_labels]
    )

    # Determine threshold between lowest and second-lowest clusters
    # Use midpoint between the two lowest cluster centers
    threshold = (sorted_medoids[0] + sorted_medoids[1]) / 2.0

    return sorted_labels, sorted_medoids, threshold


def _get_intensity_filename(
    label_filename: str, label_suffix: str = "_convpaint_labels_filtered.tif"
) -> str:
    """
    Convert label filename to intensity filename by removing suffix.

    Parameters
    ----------
    label_filename : str
        Filename of the label image
    label_suffix : str
        Suffix to remove from label filename (default: "_convpaint_labels_filtered.tif")

    Returns
    -------
    str
        Intensity image filename
    """
    if label_filename.endswith(label_suffix):
        # Remove the label suffix and add .tif
        base_name = label_filename[: -len(label_suffix)]
        return base_name + ".tif"
    else:
        # If suffix doesn't match, assume same filename
        return label_filename


def _filter_labels_by_threshold(
    label_image: np.ndarray,
    label_intensities: Dict[int, float],
    threshold: float,
) -> np.ndarray:
    """
    Filter labels based on intensity threshold.

    Parameters
    ----------
    label_image : np.ndarray
        Label image with integer labels
    label_intensities : Dict[int, float]
        Dictionary mapping label IDs to mean intensities
    threshold : float
        Intensity threshold - labels below this are removed

    Returns
    -------
    np.ndarray
        Filtered label image with same dtype as input
    """
    filtered_image = label_image.copy()

    # Remove labels with intensity below threshold
    for label_id, intensity in label_intensities.items():
        if intensity < threshold:
            filtered_image[label_image == label_id] = 0

    return filtered_image


@BatchProcessingRegistry.register(
    name="Filter Labels by Intensity (K-medoids)",
    suffix="_intensity_filtered",
    description="Filter out labels with low intensity using k-medoids clustering. Finds corresponding intensity image in same folder. Choose 2 clusters for simple low/high separation, or 3 clusters when you have distinct noise/signal/strong-signal populations.",
    parameters={
        "n_clusters": {
            "type": int,
            "default": 2,
            "description": "Number of clusters (2 or 3). Use 2 for simple low/high separation, 3 for noise/diffuse/strong separation.",
        },
        "save_stats": {
            "type": bool,
            "default": True,
            "description": "Save clustering statistics to CSV file",
        },
    },
)
def filter_labels_by_intensity(
    image: np.ndarray,
    n_clusters: int = 2,
    save_stats: bool = True,
) -> np.ndarray:
    """
    Filter labels based on intensity using k-medoids clustering.

    This function processes pairs of label and intensity images in the same folder.
    For each label image, it finds the corresponding intensity image (removes
    "_convpaint_labels_filtered.tif" suffix from label filename to find intensity file),
    calculates mean intensity per label, performs k-medoids clustering to identify
    intensity groups, and filters out labels in the low intensity cluster.

    Use n_clusters=2 for simple separation (bad vs. good signal).
    Use n_clusters=3 when you have distinct populations (noise, diffuse signal, strong signal).

    Parameters
    ----------
    image : np.ndarray
        Label image with integer labels
    n_clusters : int
        Number of clusters (2 or 3)
    save_stats : bool
        Whether to save clustering statistics to CSV

    Returns
    -------
    np.ndarray
        Filtered label image with low-intensity labels removed
    """
    # Extract current filepath from call stack
    current_filepath = None
    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals
        if "filepath" in frame_locals:
            current_filepath = frame_locals["filepath"]
            break

    if current_filepath is None:
        raise ValueError(
            "Could not determine current file path from call stack"
        )

    if n_clusters not in [2, 3]:
        raise ValueError(f"n_clusters must be 2 or 3, got {n_clusters}")

    # Convert semantic labels to instance labels if needed
    original_dtype = image.dtype
    image = _convert_semantic_to_instance(image)

    # Check if we actually have any labels after conversion
    unique_labels = np.unique(image[image != 0])
    if len(unique_labels) == 0:
        print("⚠️  No labels found in image, returning empty image")
        return np.zeros_like(image)

    print(f"📋 Found {len(unique_labels)} labels in the image")

    # Find corresponding intensity image in same folder
    label_path = Path(current_filepath)
    label_filename = label_path.name
    intensity_filename = _get_intensity_filename(label_filename)
    intensity_path = label_path.parent / intensity_filename

    if not intensity_path.exists():
        print(
            f"⚠️  No corresponding intensity image found for {label_filename}"
        )
        print(f"   Expected: {intensity_filename}")
        print(f"   Full path: {intensity_path}")
        print("   Skipping this file...")
        return image  # Return original image unchanged

    # Load intensity image directly with tifffile
    try:
        import tifffile

        intensity_image = tifffile.imread(str(intensity_path))
    except (FileNotFoundError, OSError) as e:
        print(f"⚠️  Could not read intensity image: {intensity_path}")
        print(f"   Error: {e}")
        print("   Skipping this file...")
        return image  # Return original if can't read intensity image

    # Validate dimensions match
    if image.shape != intensity_image.shape:
        raise ValueError(
            f"Label and intensity images must have same shape. "
            f"Label: {image.shape}, Intensity: {intensity_image.shape}"
        )

    # Calculate mean intensity for each label
    label_intensities = _calculate_label_mean_intensities(
        image, intensity_image
    )

    if len(label_intensities) == 0:
        print(f"⚠️  No labels found in {label_filename}, returning empty image")
        return np.zeros_like(image)

    # Perform k-medoids clustering
    intensities = np.array(list(label_intensities.values()))
    cluster_labels, medoids, threshold = _cluster_intensities(
        intensities, n_clusters=n_clusters
    )

    # Print results based on number of clusters
    print(f"📊 {label_filename}:")
    print(f"   Total labels: {len(label_intensities)}")

    if n_clusters == 2:
        n_low = np.sum(cluster_labels == 0)
        n_high = np.sum(cluster_labels == 1)
        print(
            f"   Low intensity cluster: {n_low} labels (medoid: {medoids[0]:.2f})"
        )
        print(
            f"   High intensity cluster: {n_high} labels (medoid: {medoids[1]:.2f})"
        )
        print(f"   Threshold: {threshold:.2f}")
        print(f"   Keeping {n_high} labels, removing {n_low} labels")

        # Save statistics if requested
        if save_stats and _HAS_PANDAS:
            stats = {
                "filename": label_filename,
                "n_clusters": n_clusters,
                "total_labels": len(label_intensities),
                "low_cluster_count": n_low,
                "high_cluster_count": n_high,
                "low_cluster_medoid": medoids[0],
                "high_cluster_medoid": medoids[1],
                "threshold": threshold,
            }

            stats_dir = (
                Path(current_filepath).parent / "intensity_filter_stats"
            )
            stats_dir.mkdir(exist_ok=True)
            stats_file = stats_dir / "clustering_stats.csv"

            df = pd.DataFrame([stats])
            if stats_file.exists():
                df.to_csv(stats_file, mode="a", header=False, index=False)
            else:
                df.to_csv(stats_file, index=False)

    else:  # n_clusters == 3
        n_low = np.sum(cluster_labels == 0)
        n_medium = np.sum(cluster_labels == 1)
        n_high = np.sum(cluster_labels == 2)
        print(
            f"   Low intensity cluster: {n_low} labels (medoid: {medoids[0]:.2f})"
        )
        print(
            f"   Medium intensity cluster: {n_medium} labels (medoid: {medoids[1]:.2f})"
        )
        print(
            f"   High intensity cluster: {n_high} labels (medoid: {medoids[2]:.2f})"
        )
        print(f"   Threshold: {threshold:.2f}")
        print(
            f"   Keeping {n_medium + n_high} labels, removing {n_low} labels"
        )

        # Save statistics if requested
        if save_stats and _HAS_PANDAS:
            stats = {
                "filename": label_filename,
                "n_clusters": n_clusters,
                "total_labels": len(label_intensities),
                "low_cluster_count": n_low,
                "medium_cluster_count": n_medium,
                "high_cluster_count": n_high,
                "low_cluster_medoid": medoids[0],
                "medium_cluster_medoid": medoids[1],
                "high_cluster_medoid": medoids[2],
                "threshold": threshold,
            }

            stats_dir = (
                Path(current_filepath).parent / "intensity_filter_stats"
            )
            stats_dir.mkdir(exist_ok=True)
            stats_file = stats_dir / "clustering_stats.csv"

            df = pd.DataFrame([stats])
            if stats_file.exists():
                df.to_csv(stats_file, mode="a", header=False, index=False)
            else:
                df.to_csv(stats_file, index=False)

    # Filter labels
    filtered_image = _filter_labels_by_threshold(
        image, label_intensities, threshold
    )

    # Convert back to original dtype
    if filtered_image.dtype != original_dtype:
        filtered_image = filtered_image.astype(original_dtype)

    return filtered_image
