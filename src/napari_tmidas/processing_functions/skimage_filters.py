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

import contextlib
import os

import pandas as pd

from napari_tmidas._file_selector import ProcessingWorker
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
        description="Threshold image using Otsu's method to obtain a binary image",
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
        description="Threshold image using Otsu's method to obtain a multi-label image",
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
        description="Threshold image using a fixed threshold to obtain a binary image",
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
        name="Remove Small Labels",
        suffix="_rm_small",
        description="Remove small labels from label images",
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
        Remove small labels from label images
        """
        return skimage.morphology.remove_small_objects(
            image, min_size=min_size
        )

    @BatchProcessingRegistry.register(
        name="Invert Image",
        suffix="_inverted",
        description="Invert pixel values in the image using scikit-image's invert function",
    )
    def invert_image(image: np.ndarray) -> np.ndarray:
        """
        Invert the image pixel values.

        This function inverts the values in an image using scikit-image's invert function,
        which handles different data types appropriately.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image array

        Returns:
        --------
        numpy.ndarray
            Inverted image with the same data type as the input
        """
        # Make a copy to avoid modifying the original
        image_copy = image.copy()

        # Use skimage's invert function which handles all data types properly
        return skimage.util.invert(image_copy)

    @BatchProcessingRegistry.register(
        name="Semantic to Instance Segmentation",
        suffix="_instance",
        description="Convert semantic segmentation masks to instance segmentation labels using connected components",
    )
    def semantic_to_instance(image: np.ndarray) -> np.ndarray:
        """
        Convert semantic segmentation masks to instance segmentation labels.

        This function takes a binary or multi-class semantic segmentation mask and
        converts it to an instance segmentation by finding connected components.
        Each connected region receives a unique label.

        Parameters:
        -----------
        image : numpy.ndarray
            Input semantic segmentation mask

        Returns:
        --------
        numpy.ndarray
            Instance segmentation with unique labels for each connected component
        """
        # Create a copy to avoid modifying the original
        instance_mask = image.copy()

        # If the input is multi-class, process each class separately
        if np.max(instance_mask) > 1:
            # Get unique non-zero class values
            class_values = np.unique(instance_mask)
            class_values = class_values[
                class_values > 0
            ]  # Remove background (0)

            # Create an empty output mask
            result = np.zeros_like(instance_mask, dtype=np.uint32)

            # Process each class
            label_offset = 0
            for class_val in class_values:
                # Create binary mask for this class
                binary_mask = (instance_mask == class_val).astype(np.uint8)

                # Find connected components
                labeled = skimage.measure.label(binary_mask, connectivity=2)

                # Skip if no components found
                if np.max(labeled) == 0:
                    continue

                # Add offset to avoid label overlap between classes
                labeled[labeled > 0] += label_offset

                # Add to result
                result = np.maximum(result, labeled)

                # Update offset for next class
                label_offset = np.max(result)
        else:
            # For binary masks, just find connected components
            result = skimage.measure.label(instance_mask > 0, connectivity=2)

        return result.astype(np.uint32)

    @BatchProcessingRegistry.register(
        name="Extract Region Properties",
        suffix="_props",  # Changed to indicate this is for CSV output only
        description="Extract properties of labeled regions and save as CSV (no image output)",
        parameters={
            "properties": {
                "type": str,
                "default": "area,bbox,centroid,eccentricity,euler_number,perimeter",
                "description": "Comma-separated list of properties to extract (e.g., area,perimeter,centroid)",
            },
            "intensity_image": {
                "type": bool,
                "default": False,
                "description": "Use input as intensity image for intensity-based measurements",
            },
            "min_area": {
                "type": int,
                "default": 0,
                "min": 0,
                "max": 100000,
                "description": "Minimum area to include in results (pixels)",
            },
        },
    )
    def extract_region_properties(
        image: np.ndarray,
        properties: str = "area,bbox,centroid,eccentricity,euler_number,perimeter",
        intensity_image: bool = False,
        min_area: int = 0,
    ) -> np.ndarray:
        """
        Extract properties of labeled regions in an image and save results as CSV.

        This function analyzes all labeled regions in a label image and computes
        various region properties like area, perimeter, centroid, etc. The results
        are saved as a CSV file. The input image is returned unchanged.

        Parameters:
        -----------
        image : numpy.ndarray
            Input label image (instance segmentation)
        properties : str
            Comma-separated list of properties to extract
            See scikit-image documentation for all available properties:
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        intensity_image : bool
            Whether to use the input image as intensity image for intensity-based measurements
        min_area : int
            Minimum area (in pixels) for regions to include in results

        Returns:
        --------
        numpy.ndarray
            The original image (unchanged)
        """
        # Check if we have a proper label image
        if image.ndim < 2 or np.max(image) == 0:
            print(
                "Input must be a valid label image with at least one labeled region"
            )
            return image

        # Convert image to proper format for regionprops
        label_image = image.astype(np.int32)

        # Parse the properties list
        prop_list = [prop.strip() for prop in properties.split(",")]

        # Get region properties
        if intensity_image:
            # Use the same image as both label and intensity image # this is wrong
            regions = skimage.measure.regionprops(
                label_image, intensity_image=image
            )
        else:
            regions = skimage.measure.regionprops(label_image)

        # Collect property data
        data = []
        for region in regions:
            # Skip regions that are too small
            if region.area < min_area:
                continue

            # Get all requested properties
            region_data = {"label": region.label}
            for prop in prop_list:
                try:
                    value = getattr(region, prop)

                    # Handle different types of properties
                    if isinstance(value, tuple) or (
                        isinstance(value, np.ndarray) and value.ndim > 0
                    ):
                        # For tuple/array properties like centroid, bbox, etc.
                        if isinstance(value, tuple):
                            value = np.array(value)

                        # For each element in the tuple/array
                        for i, val in enumerate(value):
                            region_data[f"{prop}_{i}"] = val
                    else:
                        # For scalar properties like area, perimeter, etc.
                        region_data[prop] = value
                except AttributeError:
                    print(f"Property '{prop}' not found, skipping")
                    continue

            data.append(region_data)

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Store the DataFrame as an attribute of the function
        extract_region_properties.csv_data = df
        extract_region_properties.save_csv = True
        extract_region_properties.no_image_output = (
            True  # Indicate no image output needed
        )

        print(f"Extracted properties for {len(data)} regions")
        return image

    # Monkey patch to handle saving CSV files without creating a new image file
    try:
        # Check if ProcessingWorker is imported and available
        original_process_file = ProcessingWorker.process_file

        # Create a new version that handles saving CSV
        def process_file_with_csv_export(self, filepath):
            """Modified process_file function that saves CSV after processing."""
            result = original_process_file(self, filepath)

            # Check if there's a result and if we should save CSV
            if isinstance(result, dict) and "processed_file" in result:
                output_path = result["processed_file"]

                # Check if the processing function had CSV data
                if (
                    hasattr(self.processing_func, "save_csv")
                    and self.processing_func.save_csv
                    and hasattr(self.processing_func, "csv_data")
                ):

                    # Get the CSV data
                    df = self.processing_func.csv_data

                    # For functions that don't need an image output, use the original filepath
                    # as the base for the CSV filename
                    if (
                        hasattr(self.processing_func, "no_image_output")
                        and self.processing_func.no_image_output
                    ):
                        # Use the original filepath without creating a new image file
                        base_path = os.path.splitext(filepath)[0]
                        csv_path = f"{base_path}_regionprops.csv"

                        # Don't save a duplicate image file
                        if (
                            os.path.exists(output_path)
                            and output_path != filepath
                        ):
                            contextlib.suppress(OSError)
                    else:
                        # Create CSV filename from the output image path
                        csv_path = (
                            os.path.splitext(output_path)[0]
                            + "_regionprops.csv"
                        )

                    # Save the CSV file
                    df.to_csv(csv_path, index=False)
                    print(f"Saved region properties to {csv_path}")

                    # Add the CSV file to the result
                    result["secondary_files"] = [csv_path]

                    # If we don't need an image output, update the result to just point to the CSV
                    if (
                        hasattr(self.processing_func, "no_image_output")
                        and self.processing_func.no_image_output
                    ):
                        result["processed_file"] = csv_path

            return result

        # Apply the monkey patch
        ProcessingWorker.process_file = process_file_with_csv_export

    except (NameError, AttributeError) as e:
        print(f"Warning: Could not apply CSV export patch: {e}")
        print(
            "Region properties will be extracted but CSV files may not be saved"
        )


# binary to labels
@BatchProcessingRegistry.register(
    name="Binary to Labels",
    suffix="_labels",
    description="Convert binary images to label images (connected components)",
)
def binary_to_labels(image: np.ndarray) -> np.ndarray:
    """
    Convert binary images to label images (connected components)
    """
    # Make a copy of the input image to avoid modifying the original
    label_image = image.copy()

    # Convert binary image to label image using connected components
    label_image = skimage.measure.label(label_image, connectivity=2)

    return label_image


@BatchProcessingRegistry.register(
    name="Convert to 8-bit (uint8)",
    suffix="_uint8",
    description="Convert image data to 8-bit (uint8) format with proper scaling",
)
def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image data to 8-bit (uint8) format with proper scaling.

    This function handles any input image dimensions (including TZYX) and properly
    rescales data to the 0-1 range before conversion to uint8. Ideal for scientific
    imaging data with arbitrary value ranges.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image array of any numerical dtype

    Returns:
    --------
    numpy.ndarray
        8-bit image with shape preserved and values properly scaled
    """
    # Rescale to 0-1 range (works for any input range, negative or positive)
    img_rescaled = skimage.exposure.rescale_intensity(image, out_range=(0, 1))

    # Convert the rescaled image to uint8
    return skimage.img_as_ubyte(img_rescaled)
