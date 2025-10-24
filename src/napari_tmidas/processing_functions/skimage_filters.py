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

# Lazy imports for optional heavy dependencies
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

from napari_tmidas._file_selector import ProcessingWorker
from napari_tmidas._registry import BatchProcessingRegistry

if SKIMAGE_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Resize Labels (Nearest)",
        suffix="_scaled",
        description="Resize a label mask or label image by a scale factor using nearest-neighbor interpolation (order=0, anti_aliasing=False) to preserve label integrity.",
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
        Resize a label mask or label image by a scale factor using nearest-neighbor interpolation to preserve label integrity.
        """
        scale_factor = float(scale_factor)
        if scale_factor == 1.0:
            return label_image
        import numpy as np
        from skimage.transform import resize

        new_shape = tuple(
            (np.array(label_image.shape) * scale_factor).astype(int)
        )
        scaled_labels = resize(
            label_image,
            new_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(label_image.dtype)
        return scaled_labels

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
        # Return 255 for values above threshold, 0 for values below
        return np.where(image > thresh, 255, 0).astype(np.uint8)

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
        # Return 255 for values above threshold, 0 for values below
        # This ensures the binary image is visible when viewed as a regular image
        return np.where(image > threshold, 255, 0).astype(np.uint8)

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

else:
    # Export stub functions that raise ImportError when called
    def invert_image(*args, **kwargs):
        raise ImportError(
            "scikit-image is not available. Please install scikit-image to use this function."
        )

    def equalize_histogram(*args, **kwargs):
        raise ImportError(
            "scikit-image is not available. Please install scikit-image to use this function."
        )

    def otsu_thresholding(*args, **kwargs):
        raise ImportError(
            "scikit-image is not available. Please install scikit-image to use this function."
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


# ============================================================================
# Bright Region Extraction Functions
# ============================================================================

if SKIMAGE_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Percentile Threshold (Keep Brightest)",
        suffix="_percentile",
        description="Keep only pixels above a brightness percentile, zero out the rest",
        parameters={
            "percentile": {
                "type": float,
                "default": 90.0,
                "min": 0.0,
                "max": 100.0,
                "description": "Keep pixels brighter than this percentile (0-100)",
            },
            "output_type": {
                "type": str,
                "default": "original",
                "options": ["original", "binary"],
                "description": "Output original values or binary mask",
            },
        },
    )
    def percentile_threshold(
        image: np.ndarray,
        percentile: float = 90.0,
        output_type: str = "original",
    ) -> np.ndarray:
        """
        Keep only pixels above a certain brightness percentile.

        This function calculates the specified percentile of pixel intensities
        and keeps only pixels brighter than that threshold. Darker pixels are
        set to zero.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image array
        percentile : float
            Percentile threshold (0-100). Higher values keep fewer, brighter pixels.
        output_type : str
            'original' returns the original pixel values for pixels above threshold,
            'binary' returns a binary mask (255 for above threshold, 0 otherwise)

        Returns:
        --------
        numpy.ndarray
            Image with only bright regions preserved
        """
        # Calculate the percentile threshold
        threshold = np.percentile(image, percentile)

        if output_type == "binary":
            # Return binary mask
            return np.where(image > threshold, 255, 0).astype(np.uint8)
        else:
            # Return original values above threshold, zero elsewhere
            result = image.copy()
            result[image <= threshold] = 0
            return result

    @BatchProcessingRegistry.register(
        name="Rolling Ball Background Subtraction",
        suffix="_rollingball",
        description="Remove uneven background using rolling ball algorithm (like ImageJ)",
        parameters={
            "radius": {
                "type": int,
                "default": 50,
                "min": 5,
                "max": 200,
                "description": "Radius of rolling ball (larger = remove broader background)",
            }
        },
    )
    def rolling_ball_background(
        image: np.ndarray, radius: int = 50
    ) -> np.ndarray:
        """
        Remove background using rolling ball algorithm.

        This algorithm estimates and removes uneven background by simulating
        a ball rolling under the image surface. It's particularly effective
        for fluorescence microscopy images with uneven illumination.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image array
        radius : int
            Radius of the rolling ball. Should be larger than the largest
            feature you want to keep. Larger values remove broader background
            variations.

        Returns:
        --------
        numpy.ndarray
            Background-subtracted image with bright features preserved
        """
        from skimage.restoration import rolling_ball

        # Estimate background
        background = rolling_ball(image, radius=radius)

        # Subtract background and clip to valid range
        result = image.astype(np.float32) - background
        result = np.clip(result, 0, None)

        # Convert back to original dtype range if needed
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)

        return result

    @BatchProcessingRegistry.register(
        name="Adaptive Threshold (Bright Bias)",
        suffix="_adaptive_bright",
        description="Adaptive thresholding biased to keep bright regions",
        parameters={
            "block_size": {
                "type": int,
                "default": 35,
                "min": 3,
                "max": 201,
                "description": "Size of local neighborhood (must be odd)",
            },
            "offset": {
                "type": float,
                "default": -10.0,
                "min": -128.0,
                "max": 128.0,
                "description": "Constant subtracted from mean (negative = keep more bright pixels)",
            },
        },
    )
    def adaptive_threshold_bright(
        image: np.ndarray, block_size: int = 35, offset: float = -10.0
    ) -> np.ndarray:
        """
        Apply adaptive thresholding with bias toward bright regions.

        Unlike global thresholding, adaptive thresholding calculates a threshold
        for each pixel based on its local neighborhood. The negative offset
        biases the threshold to keep more bright pixels.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image array
        block_size : int
            Size of the local neighborhood for threshold calculation. Must be odd.
            Larger values consider broader neighborhoods.
        offset : float
            Value subtracted from the local mean. Negative values (like -10)
            lower the threshold, keeping more bright pixels.

        Returns:
        --------
        numpy.ndarray
            Binary image (255 for bright regions, 0 elsewhere)
        """
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = skimage.img_as_ubyte(image)

        # Apply adaptive thresholding
        threshold = skimage.filters.threshold_local(
            image, block_size=block_size, offset=offset
        )

        # Create binary mask
        binary = image > threshold

        # Return as uint8 (255/0)
        return (binary * 255).astype(np.uint8)
