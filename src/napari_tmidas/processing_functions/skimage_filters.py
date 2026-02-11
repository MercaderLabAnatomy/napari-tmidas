# processing_functions/skimage_filters.py
"""
Processing functions that depend on scikit-image.
"""
import concurrent.futures

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


# Lazy imports for optional heavy dependencies
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

from napari_tmidas._registry import BatchProcessingRegistry

if SKIMAGE_AVAILABLE:

    def _equalize_histogram_dask(
        image, clip_limit: float, kernel_size: int, max_workers: int
    ):
        """
        Apply CLAHE to a Dask array, processing each T,C combination independently.

        CLAHE operates on histograms, so it MUST be applied to complete ZYX volumes
        independently for each T and each C. Uses map_blocks to ensure chunks don't
        span T or C boundaries, then applies map_overlap for spatial processing.
        """
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Dask is required for processing large Zarr arrays. "
                "Install with: pip install dask[array]"
            ) from e

        original_dtype = image.dtype

        # Auto-calculate kernel size if not specified
        if kernel_size <= 0:
            # Use 1/8 of the smaller spatial dimension
            min_dim = min(image.shape[-2:])
            kernel_size = max(16, min(128, min_dim // 8))

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        print(
            f"Processing Dask array with CLAHE: "
            f"kernel_size={kernel_size}, clip_limit={clip_limit}"
        )
        print(f"Array shape: {image.shape}, chunks: {image.chunks}")

        def apply_clahe_block(block, block_id=None):
            """
            Apply CLAHE to a block that contains complete ZYX volumes for specific T,C.
            The block should have T,C dimensions with size 1 (single timepoint/channel).
            """
            # Remove singleton T,C dimensions for processing
            if block.ndim == 5:  # TCZYX with T=1, C=1
                zyx = block[0, 0]  # Get ZYX volume
                result_zyx = skimage.exposure.equalize_adapthist(
                    zyx, kernel_size=kernel_size, clip_limit=clip_limit
                )
                # Restore T,C dimensions
                result = result_zyx[np.newaxis, np.newaxis, ...]
            elif block.ndim == 4:  # CZYX or TZYX with first dim=1
                zyx = block[0]  # Get ZYX volume
                result_zyx = skimage.exposure.equalize_adapthist(
                    zyx, kernel_size=kernel_size, clip_limit=clip_limit
                )
                result = result_zyx[np.newaxis, ...]
            elif block.ndim == 3:  # ZYX
                result = skimage.exposure.equalize_adapthist(
                    block, kernel_size=kernel_size, clip_limit=clip_limit
                )
            else:  # 2D
                result = skimage.exposure.equalize_adapthist(
                    block, kernel_size=kernel_size, clip_limit=clip_limit
                )

            # Convert back to original dtype
            if np.issubdtype(original_dtype, np.integer):
                iinfo = np.iinfo(original_dtype)
                result = (result * (iinfo.max - iinfo.min) + iinfo.min).astype(
                    original_dtype
                )
            else:
                result = result.astype(original_dtype)

            return result

        # Calculate overlap depth for spatial dimensions
        depth = kernel_size // 2

        # Rechunk to ensure T,C dimensions have chunk size 1
        # and Z dimension is not chunked (complete Z stacks)
        if image.ndim == 5:  # TCZYX
            print(
                f"Processing {image.shape[0]} timepoints × {image.shape[1]} channels"
            )
            # Rechunk: each chunk should be (1,1,Z,Y,X) to keep complete ZYX volumes
            target_chunks = (1, 1, image.shape[2], "auto", "auto")
            image_rechunked = image.rechunk(target_chunks)
            print(f"Rechunked to: {image_rechunked.chunks}")

            # Apply map_overlap with depth only on Y,X
            result = da.map_overlap(
                apply_clahe_block,
                image_rechunked,
                depth={
                    0: 0,
                    1: 0,
                    2: 0,
                    3: depth,
                    4: depth,
                },  # T,C,Z:0, Y,X:depth
                boundary="reflect",
                dtype=original_dtype,
            )

        elif image.ndim == 4:  # CZYX or TZYX
            print(f"Processing {image.shape[0]} volumes")
            # Rechunk: (1,Z,Y,X) to keep complete ZYX volumes
            target_chunks = (1, image.shape[1], "auto", "auto")
            image_rechunked = image.rechunk(target_chunks)
            print(f"Rechunked to: {image_rechunked.chunks}")

            result = da.map_overlap(
                apply_clahe_block,
                image_rechunked,
                depth={0: 0, 1: 0, 2: depth, 3: depth},  # T/C,Z:0, Y,X:depth
                boundary="reflect",
                dtype=original_dtype,
            )

        elif image.ndim == 3:  # ZYX
            print("Processing single ZYX volume")
            # Rechunk: (Z,Y,X) complete Z
            target_chunks = (image.shape[0], "auto", "auto")
            image_rechunked = image.rechunk(target_chunks)

            result = da.map_overlap(
                apply_clahe_block,
                image_rechunked,
                depth={0: 0, 1: depth, 2: depth},  # Z:0, Y,X:depth
                boundary="reflect",
                dtype=original_dtype,
            )

        else:  # 2D (YX)
            print("Processing single 2D image")
            result = da.map_overlap(
                apply_clahe_block,
                image,
                depth={0: depth, 1: depth},  # Y,X:depth
                boundary="reflect",
                dtype=original_dtype,
            )

        print(
            "CLAHE applied to Dask array (will process chunks in parallel on compute)"
        )
        return result

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    @BatchProcessingRegistry.register(
        name="CLAHE (Adaptive Histogram Equalization)",
        suffix="_clahe",
        description="Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast, especially useful for dark images with weak bright features",
        parameters={
            "clip_limit": {
                "type": float,
                "default": 0.01,
                "description": "Clipping limit for contrast (0.01 = 1%). Higher values give more contrast but may amplify noise. Range: 0.001-0.1",
            },
            "kernel_size": {
                "type": int,
                "default": 0,
                "description": "Size of the local region (0 = auto-calculate based on image size). For small features use smaller values (e.g., 32), for large features use larger values (e.g., 128)",
            },
            "max_workers": {
                "type": int,
                "default": 4,
                "description": "Maximum number of parallel workers (default: 4). Lower values use less memory but are slower. Range: 1-16",
            },
        },
    )
    def equalize_histogram(
        image: np.ndarray,
        clip_limit: float = 0.01,
        kernel_size: int = 0,
        max_workers: int = 4,
        _source_filepath: str = None,
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.

        This is much better than standard histogram equalization for dark images with
        weak bright features like membranes, as it works locally and prevents over-brightening
        of background regions.

        Parameters
        ----------
        image : np.ndarray or dask.array
            Input image (supports 2D, 3D, and 4D arrays like YX, ZYX, TYX, TZYX)
            Can be a Dask array for out-of-core processing of large Zarr files
        clip_limit : float
            Clipping limit for contrast limiting (normalized to 0-1 range, e.g., 0.01 = 1%)
            Higher values give more contrast but may amplify noise
        kernel_size : int
            Size of the contextual regions (0 = auto-calculate based on image size)
        max_workers : int
            Maximum number of parallel workers for processing large datasets (default: 4)
            Lower values reduce memory usage, higher values increase speed
        _source_filepath : str, optional
            Internal parameter for Zarr-aware processing (passed automatically)

        Returns
        -------
        np.ndarray or dask.array
            CLAHE-enhanced image with same dtype as input

        Notes
        -----
        For large multi-dimensional datasets (TZYX), processing is parallelized across
        the first dimension to utilize multiple CPU cores effectively. The max_workers
        parameter controls memory usage vs speed tradeoff.

        For Dask arrays from Zarr files, uses map_blocks for efficient chunk-wise processing
        without loading the entire array into memory.
        """
        # Check if input is a Dask array
        is_dask = hasattr(image, "chunks") and hasattr(image, "map_blocks")

        if is_dask:
            print(
                f"Applying CLAHE to Dask array with shape {image.shape}, "
                f"chunks {image.chunks}"
            )
            return _equalize_histogram_dask(
                image, clip_limit, kernel_size, max_workers
            )

        # Store original dtype to convert back later
        original_dtype = image.dtype

        # Print diagnostic info for multi-dimensional data
        if image.ndim > 2:
            print(
                f"Applying CLAHE to {image.ndim}D image with shape {image.shape}"
            )

        # Auto-calculate kernel size if not specified
        if kernel_size <= 0:
            # Use 1/8 of the smaller dimension, but cap between 16 and 128
            min_dim = min(
                image.shape[-2:]
            )  # Last 2 dimensions are spatial (Y, X)
            kernel_size = max(16, min(128, min_dim // 8))

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        if image.ndim > 2:
            print(f"Using kernel_size={kernel_size}, clip_limit={clip_limit}")

        # Clamp max_workers to reasonable range
        max_workers = max(1, min(max_workers, 16))

        # For 4D data (TZYX), parallelize across first dimension for better performance
        if image.ndim == 4 and image.shape[0] > 1:
            print(
                f"Parallelizing CLAHE across {image.shape[0]} timepoints/slices using {max_workers} workers..."
            )

            def process_slice(idx):
                """Process a single slice along first dimension"""
                result_slice = skimage.exposure.equalize_adapthist(
                    image[idx], kernel_size=kernel_size, clip_limit=clip_limit
                )
                if (idx + 1) % max(1, image.shape[0] // 10) == 0:
                    print(f"  Processed {idx + 1}/{image.shape[0]} slices")
                return result_slice

            # Process in parallel with limited workers to control memory usage
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_slice, i)
                    for i in range(image.shape[0])
                ]
                results = [future.result() for future in futures]

            result = np.stack(results, axis=0)
            print("CLAHE processing complete!")

        elif image.ndim == 3 and image.shape[0] > 5:
            # For 3D data with many slices, also parallelize
            print(
                f"Parallelizing CLAHE across {image.shape[0]} slices using {max_workers} workers..."
            )

            def process_slice(idx):
                """Process a single 2D slice"""
                result_slice = skimage.exposure.equalize_adapthist(
                    image[idx], kernel_size=kernel_size, clip_limit=clip_limit
                )
                if (idx + 1) % max(1, image.shape[0] // 10) == 0:
                    print(f"  Processed {idx + 1}/{image.shape[0]} slices")
                return result_slice

            # Process in parallel with limited workers to control memory usage
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_slice, i)
                    for i in range(image.shape[0])
                ]
                results = [future.result() for future in futures]

            result = np.stack(results, axis=0)
            print("CLAHE processing complete!")
        else:
            # For 2D or small 3D data, use native implementation
            if image.ndim > 2:
                print("Processing...")
            result = skimage.exposure.equalize_adapthist(
                image, kernel_size=kernel_size, clip_limit=clip_limit
            )
            if image.ndim > 2:
                print("CLAHE processing complete!")

        # Convert back to original dtype to preserve compatibility
        if np.issubdtype(original_dtype, np.integer):
            # For integer types, scale back to original range
            iinfo = np.iinfo(original_dtype)
            result = (result * (iinfo.max - iinfo.min) + iinfo.min).astype(
                original_dtype
            )
        else:
            # For float types, keep as is but match dtype
            result = result.astype(original_dtype)

        return result

    # simple otsu thresholding
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (semantic)",
        suffix="_otsu_semantic",
        description="Threshold image using Otsu's method to obtain a binary image. Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice.",
    )
    def otsu_thresholding(
        image: np.ndarray, dimension_order: str = "Auto"
    ) -> np.ndarray:
        """
        Threshold image using Otsu's method.

        Args:
            image: Input image (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.)
            dimension_order: Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.)
                            If TYX/ZYX/TCYX/TZYX: processes each frame/slice independently
                            If CYX: processes each channel independently
                            If YX or Auto: processes as single 2D image

        Returns:
            Binary image with same shape as input (255=foreground, 0=background)
        """
        image = skimage.img_as_ubyte(image)  # convert to 8-bit

        # Handle different dimension orders
        if dimension_order in ["TYX", "ZYX", "TCYX", "TZYX", "ZCYX", "TZCYX"]:
            # Process frame-by-frame or slice-by-slice
            result = np.zeros_like(image, dtype=np.uint8)

            # Determine which axes to iterate over
            if len(image.shape) == 3:  # TYX or ZYX
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = np.where(image[i] > thresh, 255, 0).astype(
                        np.uint8
                    )
            elif len(image.shape) == 4:  # TCYX, TZYX, ZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        thresh = skimage.filters.threshold_otsu(image[i, j])
                        result[i, j] = np.where(
                            image[i, j] > thresh, 255, 0
                        ).astype(np.uint8)
            elif len(image.shape) == 5:  # TZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        for k in range(image.shape[2]):
                            thresh = skimage.filters.threshold_otsu(
                                image[i, j, k]
                            )
                            result[i, j, k] = np.where(
                                image[i, j, k] > thresh, 255, 0
                            ).astype(np.uint8)
            else:
                # Fallback for unexpected shapes
                thresh = skimage.filters.threshold_otsu(image)
                result = np.where(image > thresh, 255, 0).astype(np.uint8)

            return result
        elif dimension_order == "CYX":
            # Process each channel independently
            if len(image.shape) >= 3:
                result = np.zeros_like(image, dtype=np.uint8)
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = np.where(image[i] > thresh, 255, 0).astype(
                        np.uint8
                    )
                return result
            else:
                # Fallback if not actually multi-channel
                thresh = skimage.filters.threshold_otsu(image)
                return np.where(image > thresh, 255, 0).astype(np.uint8)
        else:
            # YX or Auto: process as single image
            thresh = skimage.filters.threshold_otsu(image)
            return np.where(image > thresh, 255, 0).astype(np.uint8)

    # instance segmentation
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (instance)",
        suffix="_otsu_labels",
        description="Threshold image using Otsu's method to obtain a multi-label image. Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice.",
    )
    def otsu_thresholding_instance(
        image: np.ndarray, dimension_order: str = "Auto"
    ) -> np.ndarray:
        """
        Threshold image using Otsu's method to create instance labels.

        Args:
            image: Input image (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.)
            dimension_order: Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.)
                            If TYX/ZYX/TCYX/TZYX: processes each frame/slice independently
                            If CYX: processes each channel independently
                            If YX or Auto: processes as single 2D image

        Returns:
            Label image with same shape as input (0=background, 1,2,3...=objects)
        """
        image = skimage.img_as_ubyte(image)  # convert to 8-bit

        # Handle different dimension orders
        if dimension_order in ["TYX", "ZYX", "TCYX", "TZYX", "ZCYX", "TZCYX"]:
            # Process frame-by-frame or slice-by-slice
            result = np.zeros_like(image, dtype=np.uint32)

            # Determine which axes to iterate over
            if len(image.shape) == 3:  # TYX or ZYX
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = skimage.measure.label(
                        image[i] > thresh
                    ).astype(np.uint32)
            elif len(image.shape) == 4:  # TCYX, TZYX, ZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        thresh = skimage.filters.threshold_otsu(image[i, j])
                        result[i, j] = skimage.measure.label(
                            image[i, j] > thresh
                        ).astype(np.uint32)
            elif len(image.shape) == 5:  # TZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        for k in range(image.shape[2]):
                            thresh = skimage.filters.threshold_otsu(
                                image[i, j, k]
                            )
                            result[i, j, k] = skimage.measure.label(
                                image[i, j, k] > thresh
                            ).astype(np.uint32)
            else:
                # Fallback for unexpected shapes
                thresh = skimage.filters.threshold_otsu(image)
                result = skimage.measure.label(image > thresh).astype(
                    np.uint32
                )

            return result
        elif dimension_order == "CYX":
            # Process each channel independently
            if len(image.shape) >= 3:
                result = np.zeros_like(image, dtype=np.uint32)
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = skimage.measure.label(
                        image[i] > thresh
                    ).astype(np.uint32)
                return result
            else:
                # Fallback if not actually multi-channel
                thresh = skimage.filters.threshold_otsu(image)
                return skimage.measure.label(image > thresh).astype(np.uint32)
        else:
            # YX or Auto: process as single image
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
        Remove small labels from label images.

        Works for 2D, 3D, and higher dimensional label images (4D = TZYX, CZYX, etc.).
        For 4D+ data, processes each 3D volume independently.
        Removes connected components (objects) whose area (2D) or volume (3D)
        is smaller than or equal to min_size.

        Parameters
        ----------
        image : np.ndarray
            Label image (2D, 3D, 4D, or higher dimensional)
        min_size : int
            Minimum size threshold in pixels/voxels. Objects with size <= min_size are removed.

        Returns
        -------
        np.ndarray
            Label image with small objects removed
        """
        # For 4D+ data, process each 3D volume separately
        if image.ndim > 3:
            print(
                f"Processing {image.ndim}D label image with shape {image.shape}"
            )
            result = np.zeros_like(image)
            # Process each 3D volume in the first dimension
            for i in range(image.shape[0]):
                result[i] = remove_small_objects(image[i], min_size=min_size)
            return result

        # Use max_size parameter for scikit-image >= 0.26.0
        # which removes objects with size <= max_size
        # This matches the behavior we want (remove objects <= min_size)
        try:
            # Try new API (scikit-image >= 0.26)
            return skimage.morphology.remove_small_objects(
                image, max_size=min_size
            )
        except TypeError:
            # Fall back to old API (scikit-image < 0.26)
            # Note: old min_size removes objects < min_size (strictly less than)
            # To match new behavior, we add 1
            return skimage.morphology.remove_small_objects(
                image, min_size=min_size + 1
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

    # Note: Old "Extract Region Properties" function removed
    # Use "Extract Regionprops to CSV" from regionprops_analysis.py instead
    # which properly handles multi-dimensional data (T, C, Z dimensions)
    # and creates a single CSV for all images in a folder

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
