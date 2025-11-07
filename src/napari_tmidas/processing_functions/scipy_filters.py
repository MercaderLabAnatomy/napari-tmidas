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
        name="Resize Labels (Nearest, SciPy)",
        suffix="_scaled",
        description="Resize a label mask or label image by a scale factor using nearest-neighbor interpolation (scipy.ndimage.zoom, grid_mode=True) to preserve label integrity without shifting position.",
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
    def resize_labels(
        label_image: np.ndarray, scale_factor: float = 1.0
    ) -> np.ndarray:
        """
        Resize labeled objects while maintaining original array dimensions.

        Objects are scaled isotropically and centered within the original
        coordinate system, preserving spatial relationships with other data.

        Parameters
        ----------
        label_image : np.ndarray
            3D label image where each unique value represents a distinct object
        scale_factor : float
            Scaling factor (e.g., 0.8 = 80% size, 1.2 = 120% size)

        Returns
        -------
        np.ndarray
            Resized label image with same dimensions as input
        """
        import numpy as np
        from scipy.ndimage import zoom

        scale_factor = float(scale_factor)
        if scale_factor == 1.0:
            return label_image.copy()

        original_shape = np.array(label_image.shape)

        # Resize the labeled objects
        scaled = zoom(
            label_image,
            zoom=scale_factor,
            order=0,  # Preserve label values
            grid_mode=True,  # Consistent coordinate system
            mode="grid-constant",
            cval=0,
        ).astype(label_image.dtype)

        new_shape = np.array(scaled.shape)
        result = np.zeros(original_shape, dtype=label_image.dtype)

        # Center the resized objects in the original array
        offset = ((original_shape - new_shape) / 2).astype(int)

        if scale_factor < 1.0:
            # Place smaller objects in center
            slices = tuple(slice(o, o + s) for o, s in zip(offset, new_shape))
            result[slices] = scaled
        else:
            # Extract center region from larger objects
            slices = tuple(
                slice(-o if o < 0 else 0, s - o if o < 0 else s)
                for o, s in zip(offset, original_shape)
            )
            result = scaled[slices]

        return result

    @BatchProcessingRegistry.register(
        name="Subdivide Labels into 3 Layers",
        suffix="_layers",
        description="Subdivide each labeled object into 3 concentric layers and return a single label image where each layer receives a unique ID offset.",
        parameters={
            "is_half_body": {
                "type": bool,
                "default": False,
                "description": "Enable this if the object is cut in half (e.g., half-spheroid). This will create layers as if it were a full body, so the cut surface shows inner/middle/outer layers.",
            },
            "cut_axis": {
                "type": int,
                "default": 0,
                "min": 0,
                "max": 2,
                "description": "For half-bodies: which axis the object is cut along (0=Z, 1=Y, 2=X). Only the cut axis will be scaled 2x, not all dimensions.",
            },
        },
    )
    def subdivide_labels_3layers(
        label_image: np.ndarray, is_half_body: bool = False, cut_axis: int = 0
    ) -> np.ndarray:
        """Subdivide labeled objects into three concentric layers.

        Each object is partitioned into inner, middle, and outer shells of approximately
        equal thickness. The layers are combined into a single label image using
        non-overlapping ID ranges so they remain distinguishable.

        Parameters
        ----------
        label_image : np.ndarray
            Label image where each unique value represents a distinct object.
        is_half_body : bool, optional
            If True, treats the object as a half-body (e.g., half-spheroid cut at a plane).
            Only the specified cut_axis will be scaled by 2x (not all dimensions) to avoid
            excessive memory usage. The cut surface will then show all three layers
            (inner, middle, outer) as if it were the interior of a complete object.
            Default is False.
        cut_axis : int, optional
            For half-bodies: specifies which axis the object is cut along (0=Z, 1=Y, 2=X).
            Only this axis will be scaled 2x to virtually complete the object. For a
            hemisphere cut horizontally, use axis 0 (Z). Default is 0.

        Returns
        -------
        numpy.ndarray
            Single label image containing all three layers with unique label IDs.
        """
        # Define scale factors for the three boundaries
        # To get equal thickness, we need to think about the "radius" reduction
        # For 3 equal layers, we want boundaries at 1.0, ~0.67, ~0.33
        scale_middle = 0.67  # ~67% size
        scale_inner = 0.33  # ~33% size

        original_shape = np.array(label_image.shape)

        # Store information for mapping back to original coordinates
        half_body_offset = None  # If it's a half-body, we need to virtually "complete" the object first
        # by mirroring it along the cut axis to create a full object
        if is_half_body:
            # Validate cut_axis
            if cut_axis < 0 or cut_axis >= label_image.ndim:
                raise ValueError(
                    f"cut_axis must be between 0 and {label_image.ndim - 1}, got {cut_axis}"
                )

            # Find bounding box of the object along cut_axis
            axes_to_sum = tuple(
                i for i in range(label_image.ndim) if i != cut_axis
            )
            projection = np.sum(label_image > 0, axis=axes_to_sum)
            nonzero_indices = np.where(projection > 0)[0]

            if len(nonzero_indices) == 0:
                # Empty image
                return np.zeros_like(label_image)

            # Get bounding box along cut axis
            bbox_start = nonzero_indices[0]
            bbox_end = nonzero_indices[-1] + 1

            # Extract just the object portion along cut_axis
            extract_slices = [slice(None)] * label_image.ndim
            extract_slices[cut_axis] = slice(bbox_start, bbox_end)
            object_portion = label_image[tuple(extract_slices)]

            # Mirror this portion to create a complete object
            flipped = np.flip(object_portion, axis=cut_axis)
            work_image = np.concatenate(
                [flipped, object_portion], axis=cut_axis
            )
            work_shape = np.array(work_image.shape)

            # Remember the offset for mapping back
            half_body_offset = bbox_start
        else:
            work_image = label_image
            work_shape = original_shape

        # Helper function to create a scaled version centered in working space
        def create_scaled_labels(scale_factor):
            if scale_factor == 1.0:
                return work_image.copy()

            scaled = ndimage.zoom(
                work_image,
                zoom=scale_factor,
                order=0,  # Preserve label values
                grid_mode=True,  # Consistent coordinate system
                mode="grid-constant",
                cval=0,
            ).astype(work_image.dtype)

            new_shape = np.array(scaled.shape)
            result = np.zeros(work_shape, dtype=work_image.dtype)

            # Center the resized objects in the working array
            offset = ((work_shape - new_shape) / 2).astype(int)
            slices = tuple(slice(o, o + s) for o, s in zip(offset, new_shape))
            result[slices] = scaled

            return result

        # Create the three scaled versions
        full_labels = work_image.copy()  # Outer boundary (100%)
        middle_labels = create_scaled_labels(
            scale_middle
        )  # Middle boundary (~67%)
        inner_labels = create_scaled_labels(
            scale_inner
        )  # Inner boundary (~33%)

        # Layer 3 (outermost shell): Full - Middle
        layer3 = full_labels.copy()
        layer3[middle_labels > 0] = 0

        # Layer 2 (middle shell): Middle - Inner
        layer2 = middle_labels.copy()
        layer2[inner_labels > 0] = 0

        # Layer 1 (innermost core): Inner
        layer1 = inner_labels.copy()

        max_label = int(label_image.max()) if label_image.size else 0
        if max_label == 0:
            return np.zeros_like(label_image)

        if np.issubdtype(label_image.dtype, np.integer):
            max_needed = max_label * 3
            dtype_choices = [label_image.dtype, np.uint32, np.uint64]
            for dtype in dtype_choices:
                try:
                    info = np.iinfo(dtype)
                except ValueError:
                    continue
                if max_needed <= info.max:
                    result_dtype = dtype
                    break
            else:
                result_dtype = np.uint64
        else:
            result_dtype = np.uint32

        result = np.zeros(work_shape, dtype=result_dtype)

        layer1_mask = layer1 > 0
        if np.any(layer1_mask):
            result[layer1_mask] = layer1[layer1_mask].astype(
                result_dtype, copy=False
            )

        layer2_mask = layer2 > 0
        if np.any(layer2_mask):
            result[layer2_mask] = (
                layer2[layer2_mask].astype(result_dtype, copy=False)
                + max_label
            )

        layer3_mask = layer3 > 0
        if np.any(layer3_mask):
            result[layer3_mask] = layer3[layer3_mask].astype(
                result_dtype, copy=False
            ) + (2 * max_label)

        # If half-body mode, extract back the original half and place in original coordinates
        if is_half_body:
            # Extract the second half (the original object portion with all layers)
            slices = [slice(None)] * result.ndim
            mid_point = work_shape[cut_axis] // 2
            slices[cut_axis] = slice(mid_point, work_shape[cut_axis])
            result_object = result[tuple(slices)]

            # Place back into original volume at original position
            final_result = np.zeros(original_shape, dtype=result.dtype)
            place_slices = [slice(None)] * result.ndim
            place_slices[cut_axis] = slice(
                half_body_offset,
                half_body_offset + result_object.shape[cut_axis],
            )
            final_result[tuple(place_slices)] = result_object
            result = final_result

        return result

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

else:
    # Export stub functions that raise ImportError when called
    def gaussian_blur(*args, **kwargs):
        raise ImportError(
            "SciPy is not available. Please install scipy to use this function."
        )

    def median_filter(*args, **kwargs):
        raise ImportError(
            "SciPy is not available. Please install scipy to use this function."
        )
