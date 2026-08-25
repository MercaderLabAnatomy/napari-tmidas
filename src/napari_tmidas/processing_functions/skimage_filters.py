# processing_functions/skimage_filters.py
"""
Processing functions that depend on scikit-image.
"""
import concurrent.futures
import os

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

# Dimension-order strings offered by the "dimension_order" dropdown widget.
_DIMENSION_ORDER_AXES = frozenset(
    {"YX", "CYX", "TYX", "ZYX", "TCYX", "TZYX", "ZCYX", "TZCYX", "TCZYX"}
)


def _iter_dimension_blocks(image: np.ndarray, dimension_order: str):
    """
    Yield (index, block) pairs for per-slice connected-component labeling.

    T and C are independent axes (each timepoint/channel is looped over
    separately), while Z is kept together with Y,X in a single block so a
    3D object spanning multiple Z slices gets one label instead of a
    different label per slice.

    Raises ValueError if dimension_order can't be resolved for a >2D image:
    guessing wrong here doesn't crash, it silently mislabels data (merging
    labels across time/channel, or splitting one 3D object into many), so
    the caller needs to pick the correct dimension order explicitly.
    """
    if image.ndim <= 2:
        yield (), image
        return

    axes = str(dimension_order or "").upper()
    if axes not in _DIMENSION_ORDER_AXES or len(axes) != image.ndim:
        # A hint like "TCZYX" stays stale after a single channel is
        # extracted upstream, leaving the image channel-free while the hint
        # still lists "C" -- strip it and re-check before giving up.
        stripped = axes.replace("C", "")
        if stripped in _DIMENSION_ORDER_AXES and len(stripped) == image.ndim:
            axes = stripped
        else:
            raise ValueError(
                f"Cannot determine how to label a {image.ndim}D image from "
                f"dimension_order={dimension_order!r}. Please select the "
                "correct dimension order (e.g. TYX, ZYX, TZYX, TCZYX, ...) "
                "from the dimension order dropdown, so 3D objects spanning "
                "Z slices are labeled correctly instead of being split into "
                "one label per slice."
            )

    independent_positions = [i for i, a in enumerate(axes) if a in ("T", "C")]
    if not independent_positions:
        yield (), image
        return

    independent_shape = [image.shape[i] for i in independent_positions]
    for combo in np.ndindex(*independent_shape):
        index = [slice(None)] * image.ndim
        for pos, val in zip(independent_positions, combo):
            index[pos] = val
        index = tuple(index)
        yield index, image[index]


def _stream_remove_small_labels(
    source_path, output_path, min_size: int
) -> str:
    """
    Remove small labels from a stack on disk without ever holding it in RAM.

    A tracked stack is mostly background, so a 90 MB compressed TIFF can be
    70 GB dense — loading it (let alone allocating a same-sized output) is what
    gets the process OOM-killed.  Both passes here read one YX plane at a time
    via ``_PlaneReader`` and the result is streamed straight into the writer,
    so peak memory is a couple of planes regardless of stack size.

    Semantics match the in-memory path exactly: each trailing 3D volume
    (ZYX) is treated independently, label IDs are the objects (no
    re-labelling), and every label whose voxel count in that volume is
    ``<= min_size`` is set to 0.

    Returns the path written.
    """
    import tifffile

    from napari_tmidas.processing_functions.intensity_label_filter import (
        _PlaneReader,
    )

    with _PlaneReader(source_path) as labels:
        shape = labels.shape
        ndim = len(shape)
        if ndim < 2:
            raise ValueError(
                f"{source_path}: expected a 2D+ label image, got shape {shape}"
            )

        # Grouping mirrors the recursive in-memory version: everything before
        # the trailing 3 axes indexes an independent volume.
        group_shape = shape[:-3] if ndim > 3 else ()
        inner_shape = shape[len(group_shape) : -2]
        planes_per_group = int(np.prod(inner_shape)) if inner_shape else 1
        n_groups = int(np.prod(group_shape)) if group_shape else 1
        n_planes = n_groups * planes_per_group

        # Keep the input dtype: narrowing would need a global max-label pass
        # before the writer can be opened, and the in-memory path preserves it
        # too.  (uint32 is the floor for anything that must stay label-like.)
        out_dtype = labels.dtype
        print(
            f"🔎 Streaming {os.path.basename(str(source_path))} "
            f"{shape} {labels.dtype}: {n_groups} volume(s) x "
            f"{planes_per_group} plane(s), removing labels <= {min_size} voxels"
        )

        def group_indices(group):
            for inner in np.ndindex(*inner_shape) if inner_shape else [()]:
                yield tuple(group) + inner

        removed_total = 0
        kept_total = 0
        done = 0

        def plane_iterator():
            nonlocal removed_total, kept_total, done
            for group in np.ndindex(*group_shape) if group_shape else [()]:
                # Pass 1: voxel count per label ID within this volume.
                counts = np.zeros(1, dtype=np.int64)
                for index in group_indices(group):
                    flat = labels.plane(index).ravel()
                    if flat.min() < 0:
                        raise ValueError(
                            f"Label image contains negative values at {index}"
                        )
                    plane_counts = np.bincount(flat)
                    if plane_counts.size > counts.size:
                        grown = np.zeros(plane_counts.size, dtype=np.int64)
                        grown[: counts.size] = counts
                        counts = grown
                    counts[: plane_counts.size] += plane_counts

                # LUT: identity, except small labels (and background) -> 0.
                lut = np.arange(counts.size, dtype=np.int64)
                small = counts <= min_size
                lut[small] = 0
                lut[0] = 0
                lut = lut.astype(out_dtype)

                present = np.nonzero(counts)[0]
                present = present[present > 0]
                n_removed = int(np.count_nonzero(small[present]))
                removed_total += n_removed
                kept_total += present.size - n_removed

                # Pass 2: apply the LUT plane by plane, straight to the writer.
                for index in group_indices(group):
                    yield np.take(lut, labels.plane(index))
                    done += 1
                    if done % 200 == 0 or done == n_planes:
                        print(
                            f"   {done}/{n_planes} planes written", flush=True
                        )

        axes = {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}.get(ndim)
        print(f"💾 Writing {os.path.basename(str(output_path))} ({out_dtype})")
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as writer:
            writer.write(
                plane_iterator(),
                shape=shape,
                dtype=out_dtype,
                compression="zlib",
                # Without this, tifffile reads a leading axis of length 3 or 4
                # (e.g. 4 z-slices) as RGB samples and stores the stack as
                # separate component planes, which breaks the plane iterator.
                photometric="minisblack",
                # Threaded compression queues encoded segments with no
                # backpressure, so peak scales with the whole output instead
                # of one plane.  See merge_small_labels for the measurements.
                maxworkers=1,
                metadata={"axes": axes} if axes else None,
            )

        print(
            f"✅ {os.path.basename(str(output_path))}: "
            f"removed {removed_total} label(s), kept {kept_total}"
        )
    return str(output_path)


if SKIMAGE_AVAILABLE:

    def _resolve_resize_target(shape_yx, scale_factor):
        """Resolve output Y/X from a required scale factor."""
        scale_factor = float(scale_factor)
        if scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be > 0, got {scale_factor}"
            )
        resolved_y = max(1, int(round(shape_yx[0] * scale_factor)))
        resolved_x = max(1, int(round(shape_yx[1] * scale_factor)))
        return resolved_y, resolved_x

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
        description="Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast, especially useful for dark images with weak bright features. For multichannel images, select which channel(s) to process.",
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
                "default": 1,
                "min": 1,
                "max": 16,
                "description": "Maximum number of parallel workers (default: 1). Higher values use more memory but are faster. Range: 1-16",
            },
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def equalize_histogram(
        image: np.ndarray,
        clip_limit: float = 0.01,
        kernel_size: int = 0,
        max_workers: int = 1,
        _source_filepath: str = None,
        channel: str = "all",
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
            Maximum number of parallel workers for processing large datasets (default: 1)
            Higher values increase speed but use more memory
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

        # equalize_adapthist returns float64 — 8 bytes per voxel. Collecting
        # every slice into a list, np.stack-ing it, and only then converting
        # back to the input dtype held three full-size copies at once (a float
        # list, the stacked float array, and the scaled result). Instead,
        # allocate the final-dtype output up front and convert each slice into
        # it as soon as it is ready, so only `max_workers` float slices are
        # ever live.
        result = np.empty(image.shape, dtype=original_dtype)
        is_integer_out = np.issubdtype(original_dtype, np.integer)

        def store(destination, float_slice):
            """Scale a CLAHE result back to the input dtype, in place."""
            if is_integer_out:
                iinfo = np.iinfo(original_dtype)
                scaled = float_slice * (iinfo.max - iinfo.min) + iinfo.min
                np.copyto(destination, scaled, casting="unsafe")
            else:
                np.copyto(destination, float_slice, casting="unsafe")

        def run_parallel(n_slices, label):
            print(
                f"Parallelizing CLAHE across {n_slices} {label} "
                f"using {max_workers} workers..."
            )

            def process_slice(idx):
                store(
                    result[idx],
                    skimage.exposure.equalize_adapthist(
                        image[idx],
                        kernel_size=kernel_size,
                        clip_limit=clip_limit,
                    ),
                )
                if (idx + 1) % max(1, n_slices // 10) == 0:
                    print(f"  Processed {idx + 1}/{n_slices} slices")

            # Threads write into disjoint slices of `result`, so no locking is
            # needed; returning None keeps nothing alive between tasks.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                list(executor.map(process_slice, range(n_slices)))
            print("CLAHE processing complete!")

        # For 4D data (TZYX), parallelize across first dimension for better performance
        if image.ndim == 4 and image.shape[0] > 1:
            run_parallel(image.shape[0], "timepoints/slices")
        elif image.ndim == 3 and image.shape[0] > 5:
            # For 3D data with many slices, also parallelize
            run_parallel(image.shape[0], "slices")
        else:
            # For 2D or small 3D data, use native implementation
            if image.ndim > 2:
                print("Processing...")
            store(
                result,
                skimage.exposure.equalize_adapthist(
                    image, kernel_size=kernel_size, clip_limit=clip_limit
                ),
            )
            if image.ndim > 2:
                print("CLAHE processing complete!")

        return result

    # simple otsu thresholding
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (semantic)",
        suffix="_otsu_semantic",
        description="Threshold image using Otsu's method to obtain a binary image. Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice. For multichannel images, select which channel(s) to process.",
        parameters={
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def otsu_thresholding(
        image: np.ndarray, dimension_order: str = "Auto", channel: str = "all"
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
            Binary label image with same shape as input (1=foreground,
            0=background). uint32 so napari/the save pipeline recognize it
            as a Labels layer (see is_label_image), matching the sibling
            "Otsu Thresholding (instance)" and "Manual Thresholding"
            functions' label-typed output.
        """
        # threshold_otsu works on any numeric dtype directly; forcing an
        # 8-bit downcast here would quantize away most of the dynamic range
        # for uint16 microscopy data (whose true max is often far below
        # 65535) and measurably shift the computed threshold.

        # Handle different dimension orders
        if dimension_order in [
            "TYX",
            "ZYX",
            "TCYX",
            "TZYX",
            "ZCYX",
            "TZCYX",
            "TCZYX",
        ]:
            # Process frame-by-frame or slice-by-slice
            result = np.zeros_like(image, dtype=np.uint32)

            # Determine which axes to iterate over
            if len(image.shape) == 3:  # TYX or ZYX
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = (image[i] > thresh).astype(np.uint32)
            elif len(image.shape) == 4:  # TCYX, TZYX, ZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        thresh = skimage.filters.threshold_otsu(image[i, j])
                        result[i, j] = (image[i, j] > thresh).astype(
                            np.uint32
                        )
            elif len(image.shape) == 5:  # TZCYX
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        for k in range(image.shape[2]):
                            thresh = skimage.filters.threshold_otsu(
                                image[i, j, k]
                            )
                            result[i, j, k] = (
                                image[i, j, k] > thresh
                            ).astype(np.uint32)
            else:
                # Fallback for unexpected shapes
                thresh = skimage.filters.threshold_otsu(image)
                result = (image > thresh).astype(np.uint32)

            return result
        elif dimension_order == "CYX":
            # Process each channel independently
            if len(image.shape) >= 3:
                result = np.zeros_like(image, dtype=np.uint32)
                for i in range(image.shape[0]):
                    thresh = skimage.filters.threshold_otsu(image[i])
                    result[i] = (image[i] > thresh).astype(np.uint32)
                return result
            else:
                # Fallback if not actually multi-channel
                thresh = skimage.filters.threshold_otsu(image)
                return (image > thresh).astype(np.uint32)
        else:
            # YX or Auto: process as single image
            thresh = skimage.filters.threshold_otsu(image)
            return (image > thresh).astype(np.uint32)

    # instance segmentation
    @BatchProcessingRegistry.register(
        name="Otsu Thresholding (instance)",
        suffix="_otsu_labels",
        description="Threshold image using Otsu's method to obtain a multi-label image. Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice. For multichannel images, select which channel to process.",
        parameters={
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def otsu_thresholding_instance(
        image: np.ndarray, dimension_order: str = "Auto", channel: str = "all"
    ) -> np.ndarray:
        """
        Threshold image using Otsu's method to create instance labels.

        Args:
            image: Input image (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.)
            dimension_order: Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.)
                            T and C axes are processed independently (each
                            timepoint/channel gets its own Otsu threshold and
                            labels restart at 1). Z is kept together with
                            Y,X so a 3D object spanning multiple Z slices
                            keeps a single label. YX/Auto (2D only):
                            processed as a single image.
            channel: For multichannel images, restrict processing to a single channel
                    (extracted upstream before this function runs).

        Returns:
            Label image with same shape as input (0=background, 1,2,3...=objects)
        """
        # threshold_otsu works on any numeric dtype directly; forcing an
        # 8-bit downcast here would quantize away most of the dynamic range
        # for uint16 microscopy data (whose true max is often far below
        # 65535) and measurably shift the computed threshold.
        result = np.zeros_like(image, dtype=np.uint32)
        for index, block in _iter_dimension_blocks(image, dimension_order):
            thresh = skimage.filters.threshold_otsu(block)
            result[index] = skimage.measure.label(
                block > thresh, connectivity=block.ndim
            ).astype(np.uint32)
        return result

    # simple thresholding
    @BatchProcessingRegistry.register(
        name="Manual Thresholding (8-bit)",
        suffix="_thresh",
        description="Threshold image using a fixed threshold to obtain a binary image. For multichannel images, select which channel(s) to process.",
        parameters={
            "threshold": {
                "type": int,
                "default": 128,
                "min": 0,
                "max": 255,
                "description": "Threshold value",
            },
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def simple_thresholding(
        image: np.ndarray, threshold: int = 128, channel: str = "all"
    ) -> np.ndarray:
        """
        Threshold image using a fixed threshold.

        Returns a binary label image (1=foreground, 0=background) rather
        than a 0/255 intensity image, so it's recognized as a Labels layer
        (uint32, see is_label_image) instead of a plain Image layer,
        matching the sibling "Otsu Thresholding (semantic/instance)"
        functions.
        """
        # convert to 8-bit so `threshold` (0-255) means the same thing
        # regardless of the input's original dtype/range
        image = skimage.img_as_ubyte(image)
        return (image > threshold).astype(np.uint32)

    # remove small objects
    @BatchProcessingRegistry.register(
        name="Remove Small Labels",
        suffix="_rm_small",
        description="Remove small labels from label images. Streams TIFF/Zarr stacks plane by plane, so memory stays flat regardless of stack size.",
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
        image: np.ndarray,
        min_size: int = 100,
        _source_filepath: str = None,
        _output_folder: str = None,
        _output_suffix: str = None,
    ) -> np.ndarray:
        """
        Remove small labels from label images.

        Works for 2D, 3D, and higher dimensional label images (4D = TZYX, CZYX, etc.).
        For 4D+ data, processes each 3D volume independently.
        Removes connected components (objects) whose area (2D) or volume (3D)
        is smaller than or equal to min_size.

        When the batch widget supplies a TIFF/Zarr source path and an output
        folder, the stack is streamed plane by plane and the written path is
        returned (``skip_load`` keeps the worker from loading it densely).
        Otherwise the in-memory array path below is used unchanged.

        Parameters
        ----------
        image : np.ndarray
            Label image (2D, 3D, 4D, or higher dimensional). ``None`` under
            ``skip_load``.
        min_size : int
            Minimum size threshold in pixels/voxels. Objects with size <= min_size are removed.

        Returns
        -------
        np.ndarray or str
            Label image with small objects removed, or the path written when
            streaming from disk.
        """
        # --- Streaming path: never materialise the stack ------------------
        if _source_filepath and _output_folder and _output_suffix:
            suffix = os.path.splitext(_source_filepath)[1].lower()
            if suffix in (".tif", ".tiff", ".zarr") or os.path.isdir(
                _source_filepath
            ):
                os.makedirs(_output_folder, exist_ok=True)
                stem = os.path.splitext(
                    os.path.basename(_source_filepath.rstrip("/"))
                )[0]
                output_path = os.path.join(
                    _output_folder, f"{stem}{_output_suffix}.tif"
                )
                return _stream_remove_small_labels(
                    _source_filepath, output_path, min_size
                )

        if image is None:
            # skip_load left the array unloaded and the format isn't one we
            # can stream — read it here rather than silently doing nothing.
            from napari_tmidas._file_selector import load_image_file

            image = np.asarray(load_image_file(_source_filepath))

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

    # skip_load=True: the worker must NOT call load_image_file for this
    # function.  A tracked stack that is 90 MB compressed can be 70 GB dense,
    # and a dense load plus the same-sized output allocation is what got the
    # process OOM-killed.  The function streams the file itself instead.
    remove_small_objects.skip_load = True

    @BatchProcessingRegistry.register(
        name="Invert Image",
        suffix="_inverted",
        description="Invert pixel values in the image using scikit-image's invert function. For multichannel images, select which channel(s) to process.",
        parameters={
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def invert_image(image: np.ndarray, channel: str = "all") -> np.ndarray:
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
        # skimage.util.invert already returns a new array, so the defensive
        # copy the input used to get was a wasted full-size allocation.
        return skimage.util.invert(image)

    def _semantic_to_instance_block(mask_block: np.ndarray) -> np.ndarray:
        """Connected-component labeling for a single 2D or 3D (ZYX) mask block."""
        connectivity = mask_block.ndim
        if np.max(mask_block) > 1:
            # Get unique non-zero class values
            class_values = np.unique(mask_block)
            class_values = class_values[
                class_values > 0
            ]  # Remove background (0)

            # Create an empty output mask
            result = np.zeros(mask_block.shape, dtype=np.uint32)

            # Process each class.  Classes partition the block, so their
            # components never overlap — writing each class straight into the
            # result under a `where` mask is equivalent to the old
            # `np.maximum(result, labeled)` but avoids allocating a fresh
            # full-size array on every iteration.
            label_offset = 0
            for class_val in class_values:
                # Pass the boolean mask directly; the old .astype(np.uint8)
                # materialised an extra copy of the block per class
                labeled = skimage.measure.label(
                    mask_block == class_val, connectivity=connectivity
                )

                # Skip if no components found
                n_components = int(labeled.max())
                if n_components == 0:
                    continue

                # Offset so labels stay unique across classes.  measure.label
                # numbers components 1..n contiguously, so the next offset is
                # simply the running total — no full rescan of `result`.
                nonzero = labeled > 0
                np.add(labeled, label_offset, out=labeled, where=nonzero)
                np.copyto(result, labeled, casting="unsafe", where=nonzero)
                label_offset += n_components

            return result
        else:
            # For binary masks, just find connected components
            return skimage.measure.label(
                mask_block > 0, connectivity=connectivity
            ).astype(np.uint32)

    @BatchProcessingRegistry.register(
        name="Semantic to Instance Segmentation",
        suffix="_instance",
        description="Convert semantic segmentation masks to instance segmentation labels using connected components. Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice. For multichannel images, select which channel to process.",
        parameters={
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def semantic_to_instance(
        image: np.ndarray, dimension_order: str = "Auto", channel: str = "all"
    ) -> np.ndarray:
        """
        Convert semantic segmentation masks to instance segmentation labels.

        This function takes a binary or multi-class semantic segmentation mask and
        converts it to an instance segmentation by finding connected components.
        Each connected region receives a unique label.

        Args:
            image: Input mask (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.)
            dimension_order: Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.)
                            T and C axes are processed independently (labels
                            restart per timepoint/channel). Z is kept
                            together with Y,X so a 3D object spanning
                            multiple Z slices keeps a single label. YX/Auto
                            (2D only): processed as a single block.
            channel: For multichannel images, restrict processing to a single channel
                    (extracted upstream before this function runs).

        Returns:
            Instance segmentation with unique labels for each connected component
            (labels restart per timepoint/channel when processed independently)
        """
        # Blocks are only ever read, so iterate the input directly instead of
        # allocating a full-size copy of it first.
        result = np.zeros(image.shape, dtype=np.uint32)
        for index, block in _iter_dimension_blocks(image, dimension_order):
            result[index] = _semantic_to_instance_block(block)
        return result

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
    description="Convert binary images to label images (connected components). Supports dimension_order hint (TYX, ZYX, etc.) to process frame-by-frame or slice-by-slice. For multichannel images, select which channel to process.",
    parameters={
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to process (automatically detected from multichannel images)",
        },
    },
)
def binary_to_labels(
    image: np.ndarray, dimension_order: str = "Auto", channel: str = "all"
) -> np.ndarray:
    """
    Convert binary images to label images (connected components).

    Args:
        image: Input binary image (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.)
        dimension_order: Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.)
                        T and C axes are processed independently (labels
                        restart per timepoint/channel). Z is kept together
                        with Y,X so a 3D object spanning multiple Z slices
                        keeps a single label instead of one per slice.
                        YX/Auto (2D only): processed as a single block.
        channel: For multichannel images, restrict processing to a single channel
                (extracted upstream before this function runs).

    Returns:
        Label image with unique labels per connected component
        (labels restart per timepoint/channel when processed independently)
    """
    # No copy: blocks are only read, and measure.label does not modify its
    # input, so copying the whole stack just to discard it doubled peak memory.
    result = np.zeros(image.shape, dtype=np.uint32)
    for index, block in _iter_dimension_blocks(image, dimension_order):
        # Assigning into the uint32 result converts in place; an explicit
        # .astype() would build a second full-size array first.
        result[index] = skimage.measure.label(block, connectivity=block.ndim)
    return result


@BatchProcessingRegistry.register(
    name="Convert to 8-bit (uint8)",
    suffix="_uint8",
    description="Convert image data to 8-bit (uint8) format with proper scaling. For multichannel images, select which channel(s) to process.",
    parameters={
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to process (automatically detected from multichannel images)",
        },
    },
)
def convert_to_uint8(image: np.ndarray, channel: str = "all") -> np.ndarray:
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
    # The obvious `img_as_ubyte(rescale_intensity(image))` builds a full-size
    # float64 image and several more temporaries just to emit one byte per
    # voxel — ~6x the input in peak memory. Do the same arithmetic one YX plane
    # at a time into a pre-allocated uint8 result instead.  uint8 input is not
    # short-circuited: rescale_intensity stretches it to the full range too.
    imin = float(image.min())
    imax = float(image.max())

    result = np.empty(image.shape, dtype=np.uint8)
    if imax == imin:
        # Constant image: rescale_intensity clips the single value into the
        # output range, so 0 stays 0 and anything else saturates to 255.
        result.fill(0 if imin <= 0 else 255)
        return result

    span = imax - imin
    leading = image.shape[:-2] if image.ndim > 2 else ()
    plane_shape = image.shape[-2:] if image.ndim >= 2 else image.shape
    scratch = np.empty(plane_shape, dtype=np.float64)

    for index in (np.ndindex(*leading) if leading else [()]):
        # Same order of operations as rescale_intensity followed by
        # img_as_ubyte — subtract, divide, then scale by 255 — so values
        # landing exactly on .5 round identically.
        np.subtract(image[index], imin, out=scratch, casting="unsafe")
        np.divide(scratch, span, out=scratch)
        np.clip(scratch, 0, 1, out=scratch)
        np.multiply(scratch, 255, out=scratch)
        np.rint(scratch, out=scratch)
        result[index] = scratch

    return result


@BatchProcessingRegistry.register(
    name="Resize Image by YX Scale (skimage)",
    suffix="_yx_resized",
    description="Resize intensity images by a YX scale factor for faster downstream processing while preserving T/Z axes. For multichannel images, select which channel(s) to process.",
    parameters={
        "scale_factor": {
            "type": float,
            "default": 0.5,
            "min": 0.0001,
            "max": 100.0,
            "description": "YX scale factor. For example, 0.5 resizes both Y and X to half size.",
        },
        "dim_order": {
            "type": str,
            "default": "auto",
            "options": ["auto", "YX", "ZYX", "TYX", "TZYX", "TCZYX"],
            "description": "Input dimension order. 'auto' maps ndim 2->YX, 3->ZYX, 4->TZYX, 5->TCZYX.",
        },
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to process (automatically detected from multichannel images)",
        },
    },
)
def resize_image_fixed_yx(
    image: np.ndarray,
    scale_factor: float = 0.5,
    dim_order: str = "auto",
    channel: str = "all",
) -> np.ndarray:
    """
    Resize image YX plane(s) to fixed dimensions using skimage.

    Supports any dimension order ending in YX (YX, CYX, TYX, ZYX, TCYX, TZYX,
    ZCYX, TZCYX, TCZYX, …).  All leading T/C/Z dimensions are preserved;
    only the YX plane is resized.
    """
    from skimage.transform import resize

    target_y, target_x = _resolve_resize_target(image.shape[-2:], scale_factor)

    # Only the trailing YX plane is resized; any leading axes (T, C, Z, …) are
    # preserved as-is, so every dimension order ending in YX is supported.
    dim_order = str(dim_order).upper()
    if dim_order != "AUTO":
        if not dim_order.endswith("YX"):
            raise ValueError(
                f"Unsupported dim_order '{dim_order}'. The last two axes must "
                "be Y and X (e.g. YX, CYX, TYX, ZYX, TZYX, TCZYX) or 'auto'."
            )
        if len(dim_order) != image.ndim:
            raise ValueError(
                f"dim_order '{dim_order}' is incompatible with image.ndim={image.ndim}"
            )
    elif image.ndim < 2:
        raise ValueError(
            f"Resizing requires at least 2 dimensions, got {image.ndim}"
        )

    def _resize_2d(slice_2d: np.ndarray) -> np.ndarray:
        anti_aliasing = (
            target_y < slice_2d.shape[0] or target_x < slice_2d.shape[1]
        )
        resized = resize(
            slice_2d,
            (target_y, target_x),
            order=1,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=anti_aliasing,
            clip=True,
        )
        return resized.astype(slice_2d.dtype, copy=False)

    def _resize_block(block):
        """Resize a numpy block — last 2 dims are Y, X."""
        if block.ndim == 2:
            return _resize_2d(block)
        lead = block.shape[:-2]
        flat = block.reshape(-1, block.shape[-2], block.shape[-1])
        out = np.empty((flat.shape[0], target_y, target_x), dtype=block.dtype)
        for i in range(flat.shape[0]):
            out[i] = _resize_2d(flat[i])
        return out.reshape(*lead, target_y, target_x)

    # --- Dask-native path: process lazily, one chunk at a time ---
    try:
        import dask.array as da
        if isinstance(image, da.Array):
            # Rechunk so Y and X axes are always whole planes inside each block.
            # Leading dims (T, C, Z, …) keep their original chunk sizes so
            # we never materialize more than one full-resolution YX plane per worker.
            img_rc = image.rechunk(
                image.chunks[:-2] + (image.shape[-2], image.shape[-1])
            )
            new_chunks = img_rc.chunks[:-2] + ((target_y,), (target_x,))
            print(
                f"Resize (dask): {image.shape} → "
                f"{image.shape[:-2] + (target_y, target_x)}, "
                f"processing {img_rc.npartitions} blocks lazily"
            )
            return da.map_blocks(
                _resize_block,
                img_rc,
                chunks=new_chunks,
                dtype=image.dtype,
            )
    except ImportError:
        pass

    # --- NumPy fallback path ---
    if dim_order == "YX":
        return _resize_2d(image)

    return _resize_block(image)


# ============================================================================
# OME-Zarr native resize (writes output zarr directly)
# ============================================================================

if SKIMAGE_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Resize Zarr by YX Scale (OME-Zarr native)",
        suffix="_yx_resized",
        description=(
            "Resize a zarr file using OME-Zarr native I/O. "
            "Reads the source zarr lazily via dask, resizes each YX plane with "
            "skimage (chunk-by-chunk), and writes a new OME-Zarr with preserved "
            "axes metadata and the source pyramid depth using a YX scale factor. "
            "Falls back to the skimage path for TIFF inputs."
        ),
        parameters={
            "scale_factor": {
                "type": float,
                "default": 0.5,
                "min": 0.0001,
                "max": 100.0,
                "description": "YX scale factor. For example, 0.5 resizes both Y and X to half size.",
            },
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process.",
            },
        },
    )
    def resize_zarr_native(
        image: np.ndarray,
        scale_factor: float = 0.5,
        channel: str = "all",
        _source_filepath: str = None,
        _output_folder: str = None,
        _output_suffix: str = "_yx_resized",
        _output_format: str = "zarr",
    ):
        """
        Resize a zarr (or TIFF) to fixed YX dimensions using OME-Zarr native I/O.

        For zarr inputs the full pipeline is:
          1. open source zarr via ``zarr`` + ``dask.array`` — no data loaded yet
          2. build a lazy resize graph with ``ome_zarr.dask_utils.resize``
          3. write the result with ``ome_zarr.writer.write_image``, which computes
             the dask graph in chunks and writes a proper OME-Zarr file with axes
             metadata and a 4-level multiscale pyramid
        For TIFF inputs the function falls back to the skimage resize path and
        returns a numpy array (the normal saving pipeline handles writing).
        """
        source = _source_filepath or ""
        is_zarr = source.lower().endswith(".zarr") or (
            os.path.isdir(source)
            and os.path.exists(os.path.join(source, ".zattrs"))
        )

        # ── ZARR PATH ────────────────────────────────────────────────────────
        if is_zarr:
            try:
                import zarr as zarr_lib
                import dask.array as da
                from ome_zarr.dask_utils import resize as dask_resize
                from ome_zarr.writer import write_image
                from ome_zarr.scale import Scaler
                import json

                # ── Read source metadata ──────────────────────────────────
                zroot = zarr_lib.open(source, mode="r")
                attrs = {}
                zattrs_path = os.path.join(source, ".zattrs")
                if os.path.exists(zattrs_path):
                    with open(zattrs_path) as f:
                        attrs = json.load(f)

                multiscales = attrs.get("multiscales", [])
                ms = multiscales[0] if multiscales else {}
                axes = ms.get("axes", None)

                # Infer axes list from metadata or fallback by ndim
                def _axes_for_ndim(n):
                    defaults = {2: "yx", 3: "zyx", 4: "tzyx", 5: "tczyx"}
                    return list(defaults.get(n, "tczyx"[-n:]))

                # ── Open full-resolution array ────────────────────────────
                zarr_arrays = list(zroot.array_keys())
                if not zarr_arrays:
                    # Group with multiscale datasets
                    datasets = ms.get("datasets", [{"path": "0"}])
                    arr_path = datasets[0].get("path", "0")
                else:
                    arr_path = zarr_arrays[0]

                src_arr = zroot[arr_path]
                src_da = da.from_zarr(src_arr)
                # Save original shape and source level-0 scale for
                # computing correct coordinate transforms after resize.
                src_shape_full = src_da.shape
                src_scale_l0 = None
                src_datasets = ms.get("datasets", [])
                if src_datasets:
                    src_ctf = src_datasets[0].get(
                        "coordinateTransformations", [{}]
                    )
                    if src_ctf and src_ctf[0].get("type") == "scale":
                        src_scale_l0 = src_ctf[0]["scale"]

                # ── Apply channel selection ───────────────────────────────
                # Determine channel axis from OME axes metadata
                ch_axis = None
                if axes:
                    for i, ax in enumerate(axes):
                        name = (ax.get("name", "") if isinstance(ax, dict)
                                else str(ax)).lower()
                        atype = (ax.get("type", "") if isinstance(ax, dict)
                                 else "").lower()
                        if name in ("c", "channel", "ch") or atype == "channel":
                            ch_axis = i
                            break

                if (
                    channel != "all"
                    and ch_axis is not None
                ):
                    ch_idx = int(channel)
                    print(
                        f"Extracting channel {ch_idx} (axis {ch_axis}) "
                        f"from shape {src_da.shape}"
                    )
                    src_da = src_da[
                        tuple(
                            ch_idx if i == ch_axis else slice(None)
                            for i in range(src_da.ndim)
                        )
                    ]
                    # Remove channel from axes list
                    if axes and ch_axis < len(axes):
                        axes = [a for i, a in enumerate(axes) if i != ch_axis]

                # ── Build target shape ────────────────────────────────────
                target_y, target_x = _resolve_resize_target(
                    src_da.shape[-2:], scale_factor
                )
                out_shape = src_da.shape[:-2] + (target_y, target_x)
                print(
                    f"Resize (OME-Zarr native): {src_da.shape} → {out_shape}, "
                    f"dtype={src_da.dtype}"
                )

                resized_da = dask_resize(
                    src_da.astype(float),
                    out_shape,
                    order=1,
                    mode="reflect",
                    anti_aliasing=(
                        target_y < src_da.shape[-2]
                        or target_x < src_da.shape[-1]
                    ),
                ).astype(src_da.dtype)

                # ── Build output path ─────────────────────────────────────
                basename = os.path.basename(source)
                name_no_ext = os.path.splitext(basename)[0]
                # Strip trailing .zarr from name if present
                if name_no_ext.endswith(".zarr"):
                    name_no_ext = name_no_ext[:-5]
                suffix = _output_suffix or "_yx_resized"
                out_dir = _output_folder or os.path.dirname(source)
                out_path = os.path.join(out_dir, f"{name_no_ext}{suffix}.zarr")

                print(f"Writing OME-Zarr → {out_path}")

                # ── Write OME-Zarr ────────────────────────────────────────
                # Use FSStore with '/' key separator so napari-ome-zarr can
                # read chunk files using its default FSStore path convention.
                from zarr.storage import FSStore as _FSStore
                out_store = _FSStore(
                    out_path,
                    key_separator="/",
                    mode="w",
                    auto_mkdir=True,
                )
                out_group = zarr_lib.open_group(out_store, mode="w")

                axes_for_writer = axes if axes else _axes_for_ndim(resized_da.ndim)

                # Preserve the source pyramid depth: if the source has N
                # levels, write N levels; if it has only 1, write 1.
                src_n_levels = max(len(src_datasets), 1)
                write_image(
                    image=resized_da,
                    group=out_group,
                    scaler=Scaler(
                        max_layer=src_n_levels - 1,
                        method="nearest",
                        downscale=2,
                    ),
                    axes=axes_for_writer,
                    compute=True,
                )

                # ── Fix coordinate transforms in output .zattrs ──────────
                try:
                    out_zattrs_path = os.path.join(out_path, ".zattrs")
                    out_attrs_cur = json.load(open(out_zattrs_path))
                    out_ms = out_attrs_cur.get("multiscales", [{}])[0]
                    out_datasets = out_ms.get("datasets", [])

                    if src_scale_l0 and out_datasets:
                        # Find Y and X axis indices in the SOURCE axes.
                        orig_axes = attrs.get("multiscales", [{}])[0].get(
                            "axes", None
                        )
                        ndim_src = len(src_scale_l0)
                        y_idx = ndim_src - 2
                        x_idx = ndim_src - 1
                        if orig_axes:
                            for _i, _ax in enumerate(orig_axes):
                                _n = (
                                    _ax.get("name", "")
                                    if isinstance(_ax, dict)
                                    else str(_ax)
                                ).lower()
                                if _n == "y":
                                    y_idx = _i
                                elif _n == "x":
                                    x_idx = _i

                        # Source pixel size in Y and X.
                        src_y_scale = src_scale_l0[y_idx]
                        src_x_scale = src_scale_l0[x_idx]
                        # Physical pixel size in resized image.
                        new_y_scale = src_y_scale * (
                            src_shape_full[y_idx] / target_y
                        )
                        new_x_scale = src_x_scale * (
                            src_shape_full[x_idx] / target_x
                        )

                        # Build level-0 scale based on source, swapping
                        # Y and X.  Also remove C entry if channel was
                        # extracted.
                        lvl0_scale = list(src_scale_l0)
                        lvl0_scale[y_idx] = new_y_scale
                        lvl0_scale[x_idx] = new_x_scale
                        if channel != "all" and ch_axis is not None:
                            lvl0_scale = [
                                v
                                for i, v in enumerate(lvl0_scale)
                                if i != ch_axis
                            ]
                            # Re-resolve Y/X indices after C removal.
                            y_idx_out = y_idx - (1 if ch_axis < y_idx else 0)
                            x_idx_out = x_idx - (1 if ch_axis < x_idx else 0)
                        else:
                            y_idx_out = y_idx
                            x_idx_out = x_idx

                        for n, ds in enumerate(out_datasets):
                            level_scale = list(lvl0_scale)
                            level_scale[y_idx_out] = lvl0_scale[y_idx_out] * (
                                2 ** n
                            )
                            level_scale[x_idx_out] = lvl0_scale[x_idx_out] * (
                                2 ** n
                            )
                            ds["coordinateTransformations"] = [
                                {"type": "scale", "scale": level_scale}
                            ]

                        json.dump(
                            out_attrs_cur,
                            open(out_zattrs_path, "w"),
                            indent=2,
                        )
                        print(
                            f"Updated coordinate transforms: "
                            f"level 0 Y/X scale = {new_y_scale:.4f}"
                        )
                except Exception as _e:
                    print(f"Warning: coord transform update failed: {_e}")

                # ── Copy / build omero metadata for channel contrast ─────
                try:
                    # Sample level-0 output to get channel contrast limits.
                    # Read a sparse set of (T, Z) planes so this stays fast.
                    import zarr as _zarr_mod
                    lv0_arr = _zarr_mod.open_array(
                        os.path.join(out_path, "0"), mode="r"
                    )
                    out_ndim = lv0_arr.ndim  # 4 or 5
                    n_channels = (
                        lv0_arr.shape[ch_axis if ch_axis is not None else 1]
                        if out_ndim == 5
                        else 1
                    )
                    # Sparse T and Z indices for sampling
                    T_out = lv0_arr.shape[0]
                    Z_out = lv0_arr.shape[-3] if out_ndim == 5 else 1
                    t_idxs = sorted(
                        set(
                            int(i)
                            for i in np.linspace(0, T_out - 1, min(5, T_out))
                        )
                    )
                    z_idxs = sorted(
                        set(
                            int(i)
                            for i in np.linspace(0, Z_out - 1, min(15, Z_out))
                        )
                    )

                    omero_channels = []
                    # Determine channel axis in the OUTPUT array.
                    out_ch_axis = ch_axis
                    if channel != "all" and ch_axis is not None:
                        out_ch_axis = None  # C dimension was removed

                    src_omero = attrs.get("omero", {})
                    src_ch_list = src_omero.get("channels", [])

                    n_out_channels = (
                        1
                        if out_ch_axis is None
                        else lv0_arr.shape[out_ch_axis]
                    )
                    for oc in range(n_out_channels):
                        # Build slice for this channel
                        samples = []
                        for t in t_idxs:
                            for z in z_idxs:
                                if out_ch_axis is not None and out_ndim == 5:
                                    sl = tuple(
                                        t if i == 0
                                        else oc if i == out_ch_axis
                                        else z if i == out_ndim - 3
                                        else slice(None)
                                        for i in range(out_ndim)
                                    )
                                elif out_ndim == 5:
                                    sl = (t, slice(None), z,
                                          slice(None), slice(None))
                                else:
                                    sl = (t, z, slice(None), slice(None))
                                samples.append(lv0_arr[sl].ravel())
                        flat = np.concatenate(samples)
                        lo = int(np.percentile(flat, 0.5))
                        # Use the actual max from the sampled planes so that
                        # sparse bright structures (fluorescence peaks) are not
                        # clipped.  p99.5 is kept as the background-removal
                        # lower bound only.
                        mx = int(flat.max())
                        mn = int(flat.min())
                        hi = mx  # display max = actual sample max
                        # Guard: if data has extreme outlier hot-pixels
                        # (single pixels many times brighter than p99),
                        # cap at 10× p99 to avoid a pitch-dark display.
                        p99 = int(np.percentile(flat, 99.0))
                        if hi > 10 * p99 and p99 > 0:
                            hi = 10 * p99

                        # Inherit colour/label from source omero if present
                        # (adjusting channel index for single-channel extract)
                        src_idx = (
                            int(channel) if channel != "all" else oc
                        )
                        base_ch = (
                            src_ch_list[src_idx]
                            if src_idx < len(src_ch_list)
                            else {}
                        )
                        ch_entry = dict(base_ch)
                        ch_entry["window"] = {
                            "min": mn,
                            "max": mx,
                            "start": lo,
                            "end": hi,
                        }
                        ch_entry.setdefault(
                            "label", f"Channel {src_idx}"
                        )
                        ch_entry["active"] = True
                        # napari-ome-zarr needs a hex color to build a
                        # colormap; provide defaults if the source has none.
                        _default_colors = [
                            "FFFFFF", "00FF00", "FF00FF", "00FFFF",
                            "FF0000", "0000FF", "FFFF00",
                        ]
                        ch_entry.setdefault(
                            "color",
                            _default_colors[oc % len(_default_colors)],
                        )
                        omero_channels.append(ch_entry)

                    omero_out = dict(src_omero)
                    omero_out["channels"] = omero_channels
                    omero_out.setdefault("version", "0.3")
                    out_group.attrs["omero"] = omero_out
                    print(
                        f"Wrote omero window metadata for "
                        f"{n_out_channels} channel(s)"
                    )
                except Exception as _oe:
                    print(
                        f"Warning: omero metadata generation failed: {_oe}"
                    )

                print(f"✅ OME-Zarr written: {out_path}")
                # Return the path string — _file_selector.py treats this as
                # "already saved" and skips the normal write step.
                return out_path

            except Exception as e:
                print(
                    f"OME-Zarr native resize failed ({e}), "
                    "falling back to skimage path"
                )
                import traceback
                traceback.print_exc()

        # ── TIFF / FALLBACK PATH ─────────────────────────────────────────────
        return resize_image_fixed_yx(
            image,
            scale_factor=scale_factor,
            dim_order="auto",
            channel=channel,
        )

    # Mark as zarr-aware so _file_selector keeps the dask array
    resize_zarr_native._source_filepath = True


# ============================================================================
# Bright Region Extraction Functions
# ============================================================================

if SKIMAGE_AVAILABLE:

    @BatchProcessingRegistry.register(
        name="Percentile Threshold (Keep Brightest)",
        suffix="_percentile",
        description="Keep only pixels above a brightness percentile, zero out the rest. For multichannel images, select which channel(s) to process.",
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
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def percentile_threshold(
        image: np.ndarray,
        percentile: float = 90.0,
        output_type: str = "original",
        channel: str = "all",
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
        description="Remove uneven background using rolling ball algorithm (like ImageJ). For multichannel images, select which channel(s) to process.",
        parameters={
            "radius": {
                "type": int,
                "default": 50,
                "min": 5,
                "max": 200,
                "description": "Radius of rolling ball (larger = remove broader background)",
            },
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def rolling_ball_background(
        image: np.ndarray, radius: int = 50, channel: str = "all"
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

        # rolling_ball's kernel spans every axis it is handed, so passing a
        # whole TZYX stack rolls a *4D* ball: at the default radius the
        # kernel alone is 101**4 elements (832 MB) and the cost is that
        # times the pixel count -- a 5 MB stack ran 25 minutes at 13.7 GB
        # RSS without finishing.  It is also wrong: a ball spanning T/C
        # blends background across timepoints and channels.  Every
        # dimension order in _DIMENSION_ORDER_AXES ends in "YX", so the
        # background is estimated one YX plane at a time, as ImageJ does.
        if image.ndim < 2:
            raise ValueError(
                f"Rolling ball needs a 2D+ image, got {image.ndim}D"
            )

        if image.dtype == np.uint8:
            out_dtype, ceiling = np.uint8, 255
        elif image.dtype == np.uint16:
            out_dtype, ceiling = np.uint16, 65535
        else:
            out_dtype, ceiling = np.float32, None

        # Writing each corrected plane straight into the output keeps peak
        # memory at input + output + one plane; the previous version held a
        # full-size background and two full-size float copies besides.
        result = np.empty(image.shape, dtype=out_dtype)

        # ndim == 2 yields a single empty index, i.e. the image itself.
        for index in np.ndindex(*image.shape[:-2]):
            plane = image[index]
            corrected = plane.astype(np.float32) - rolling_ball(
                plane, radius=radius
            )
            np.clip(corrected, 0, ceiling, out=corrected)
            result[index] = corrected

        return result

    def _spatial_window_size_tuple(dimension_order, ndim, size):
        """
        Build a per-axis window size: `size` on axes a local filter should
        span spatially, 1 on axes that must stay independent (T, C, ...).
        Mirrors the branching used for the median filter in scipy_filters.py
        so TYX/TCYX/etc. stacks are filtered per-slice, not blended across
        T/C. A window of 1 along an axis is odd (a threshold_local
        requirement) and, for the Gaussian method, corresponds to sigma=0
        i.e. no smoothing along that axis.
        """
        if dimension_order in ["TYX", "CYX"] and ndim == 3:
            return (1, size, size)
        elif dimension_order in ["TCYX", "TZYX", "ZCYX"] and ndim == 4:
            return (1, 1, size, size)
        elif dimension_order in ["TZCYX", "TCZYX"] and ndim == 5:
            return (1, 1, 1, size, size)
        elif dimension_order == "ZYX" and ndim == 3:
            return (size, size, size)
        elif ndim >= 3:
            # Auto or an unrecognized hint on data with more than 2 axes:
            # assume the last two axes are spatial (Y, X) and everything
            # before them (T, C, Z, ...) stays independent. Filtering across
            # those leading axes by default would blend unrelated
            # frames/channels together, and for Zarr chunks with size 1
            # along those axes (typical for TCZYX layouts) a >0 map_overlap
            # depth on them is invalid and raises.
            return (1,) * (ndim - 2) + (size, size)
        else:
            # YX (2D): filter every axis
            return (size,) * ndim

    def _clamp_overlap_depth(image, depth, label):
        """
        dask.array.map_overlap requires depth to be smaller than the chunk
        size along that axis, or it raises. The dimension-aware window
        sizing above already keeps depth at 0 on axes that must stay
        independent, but as a last-resort safety net for parameter
        combinations it doesn't cover (e.g. a large window on an axis that
        happens to have small chunks), clamp instead of crashing.
        """
        clamped = dict(depth)
        for axis, d in depth.items():
            if d <= 0:
                continue
            min_chunk = min(image.chunks[axis])
            max_allowed = max(0, min_chunk - 1)
            if d > max_allowed:
                print(
                    f"Warning: {label} overlap depth {d} on axis {axis} "
                    f"exceeds smallest chunk size {min_chunk}; clamping to "
                    f"{max_allowed} (results near chunk boundaries on this "
                    "axis may be slightly less accurate)."
                )
                clamped[axis] = max_allowed
        return clamped

    def _adaptive_threshold_bright_dask(
        image, block_size: int, offset: float, dimension_order: str
    ):
        """
        Chunk-wise version of adaptive_threshold_bright using map_overlap,
        so only a halo around each chunk is exchanged instead of loading the
        whole array into memory.
        """
        import dask.array as da

        if image.dtype != np.uint8:
            image = image.map_blocks(skimage.img_as_ubyte, dtype=np.uint8)

        size_tuple = _spatial_window_size_tuple(
            dimension_order, image.ndim, block_size
        )

        # threshold_local's default method is 'gaussian', with
        # sigma = (block_size - 1) / 6 and a truncation radius of
        # 4 * sigma (skimage.filters.gaussian's default truncate=4.0).
        # That halo is wider than block_size // 2, so it's computed
        # explicitly here rather than reusing the median filter's
        # `size // 2` rule of thumb.
        depth = {}
        for axis, axis_size in enumerate(size_tuple):
            if axis_size <= 1:
                depth[axis] = 0
            else:
                sigma = (axis_size - 1) / 6.0
                depth[axis] = int(4.0 * sigma + 0.5)
        depth = _clamp_overlap_depth(image, depth, "adaptive threshold")

        print(
            f"Applying adaptive threshold to Dask array with shape "
            f"{image.shape}, chunks {image.chunks}, window={size_tuple}, "
            f"overlap depth={depth}"
        )

        def _threshold_block(block):
            threshold = skimage.filters.threshold_local(
                block, block_size=size_tuple, offset=offset
            )
            return ((block > threshold) * 255).astype(np.uint8)

        return da.map_overlap(
            _threshold_block,
            image,
            depth=depth,
            boundary="reflect",
            dtype=np.uint8,
        )

    @BatchProcessingRegistry.register(
        name="Adaptive Threshold (Bright Bias)",
        suffix="_adaptive_bright",
        description="Adaptive thresholding biased to keep bright regions. Supports dimension_order hint (TYX, ZYX, etc.) so multi-dimensional stacks are thresholded per-slice instead of blending the neighborhood across T/C. For multichannel images, select which channel(s) to process.",
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
            "channel": {
                "type": str,
                "default": "all",
                "widget_type": "channel_selector",
                "description": "Select which channel to process (automatically detected from multichannel images)",
            },
        },
    )
    def adaptive_threshold_bright(
        image: np.ndarray,
        block_size: int = 35,
        offset: float = -10.0,
        dimension_order: str = "Auto",
        channel: str = "all",
        _source_filepath: str = None,
    ) -> np.ndarray:
        """
        Apply adaptive thresholding with bias toward bright regions.

        Unlike global thresholding, adaptive thresholding calculates a threshold
        for each pixel based on its local neighborhood. The negative offset
        biases the threshold to keep more bright pixels.

        Parameters:
        -----------
        image : numpy.ndarray or dask.array
            Input image array (YX, TYX, ZYX, CYX, TCYX, TZYX, etc.). Can be a
            Dask array, in which case chunks are thresholded in place (with a
            halo) via map_overlap instead of loading the whole array.
        block_size : int
            Size of the local neighborhood for threshold calculation. Must be odd.
            Larger values consider broader neighborhoods.
        offset : float
            Value subtracted from the local mean. Negative values (like -10)
            lower the threshold, keeping more bright pixels.
        dimension_order : str
            Dimension interpretation hint (Auto, YX, TYX, ZYX, CYX, TCYX, etc.).
            If TYX/CYX/TCYX/etc.: the neighborhood is computed per frame/channel
            (2D, or 3D for ZYX) instead of spanning across T/C, which would
            otherwise blend unrelated frames into each other's local threshold.

        Returns:
        --------
        numpy.ndarray
            Binary image (255 for bright regions, 0 elsewhere)
        """
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1

        is_dask = hasattr(image, "chunks") and hasattr(image, "map_blocks")
        if is_dask:
            return _adaptive_threshold_bright_dask(
                image, block_size, offset, dimension_order
            )

        size_tuple = _spatial_window_size_tuple(
            dimension_order, image.ndim, block_size
        )

        # Axes carrying a window of 1 are independent, so the filter is
        # separable over them.  Iterating those axes keeps the float64 array
        # that threshold_local returns down to one block instead of one the
        # size of the whole stack (8 bytes per voxel on top of everything
        # else), and lets the uint8 conversion happen a block at a time too.
        n_block_axes = 0
        for size in reversed(size_tuple):
            if size == 1:
                break
            n_block_axes += 1
        n_block_axes = max(n_block_axes, 1)

        leading = image.shape[: image.ndim - n_block_axes]
        block_window = size_tuple[image.ndim - n_block_axes :]

        result = np.empty(image.shape, dtype=np.uint8)
        for index in (np.ndindex(*leading) if leading else [()]):
            block = image[index]
            if block.dtype != np.uint8:
                # img_as_ubyte is a per-element scaling, so doing it per block
                # gives the same answer as converting the whole stack at once
                block = skimage.img_as_ubyte(block)
            threshold = skimage.filters.threshold_local(
                block, block_size=block_window, offset=offset
            )
            # `binary * 255` would go through an int64 temporary the size of
            # the whole image; write the bytes straight into the result
            np.multiply(
                block > threshold, 255, out=result[index], casting="unsafe"
            )

        return result
