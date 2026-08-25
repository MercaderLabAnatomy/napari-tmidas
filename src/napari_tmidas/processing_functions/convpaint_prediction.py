# processing_functions/convpaint_prediction.py
"""
Processing functions for semantic segmentation using napari-convpaint.

This module provides functionality to run batch inference using pretrained convpaint models.
It supports 2D (YX), 3D (ZYX), time-lapse 2D (TYX), and time-lapse 3D (TZYX) data.

For time-lapse data, the function processes each timepoint independently, similar to
CAREamics denoising and other processing functions.

The functions will automatically create and manage a dedicated environment for napari-convpaint
if it's not already installed in the main environment.
"""
import os
from importlib import import_module

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Import the environment manager for convpaint
from napari_tmidas.processing_functions.convpaint_env_manager import (
    run_convpaint_in_env,
)


def _load_convpaint_model_class():
    """Import the backend Convpaint model without requiring the Qt widget layer."""
    try:
        return import_module("napari_convpaint.convpaint_model").ConvpaintModel
    except ImportError:
        return import_module("napari_convpaint").ConvpaintModel

# Check if napari-convpaint is directly available in current environment
try:
    ConvpaintModel = _load_convpaint_model_class()

    CONVPAINT_AVAILABLE = True
    USE_DEDICATED_ENV = False
    print("napari-convpaint found in current environment, using direct import")
except ImportError:
    CONVPAINT_AVAILABLE = False
    USE_DEDICATED_ENV = True
    print(
        "napari-convpaint not found in current environment, will use dedicated environment"
    )


@BatchProcessingRegistry.register(
    name="Convpaint Prediction",
    suffix="_convpaint_labels",
    description="Semantic segmentation using pretrained convpaint model. Supports YX (2D), ZYX (3D), TYX (2D+time), and TZYX (3D+time). For multichannel images, select which channel to segment.",
    parameters={
        "channel": {
            "type": str,
            "default": "all",
            "widget_type": "channel_selector",
            "description": "Select which channel to segment (automatically detected from multichannel images)",
        },
        "model_path": {
            "type": str,
            "default": "",
            "description": "Path to pretrained convpaint model (.pkl file). Leave empty to see help.",
        },
        "image_downsample": {
            "type": int,
            "default": 2,
            "min": 1,
            "max": 8,
            "description": "Downsampling factor for processing (1=no downsampling, 2=2x, etc.). Output is automatically upsampled.",
        },
        "output_type": {
            "type": str,
            "default": "semantic",
            "options": ["semantic", "instance"],
            "description": "Output type: 'semantic' (classes only) or 'instance' (each connected component labeled separately)",
        },
        "background_label": {
            "type": int,
            "default": 1,
            "min": 0,
            "max": 255,
            "description": "Label ID representing background class (will be set to 0 in output)",
        },
        "use_cpu": {
            "type": bool,
            "default": False,
            "description": "Force CPU execution even if GPU is available (useful for GPU compatibility issues)",
        },
        "force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force using dedicated environment even if napari-convpaint is available",
        },
        "z_batch_size": {
            "type": int,
            "default": 0,
            "min": 0,
            "max": 200,
            "description": "Enable Z-batching to reduce memory (0=disabled, processes all Z-planes at once). Set to 10-20 for large datasets if running out of memory. Lower values = less memory but slower.",
        },
        "n_workers": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 16,
            "description": "Number of timepoints segmented concurrently. Workers are pinned round-robin to the visible CUDA devices, so on a 2-GPU workstation n_workers=2 puts one timepoint on each GPU. Only used for time-series data. Raising this above the GPU count makes workers share a device and can exhaust GPU memory.",
        },
    },
)
def convpaint_predict(
    image: np.ndarray,
    channel: str = "all",
    model_path: str = "",
    image_downsample: int = 2,
    output_type: str = "semantic",
    background_label: int = 1,
    use_cpu: bool = False,
    force_dedicated_env: bool = False,
    z_batch_size: int = 0,
    n_workers: int = 1,
    tmp_dir: str = None,
    _source_filepath: str = None,
    _output_folder: str = None,
    _output_suffix: str = None,
    _output_format: str = "tiff",
) -> "str | np.ndarray":
    """
    Semantic segmentation using pretrained convpaint models.

    This function loads a pretrained convpaint model from a .pkl checkpoint file
    and uses it to segment the input image. The function supports YX (2D), ZYX (3D),
    TYX (2D with time), and TZYX (3D with time) data formats. For data with time
    dimension, the function iterates through each timepoint independently.

    If napari-convpaint is not installed in the main environment, a dedicated virtual
    environment will be created and used automatically.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to segment. Supported formats:
        - YX: 2D image
        - ZYX: 3D image (Z-stack)
        - TYX: 2D time series
        - TZYX: 3D time series
    model_path : str
        Path to the pretrained convpaint model (.pkl file).
        Leave empty to see help message.
    image_downsample : int
        Downsampling factor for processing (default: 2).
        The image is downsampled during processing to reduce memory usage,
        and the output is automatically upsampled to match the input dimensions.
        Use higher values (e.g., 4) for very large images.
    output_type : str
        Output type: 'semantic' or 'instance' (default: 'semantic').
        - 'semantic': Each class has the same label value (e.g., all class 1 objects = 1)
        - 'instance': Each connected component gets a unique label (uses connected components)
    background_label : int
        Label ID representing the background class (default: 1).
        All pixels with this label value will be set to 0 in the output.
        Set to 0 if background is already labeled as 0.
    use_cpu : bool
        Force CPU execution even if GPU is available (default: False).
        Useful when GPU is not compatible with PyTorch (e.g., very new GPUs).
    force_dedicated_env : bool
        If True, forces using the dedicated environment even if napari-convpaint
        is available in the current environment (default: False).

    Returns:
    --------
    numpy.ndarray
        Segmentation labels with the same spatial dimensions as the input.
        For time series, returns labels for all timepoints.

    Raises:
    -------
    ValueError
        If model_path is empty or file doesn't exist
    RuntimeError
        If segmentation fails

    Examples:
    ---------
    # 2D image
    image_2d = np.random.rand(512, 512)
    labels_2d = convpaint_predict(
        image_2d,
        model_path='/path/to/model.pkl'
    )

    # 3D Z-stack
    image_3d = np.random.rand(50, 512, 512)
    labels_3d = convpaint_predict(
        image_3d,
        model_path='/path/to/model.pkl',
        image_downsample=2
    )

    # Time-lapse 3D (TZYX)
    timelapse = np.random.rand(20, 50, 512, 512)
    labels_timelapse = convpaint_predict(
        timelapse,
        model_path='/path/to/model.pkl',
        image_downsample=2
    )

    Notes:
    ------
    - For TZYX data, each timepoint is processed independently
    - The model must be compatible with the input image dimensions
    - Downsampling can significantly reduce memory usage for large images
    - GPU processing is automatically enabled if available
    """

    # Check if model_path is provided
    if not model_path or not model_path.strip():
        raise ValueError(
            """
convpaint_predict requires a model_path parameter.

Usage:
------
1. Train or obtain a pretrained convpaint model (.pkl file)
2. Provide the path to the model:
   model_path='/path/to/your/model.pkl'

Example model paths:
- '/home/user/models/convpaint_combo_dino_gauss_3classes.pkl'
- '/mnt/data/models/my_convpaint_model.pkl'

Model Training:
---------------
To train a convpaint model, use napari-convpaint in napari or via code.
See: https://github.com/guiwitz/napari-convpaint

Image Downsample:
-----------------
The image_downsample parameter (default: 2) controls memory usage:
- 1: No downsampling (high memory, best quality)
- 2: 2x downsampling (recommended for most cases)
- 4: 4x downsampling (for very large images or limited GPU memory)

The output is automatically upsampled to match the input dimensions.
"""
        )

    # Check if model file exists
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model file not found: {model_path}\n"
            f"Please provide a valid path to a convpaint .pkl model file."
        )

    # Validate image_downsample
    if image_downsample < 1:
        raise ValueError(
            f"image_downsample must be >= 1, got {image_downsample}"
        )

    # Validate output_type
    if output_type not in ["semantic", "instance"]:
        raise ValueError(
            f"output_type must be 'semantic' or 'instance', got '{output_type}'"
        )

    # Determine if we should use dedicated environment
    use_dedicated = (
        force_dedicated_env or USE_DEDICATED_ENV or not CONVPAINT_AVAILABLE
    )

    # Print information
    print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Model path: {model_path}")
    print(f"Image downsample: {image_downsample}x")
    print(f"Output type: {output_type}")
    print(f"CPU mode: {use_cpu}")
    if z_batch_size > 0 and image.ndim >= 3 and (image.ndim == 3 and image.shape[0] < 100 or image.ndim == 4):
        print(f"Z-batching enabled: {z_batch_size} planes per batch (memory optimization)")
    print(
        f"Using {'dedicated environment' if use_dedicated else 'current environment'}"
    )

    # Detect data dimensionality
    ndim = image.ndim

    # Time-series inputs are segmented one timepoint at a time and written
    # straight to disk.  The old path densified the whole input and then
    # pre-allocated the full uint32 output beside it — on a 31x57x2720x2720
    # source that is 26 GB plus 52 GB before a single timepoint is touched,
    # which is the OOM this replaces.  One timepoint is 0.8 GB in, 1.7 GB out.
    is_time_series = ndim == 4 or (ndim == 3 and image.shape[0] >= 100)
    if is_time_series and _output_folder and _output_suffix:
        return _segment_time_series_streaming(
            image,
            model_path=model_path,
            image_downsample=image_downsample,
            use_dedicated=use_dedicated,
            use_cpu=use_cpu,
            is_3d=(ndim == 4),
            z_batch_size=z_batch_size,
            n_workers=n_workers,
            background_label=background_label,
            output_type=output_type,
            tmp_dir=tmp_dir,
            source_filepath=_source_filepath,
            output_folder=_output_folder,
            output_suffix=_output_suffix,
            output_format=_output_format,
        )

    # Anything else (2D, a single Z-stack) is one volume: materialize it if it
    # arrived lazy, then fall through to the original in-memory path.
    if hasattr(image, "compute"):
        print("Materializing lazy input for single-volume segmentation...")
        image = np.asarray(image)

    # Process image and get result
    result = None

    if ndim == 2:
        # 2D image (YX)
        print("Processing 2D image (YX)...")
        if use_dedicated:
            result = run_convpaint_in_env(
                image,
                model_path,
                image_downsample,
                use_cpu,
                tmp_dir=tmp_dir,
            )
        else:
            result = _segment_with_convpaint(image, model_path, image_downsample, use_cpu)

    elif ndim == 3:
        # Could be ZYX (3D) or TYX (2D+time)
        # We'll assume ZYX if first dimension is small (<100), otherwise TYX
        if image.shape[0] < 100:
            # Likely ZYX (3D Z-stack)
            print(f"Processing 3D image (ZYX) with {image.shape[0]} Z-planes...")
            if z_batch_size > 0 and image.shape[0] > z_batch_size:
                print(f"Z-batching: Processing in batches of {z_batch_size} planes...")
                result = _process_zyx_in_batches(
                    image,
                    model_path,
                    image_downsample,
                    use_dedicated,
                    use_cpu,
                    z_batch_size,
                    tmp_dir=tmp_dir,
                )
            elif use_dedicated:
                result = run_convpaint_in_env(
                    image,
                    model_path,
                    image_downsample,
                    use_cpu,
                    tmp_dir=tmp_dir,
                )
            else:
                result = _segment_with_convpaint(
                    image, model_path, image_downsample, use_cpu
                )
        else:
            # Likely TYX (2D time series)
            print(
                f"Processing 2D time series (TYX) with {image.shape[0]} timepoints..."
            )
            result = _process_time_series(
                image,
                model_path,
                image_downsample,
                use_dedicated,
                use_cpu,
                is_3d=False,
                z_batch_size=z_batch_size,
                tmp_dir=tmp_dir,
            )

    elif ndim == 4:
        # TZYX (3D+time)
        print(
            f"Processing 3D time series (TZYX) with {image.shape[0]} timepoints and {image.shape[1]} Z-planes..."
        )
        result = _process_time_series(
            image,
            model_path,
            image_downsample,
            use_dedicated,
            use_cpu,
            is_3d=True,
            z_batch_size=z_batch_size,
            tmp_dir=tmp_dir,
        )

    else:
        raise ValueError(
            f"Unsupported image dimensions: {image.ndim}D. "
            f"Expected 2D (YX), 3D (ZYX or TYX), or 4D (TZYX)."
        )

    # Post-process: remove background label
    if background_label > 0:
        print(f"Removing background label {background_label} (setting to 0)...")
        result[result == background_label] = 0

    # Post-process: convert semantic to instance if requested
    if output_type == "instance":
        print("Converting semantic labels to instance labels...")
        result = _convert_semantic_to_instance(result)
        print(f"Converted to instance labels. Output shape: {result.shape}")
    
    # Ensure background is 0 (remove background label if it exists)
    # This ensures pixels labeled as 0 remain 0 (background)
    result = result.astype(np.uint32)

    return result


def _resolve_gpu_assignment(n_workers: int, use_cpu: bool, n_tasks: int):
    """
    Clamp the worker count and give each worker a CUDA device.

    Returns (n_workers, gpu_ids) where gpu_ids[i] is the device for worker i,
    or None when that worker should run on CPU.
    """
    try:
        n_workers = int(n_workers)
    except (TypeError, ValueError):
        n_workers = 1
    n_workers = max(1, min(n_workers, max(1, int(n_tasks))))

    if use_cpu:
        return n_workers, [None] * n_workers

    from napari_tmidas.processing_functions.convpaint_env_manager import (
        detect_gpu_ids,
    )

    devices = detect_gpu_ids()
    if not devices:
        print("No CUDA devices detected; workers will run on CPU")
        return n_workers, [None] * n_workers

    if n_workers > len(devices):
        print(
            f"⚠️  n_workers={n_workers} exceeds the {len(devices)} "
            "visible GPU(s); workers will share devices, which can "
            "exhaust GPU memory"
        )
    assignment = [devices[i % len(devices)] for i in range(n_workers)]
    print(
        f"Distributing {n_workers} worker(s) across GPU(s) "
        f"{sorted(set(assignment))}"
    )
    return n_workers, assignment


def _postprocess_timepoint(
    labels: np.ndarray, background_label: int, output_type: str, is_3d: bool
) -> np.ndarray:
    """
    Apply the per-timepoint tail of the pipeline.

    Background removal and instance labelling used to run on the assembled
    stack.  Both are per-timepoint operations already (connected components
    are computed per volume), so doing them here keeps the full stack from
    ever needing to exist in memory, and the result is identical.
    """
    labels = np.asarray(labels)
    if background_label > 0:
        labels[labels == background_label] = 0
    if output_type == "instance":
        from skimage import measure

        labels = _apply_connected_components(
            labels, measure, ndim=3 if is_3d else 2
        )
    return labels.astype(np.uint32, copy=False)


def _segment_time_series_streaming(
    image,
    model_path,
    image_downsample,
    use_dedicated,
    use_cpu,
    is_3d,
    z_batch_size,
    n_workers,
    background_label,
    output_type,
    tmp_dir,
    source_filepath,
    output_folder,
    output_suffix,
    output_format,
) -> str:
    """
    Segment a time series timepoint by timepoint and write the result itself.

    Peak memory is n_workers timepoints, not the whole stack, so this scales
    to inputs far larger than RAM.  Timepoints are independent, which is also
    what makes them the unit of GPU parallelism: each concurrent worker runs
    its own convpaint subprocess pinned to its own device.

    Completed timepoints land in a temporary on-disk Zarr rather than being
    accumulated in RAM; the final image is then streamed out of that store in
    order.  A buffer is needed because workers finish out of order, and Zarr
    is the one format here that takes random-index writes.

    Returns the path of the file it wrote.
    """
    import concurrent.futures
    import shutil
    import tempfile
    import threading

    import zarr

    from napari_tmidas.processing_functions.ome_output_utils import (
        write_labels_with_source_metadata,
    )

    n_timepoints = int(image.shape[0])
    n_workers, gpu_ids = _resolve_gpu_assignment(
        n_workers, use_cpu, n_timepoints
    )

    # Keep the scratch store on the output filesystem.  The system temp dir is
    # tmpfs on many Linux setups, which would put this buffer back in RAM.
    scratch_root = tmp_dir or output_folder or tempfile.gettempdir()
    os.makedirs(scratch_root, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix=".convpaint-", dir=scratch_root)
    store_path = os.path.join(scratch, "labels.zarr")

    base = os.path.splitext(os.path.basename(source_filepath or "image"))[0]
    extension = ".zarr" if str(output_format).lower() == "zarr" else ".tif"
    output_path = os.path.join(
        output_folder, f"{base}{output_suffix}{extension}"
    )

    shape = tuple(int(s) for s in image.shape)
    buffer = zarr.open_array(
        store_path,
        mode="w",
        shape=shape,
        # One YX plane per chunk, matching how the buffer is read back out.
        # Chunking a whole timepoint instead would make each chunk 1.7 GB on
        # a 57x2720x2720 volume, and since reading any part of a chunk
        # decompresses all of it, the streamed write would pull that much per
        # plane.  Writing a timepoint just touches Z chunks instead of one.
        chunks=(1,) * (len(shape) - 2) + shape[-2:],
        dtype="uint32",
        zarr_format=3,
    )

    print(
        f"Segmenting {n_timepoints} timepoints with {n_workers} worker(s); "
        f"peak is {n_workers} timepoint(s) in memory, not the "
        f"{np.prod(shape) * 4 / 1e9:.1f} GB stack"
    )

    # Worker index -> device.  A thread claims a slot for its whole run so a
    # given GPU only ever has one convpaint process of ours on it.
    slots = list(range(n_workers))
    slots_lock = threading.Lock()
    local = threading.local()
    done = 0
    done_lock = threading.Lock()

    def claim_slot():
        if getattr(local, "slot", None) is None:
            with slots_lock:
                local.slot = slots.pop()
        return local.slot

    def run_timepoint(t: int):
        nonlocal done
        gpu_id = gpu_ids[claim_slot()]
        # Materialize exactly this timepoint; for a lazy input this is the
        # only point where any pixels are read.
        frame = np.asarray(image[t])

        if is_3d and z_batch_size > 0 and frame.shape[0] > z_batch_size:
            labels = _process_zyx_in_batches(
                frame,
                model_path,
                image_downsample,
                use_dedicated,
                use_cpu,
                z_batch_size,
                tmp_dir=scratch,
                gpu_id=gpu_id,
            )
        elif use_dedicated:
            labels = run_convpaint_in_env(
                frame,
                model_path,
                image_downsample,
                use_cpu,
                tmp_dir=scratch,
                gpu_id=gpu_id,
            )
        else:
            labels = _segment_with_convpaint(
                frame, model_path, image_downsample, use_cpu, gpu_id=gpu_id
            )

        buffer[t] = _postprocess_timepoint(
            labels, background_label, output_type, is_3d
        )
        del frame, labels

        with done_lock:
            done += 1
            print(
                f"   timepoint {done}/{n_timepoints} done"
                + (f" (gpu {gpu_id})" if gpu_id is not None else ""),
                flush=True,
            )

    try:
        if n_workers == 1:
            for t in range(n_timepoints):
                run_timepoint(t)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as pool:
                # list() so the first exception propagates instead of being
                # swallowed with the remaining timepoints left unwritten.
                list(pool.map(run_timepoint, range(n_timepoints)))

        dim_order = "TZYX" if is_3d else "TYX"
        print(f"💾 Writing {os.path.basename(output_path)} ({dim_order})")
        # Handed the zarr array, not a numpy one, so this streams planes
        # instead of materializing the stack it just avoided building.
        write_labels_with_source_metadata(
            buffer,
            source_filepath,
            output_path,
            output_format,
            dim_order,
        )
    finally:
        del buffer
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"✓ Segmentation complete: {output_path}")
    return output_path


def _segment_with_convpaint(
    image, model_path, image_downsample, use_cpu=False, gpu_id=None
):
    """
    Segment a single image using convpaint (direct import).

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (YX, ZYX)
    model_path : str
        Path to pretrained model
    image_downsample : int
        Downsampling factor
    use_cpu : bool
        Force CPU execution
    gpu_id : int, optional
        CUDA device to run on.  In-process execution shares one CUDA context
        across threads, so this selects the device explicitly rather than
        through CUDA_VISIBLE_DEVICES (which only the subprocess path can set
        per call, and which would race between concurrent workers here).

    Returns:
    --------
    numpy.ndarray
        Segmentation labels
    """
    import gc
    import os

    try:
        import torch
    except ImportError:
        torch = None

    # Force CPU if requested
    if use_cpu and torch is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Forcing CPU execution (GPU disabled)")

    # Load model
    print(f"Loading model from: {model_path}")
    model = ConvpaintModel(model_path=model_path)
    print("Model loaded successfully")
    
    # If forcing CPU, update the model's GPU setting
    if use_cpu and torch is not None:
        model._param.fe_use_gpu = False
        # Move model to CPU if it's on GPU
        if hasattr(model, 'fe_model') and hasattr(model.fe_model, 'device'):
            if 'cuda' in str(model.fe_model.device):
                model.fe_model.device = torch.device('cpu')
                model.fe_model.model = model.fe_model.model.cpu()
                print("  Moved feature extractor model to CPU")
    elif (
        gpu_id is not None
        and torch is not None
        and torch.cuda.is_available()
    ):
        # Move this model instance onto its assigned device, so concurrent
        # workers in this process do not all pile onto cuda:0.
        device = torch.device(f"cuda:{int(gpu_id)}")
        if hasattr(model, "fe_model") and hasattr(model.fe_model, "device"):
            model.fe_model.device = device
            model.fe_model.model = model.fe_model.model.to(device)
            print(f"  Feature extractor pinned to {device}")
    print(f"  Model has classifier: {model.classifier is not None}")
    print(f"  Model device: {model.fe_model.device}")
    print(f"  GPU enabled: {model._param.fe_use_gpu}")

    # Set downsampling if needed
    if image_downsample > 1:
        model.set_params(
            image_downsample=image_downsample,
            tile_annotations=False,
            ignore_warnings=True,
        )
        print(f"Downsampling set to: {model._param.image_downsample}x")

    # Segment
    print(f"Running segmentation on image shape: {image.shape}...")
    segmentation = model.segment(image)

    # Remove singleton dimensions if present
    segmentation = np.squeeze(segmentation)
    print(f"Segmentation complete. Output shape: {segmentation.shape}")

    # Verify output shape matches input
    if segmentation.shape != image.shape:
        print(
            f"⚠️  Warning: Shape mismatch - expected {image.shape}, got {segmentation.shape}"
        )

    # Clear memory
    del model
    gc.collect()
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return segmentation


def _process_time_series(
    image,
    model_path,
    image_downsample,
    use_dedicated,
    use_cpu,
    is_3d=False,
    z_batch_size=0,
    tmp_dir=None,
):
    """
    Process time series data by iterating through timepoints.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (TYX or TZYX)
    model_path : str
        Path to pretrained model
    image_downsample : int
        Downsampling factor
    use_dedicated : bool
        Whether to use dedicated environment
    use_cpu : bool
        Force CPU execution
    is_3d : bool
        Whether data is 3D (TZYX) or 2D (TYX)

    Returns:
    --------
    numpy.ndarray
        Segmentation labels for all timepoints
    """
    import gc
    
    n_timepoints = image.shape[0]
    print(f"Processing {n_timepoints} timepoints...")

    # Pre-allocate output array
    # For TYX: (T, Y, X)
    # For TZYX: (T, Z, Y, X)
    output_shape = image.shape
    results = np.zeros(output_shape, dtype=np.uint32)

    # Process each timepoint
    for t in range(n_timepoints):
        print(f"\nProcessing timepoint {t+1}/{n_timepoints}...")
        timepoint_img = image[t]  # (Y, X) or (Z, Y, X)

        # Segment this timepoint
        # For 3D timepoints with large Z-stacks, use batching if enabled
        if is_3d and z_batch_size > 0 and timepoint_img.shape[0] > z_batch_size:
            print(f"  Z-batching: Processing in batches of {z_batch_size} planes...")
            timepoint_result = _process_zyx_in_batches(
                timepoint_img,
                model_path,
                image_downsample,
                use_dedicated,
                use_cpu,
                z_batch_size,
                tmp_dir=tmp_dir,
            )
        elif use_dedicated:
            timepoint_result = run_convpaint_in_env(
                timepoint_img,
                model_path,
                image_downsample,
                use_cpu,
                tmp_dir=tmp_dir,
            )
        else:
            timepoint_result = _segment_with_convpaint(
                timepoint_img, model_path, image_downsample, use_cpu
            )

        # Store result
        results[t] = timepoint_result
        
        # Clean up memory after each timepoint
        del timepoint_img, timepoint_result
        gc.collect()

    print(f"\n✓ Processing complete. Output shape: {results.shape}")
    return results


def _process_zyx_in_batches(
    image,
    model_path,
    image_downsample,
    use_dedicated,
    use_cpu,
    z_batch_size,
    tmp_dir=None,
    gpu_id=None,
):
    """
    Process a 3D ZYX image in batches along the Z-axis to reduce memory usage.

    Parameters:
    -----------
    image : numpy.ndarray
        Input 3D image (Z, Y, X)
    model_path : str
        Path to pretrained model
    image_downsample : int
        Downsampling factor
    use_dedicated : bool
        Whether to use dedicated environment
    use_cpu : bool
        Force CPU execution
    z_batch_size : int
        Number of Z-planes to process at once
    gpu_id : int, optional
        CUDA device this batch should run on

    Returns:
    --------
    numpy.ndarray
        Segmentation labels for full Z-stack
    """
    import gc
    
    n_z_planes = image.shape[0]
    output_shape = image.shape
    results = np.zeros(output_shape, dtype=np.uint32)
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_z_planes / z_batch_size))
    
    print(f"Processing {n_z_planes} Z-planes in {n_batches} batches...")
    
    # Process each batch
    for batch_idx in range(n_batches):
        start_z = batch_idx * z_batch_size
        end_z = min((batch_idx + 1) * z_batch_size, n_z_planes)
        
        print(f"  Batch {batch_idx+1}/{n_batches}: Z-planes {start_z+1}-{end_z}...")
        
        # Extract batch
        batch_img = image[start_z:end_z]  # (batch_size, Y, X)
        
        # Segment batch
        if use_dedicated:
            batch_result = run_convpaint_in_env(
                batch_img,
                model_path,
                image_downsample,
                use_cpu,
                tmp_dir=tmp_dir,
                gpu_id=gpu_id,
            )
        else:
            batch_result = _segment_with_convpaint(
                batch_img, model_path, image_downsample, use_cpu, gpu_id=gpu_id
            )
        
        # Store result
        results[start_z:end_z] = batch_result
        
        # Clean up batch data
        del batch_img, batch_result
        gc.collect()
    
    print(f"✓ Z-batching complete. Output shape: {results.shape}")
    return results


def _convert_semantic_to_instance(image: np.ndarray) -> np.ndarray:
    """
    Convert semantic segmentation to instance segmentation using connected components.

    For multi-class semantic segmentation, each class is processed separately
    and assigned unique instance labels.

    Parameters:
    -----------
    image : numpy.ndarray
        Semantic segmentation mask

    Returns:
    --------
    numpy.ndarray
        Instance segmentation with unique labels
    """
    try:
        from skimage import measure
    except ImportError:
        print(
            "Warning: scikit-image not available, returning semantic labels unchanged"
        )
        return image

    # Handle different dimensionalities
    if image.ndim == 2:
        # 2D image (YX)
        return _apply_connected_components(image, measure, ndim=2)
    elif image.ndim == 3:
        # 3D image - could be ZYX (Z-stack) or TYX (time series)
        # Heuristic: if first dimension < 100, treat as ZYX (3D volume)
        # Otherwise, treat as TYX (time series, process each timepoint as 2D)
        if image.shape[0] < 100:
            # ZYX: Process as 3D volume
            return _apply_connected_components(image, measure, ndim=3)
        else:
            # TYX: Process each timepoint as 2D
            result = np.zeros_like(image, dtype=np.uint32)
            for t in range(image.shape[0]):
                result[t] = _apply_connected_components(image[t], measure, ndim=2)
            return result
    elif image.ndim == 4:
        # 4D image (TZYX)
        # Process each timepoint as 3D volume
        result = np.zeros_like(image, dtype=np.uint32)
        for t in range(image.shape[0]):
            result[t] = _apply_connected_components(image[t], measure, ndim=3)
        return result
    else:
        print(f"Warning: Unsupported dimensions {image.ndim}D for instance conversion")
        return image


def _apply_connected_components(image_nd: np.ndarray, measure, ndim: int) -> np.ndarray:
    """
    Apply connected components to 2D or 3D semantic mask.

    Parameters:
    -----------
    image_nd : numpy.ndarray
        2D or 3D semantic segmentation mask
    measure : module
        scikit-image measure module
    ndim : int
        Number of dimensions (2 or 3)

    Returns:
    --------
    numpy.ndarray
        Instance segmentation with unique labels
    """
    # Determine connectivity (2 for 2D, 3 for 3D full connectivity)
    connectivity = None  # Full connectivity (26-connected for 3D, 8-connected for 2D)

    # If the input is multi-class, process each class separately
    if np.max(image_nd) > 1:
        # Get unique non-zero class values
        class_values = np.unique(image_nd)
        class_values = class_values[class_values > 0]  # Remove background (0)

        # Create an empty output mask
        result = np.zeros_like(image_nd, dtype=np.uint32)

        # Process each class
        label_offset = 0
        for class_val in class_values:
            # Create binary mask for this class
            binary_mask = (image_nd == class_val).astype(np.uint8)

            # Find connected components (works for both 2D and 3D)
            labeled = measure.label(binary_mask, connectivity=connectivity)

            # Skip if no components found
            if np.max(labeled) == 0:
                continue

            # Add offset to avoid label overlap between classes
            labeled[labeled > 0] += label_offset

            # Add to result
            result = np.maximum(result, labeled)

            # Update offset for next class
            label_offset = np.max(result)

        return result
    else:
        # For binary masks, just find connected components (works for both 2D and 3D)
        result = measure.label(image_nd > 0, connectivity=connectivity)
        return result.astype(np.uint32)
