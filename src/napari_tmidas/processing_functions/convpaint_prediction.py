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

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Import the environment manager for convpaint
from napari_tmidas.processing_functions.convpaint_env_manager import (
    run_convpaint_in_env,
)

# Check if napari-convpaint is directly available in current environment
try:
    from napari_convpaint.conv_paint_model import ConvpaintModel

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
    description="Semantic segmentation using pretrained convpaint model. Supports YX (2D), ZYX (3D), TYX (2D+time), and TZYX (3D+time).",
    parameters={
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
    },
)
def convpaint_predict(
    image: np.ndarray,
    model_path: str = "",
    image_downsample: int = 2,
    output_type: str = "semantic",
    background_label: int = 1,
    use_cpu: bool = False,
    force_dedicated_env: bool = False,
) -> np.ndarray:
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
    print(
        f"Using {'dedicated environment' if use_dedicated else 'current environment'}"
    )

    # Detect data dimensionality
    ndim = image.ndim

    # Process image and get result
    result = None

    if ndim == 2:
        # 2D image (YX)
        print("Processing 2D image (YX)...")
        if use_dedicated:
            result = run_convpaint_in_env(image, model_path, image_downsample, use_cpu)
        else:
            result = _segment_with_convpaint(image, model_path, image_downsample, use_cpu)

    elif ndim == 3:
        # Could be ZYX (3D) or TYX (2D+time)
        # We'll assume ZYX if first dimension is small (<100), otherwise TYX
        if image.shape[0] < 100:
            # Likely ZYX (3D Z-stack)
            print(f"Processing 3D image (ZYX) with {image.shape[0]} Z-planes...")
            if use_dedicated:
                result = run_convpaint_in_env(
                    image, model_path, image_downsample, use_cpu
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
                image, model_path, image_downsample, use_dedicated, use_cpu, is_3d=False
            )

    elif ndim == 4:
        # TZYX (3D+time)
        print(
            f"Processing 3D time series (TZYX) with {image.shape[0]} timepoints and {image.shape[1]} Z-planes..."
        )
        result = _process_time_series(
            image, model_path, image_downsample, use_dedicated, use_cpu, is_3d=True
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


def _segment_with_convpaint(image, model_path, image_downsample, use_cpu=False):
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
    image, model_path, image_downsample, use_dedicated, use_cpu, is_3d=False
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
        if use_dedicated:
            timepoint_result = run_convpaint_in_env(
                timepoint_img, model_path, image_downsample, use_cpu
            )
        else:
            timepoint_result = _segment_with_convpaint(
                timepoint_img, model_path, image_downsample, use_cpu
            )

        # Store result
        results[t] = timepoint_result

    print(f"\n✓ Processing complete. Output shape: {results.shape}")
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
