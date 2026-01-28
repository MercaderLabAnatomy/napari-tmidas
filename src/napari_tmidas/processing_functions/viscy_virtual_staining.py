# processing_functions/viscy_virtual_staining.py
"""
Processing functions for virtual staining using VisCy (Virtual Staining of Cells using deep learning).

This module provides functionality to perform virtual staining of phase contrast or
brightfield microscopy images using the VSCyto3D deep learning model. The model predicts
nuclei and membrane channels from transmitted light (phase/DIC) images.

The VSCyto3D model is specifically designed for 3D imaging with:
- Input: Phase contrast or DIC 3D images
- Output: Two channels (nuclei and membrane)
- Architecture: fcmae-based U-Net
- Required Z-stack depth: 15 slices (model processes in batches of 15)

Reference:
Guo et al. (2024) "Revealing architectural order with quantitative label-free imaging and deep learning"
DOI: 10.7554/eLife.55502

Note: This requires the viscy library to be installed in a dedicated environment.
"""
from typing import Union

import numpy as np

# Import the environment manager
from napari_tmidas.processing_functions.viscy_env_manager import (
    create_viscy_env,
    is_env_created,
    run_viscy_in_env,
)

# Check if viscy is directly available in this environment
try:
    import viscy  # noqa: F401

    VISCY_AVAILABLE = True
    print("VisCy found in current environment. Using native import.")
except ImportError:
    VISCY_AVAILABLE = False
    print(
        "VisCy not found in current environment. Will use dedicated environment."
    )

from napari_tmidas._registry import BatchProcessingRegistry


def transpose_dimensions(img, dim_order):
    """
    Transpose image dimensions to match expected VisCy input (ZYX).

    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    dim_order : str
        Dimension order of the input image (e.g., 'ZYX', 'TZYX', 'YXZ')

    Returns:
    --------
    numpy.ndarray
        Transposed image
    str
        New dimension order
    bool
        Whether the image has time dimension
    """
    # Handle time dimension if present
    has_time = "T" in dim_order

    # Standardize dimension order to ZYX or TZYX
    if has_time:
        target_dims = "TZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])
    else:
        target_dims = "ZYX"
        transpose_order = [
            dim_order.index(d) for d in target_dims if d in dim_order
        ]
        new_dim_order = "".join([dim_order[i] for i in transpose_order])

    # Perform the transpose
    img_transposed = np.transpose(img, transpose_order)

    return img_transposed, new_dim_order, has_time


@BatchProcessingRegistry.register(
    name="VisCy Virtual Staining",
    suffix="_virtual_stain",
    description="Virtual staining of phase/DIC images using VSCyto3D deep learning model. Predicts nuclei and membrane channels.",
    parameters={
        "dim_order": {
            "type": str,
            "default": "ZYX",
            "description": "Dimension order of the input (e.g., 'ZYX', 'TZYX', 'YXZ')",
        },
        "z_batch_size": {
            "type": int,
            "default": 15,
            "min": 15,
            "max": 15,
            "description": "Z slices per batch (must be 15 for VSCyto3D model)",
        },
        "output_channel": {
            "type": str,
            "default": "both",
            "options": ["both", "nuclei", "membrane"],
            "description": "Which channel(s) to output: 'both' (2 channels), 'nuclei' only, or 'membrane' only",
        },
    },
)
def viscy_virtual_staining(
    image: np.ndarray,
    dim_order: str = "ZYX",
    z_batch_size: int = 15,
    output_channel: str = "both",
    _source_filepath: str = None,  # Hidden parameter
) -> np.ndarray:
    """
    Perform virtual staining on phase/DIC images using VisCy.

    This function takes a 3D phase contrast or DIC microscopy image and performs
    virtual staining using the VSCyto3D deep learning model. The model predicts
    two channels: nuclei and membrane.

    If VisCy is not available in the current environment, a dedicated virtual
    environment will be created to run VisCy.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (phase contrast or DIC microscopy)
    dim_order : str
        Dimension order of the input (e.g., 'ZYX', 'TZYX', 'YXZ') (default: "ZYX")
    z_batch_size : int
        Number of Z slices to process at once (default: 15, required by VSCyto3D)
    output_channel : str
        Which channel(s) to output: 'both', 'nuclei', or 'membrane' (default: "both")
    _source_filepath : str
        Hidden parameter for potential optimization (not currently used)

    Returns:
    --------
    numpy.ndarray
        Virtual stained image
        - If output_channel='both': shape (Z, 2, Y, X) or (T, Z, 2, Y, X)
        - If output_channel='nuclei' or 'membrane': shape (Z, Y, X) or (T, Z, Y, X)
        where channels are:
        - Channel 0: Nuclei
        - Channel 1: Membrane

    Raises:
    -------
    ValueError
        If input doesn't have Z dimension
        If Z dimension is less than 15 slices
    RuntimeError
        If VisCy environment setup fails
        If processing fails

    Examples:
    ---------
    >>> # Process a 3D phase contrast image
    >>> phase_image = np.random.rand(15, 512, 512)  # (Z, Y, X)
    >>> virtual_stain = viscy_virtual_staining(phase_image, dim_order='ZYX')
    >>> virtual_stain.shape
    (15, 2, 512, 512)  # (Z, C, Y, X)

    >>> # Get only nuclei channel
    >>> nuclei = viscy_virtual_staining(phase_image, dim_order='ZYX', output_channel='nuclei')
    >>> nuclei.shape
    (15, 512, 512)  # (Z, Y, X)
    """
    # Validate z_batch_size
    if z_batch_size != 15:
        print(
            f"Warning: VSCyto3D requires z_batch_size=15, but {z_batch_size} was provided. Using 15."
        )
        z_batch_size = 15

    # Check dimension order
    if "Z" not in dim_order:
        raise ValueError(
            "VisCy virtual staining requires 3D images with Z dimension. "
            f"Current dimension order: {dim_order}"
        )

    # Transpose dimensions if needed
    img_transposed, new_dim_order, has_time = transpose_dimensions(
        image, dim_order
    )

    # Check Z dimension size
    z_axis = new_dim_order.index("Z")
    z_size = img_transposed.shape[z_axis]

    if z_size < 15:
        raise ValueError(
            f"VisCy virtual staining requires at least 15 Z slices. "
            f"Current image has {z_size} slices. "
            "Consider using a different processing method or acquiring more Z slices."
        )

    print(f"Processing image with shape {img_transposed.shape}")
    print(f"Dimension order: {new_dim_order}")

    # Process based on whether we have time dimension
    if has_time:
        # Process each timepoint separately
        n_timepoints = img_transposed.shape[0]
        print(f"Processing {n_timepoints} timepoints...")

        results = []
        for t in range(n_timepoints):
            print(f"  Processing timepoint {t+1}/{n_timepoints}...")
            timepoint_img = img_transposed[t]  # (Z, Y, X)

            # Process this timepoint
            result = _process_single_volume(
                timepoint_img, z_batch_size, output_channel
            )
            results.append(result)

        # Stack results
        final_result = np.stack(results, axis=0)
        print(f"✓ Processing complete. Output shape: {final_result.shape}")

    else:
        # Process single volume
        final_result = _process_single_volume(
            img_transposed, z_batch_size, output_channel
        )
        print(f"✓ Processing complete. Output shape: {final_result.shape}")

    return final_result


def _process_single_volume(
    image: np.ndarray, z_batch_size: int, output_channel: str
) -> np.ndarray:
    """
    Process a single 3D volume (ZYX) through VisCy.

    Parameters:
    -----------
    image : np.ndarray
        Input image with shape (Z, Y, X)
    z_batch_size : int
        Number of Z slices to process at once
    output_channel : str
        Which channel(s) to output: 'both', 'nuclei', or 'membrane'

    Returns:
    --------
    np.ndarray
        Virtual stained image
        - If output_channel='both': shape (Z, 2, Y, X)
        - If output_channel='nuclei' or 'membrane': shape (Z, Y, X)
    """
    # Check if VisCy is available directly
    if VISCY_AVAILABLE:
        result = _run_viscy_native(image, z_batch_size)
    else:
        # Check if environment exists
        if not is_env_created():
            print("VisCy environment not found. Creating environment...")
            print(
                "This is a one-time setup and may take several minutes..."
            )
            try:
                create_viscy_env()
                print("✓ VisCy environment created successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create VisCy environment: {e}"
                )

        # Run in dedicated environment
        print("Running VisCy in dedicated environment...")
        result = run_viscy_in_env(image, z_batch_size)

    # result shape: (Z, 2, Y, X)
    # Select output channel(s)
    if output_channel == "nuclei":
        return result[:, 0, :, :]  # (Z, Y, X)
    elif output_channel == "membrane":
        return result[:, 1, :, :]  # (Z, Y, X)
    else:  # "both"
        return result  # (Z, 2, Y, X)


def _run_viscy_native(image: np.ndarray, z_batch_size: int) -> np.ndarray:
    """
    Run VisCy natively in the current environment.

    Parameters:
    -----------
    image : np.ndarray
        Input image with shape (Z, Y, X)
    z_batch_size : int
        Number of Z slices to process at once

    Returns:
    --------
    np.ndarray
        Virtual stained image with shape (Z, 2, Y, X)
    """
    import torch
    from viscy.translation.engine import VSUNet

    # Get model path from environment manager
    from napari_tmidas.processing_functions.viscy_env_manager import (
        get_model_path,
    )

    model_path = get_model_path()

    # Load the model
    model = VSUNet.load_from_checkpoint(
        model_path,
        architecture="fcmae",
        model_config={
            "in_channels": 1,
            "out_channels": 2,
            "encoder_blocks": [3, 3, 9, 3],
            "dims": [96, 192, 384, 768],
            "decoder_conv_blocks": 2,
            "stem_kernel_size": [5, 4, 4],
            "in_stack_depth": 15,
            "pretraining": False,
        },
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")

    # Process in batches
    n_z = image.shape[0]
    n_batches = (n_z + z_batch_size - 1) // z_batch_size
    all_predictions = []

    for batch_idx in range(n_batches):
        start_z = batch_idx * z_batch_size
        end_z = min((batch_idx + 1) * z_batch_size, n_z)

        # Get batch
        batch_data = image[start_z:end_z]
        actual_size = batch_data.shape[0]

        # Pad if necessary
        if actual_size < z_batch_size:
            pad_size = z_batch_size - actual_size
            batch_data = np.pad(
                batch_data, ((0, pad_size), (0, 0), (0, 0)), mode="edge"
            )

        # Normalize
        p_low, p_high = np.percentile(batch_data, [1, 99])
        batch_data = np.clip(
            (batch_data - p_low) / (p_high - p_low + 1e-8), 0, 1
        )

        # Convert to tensor: (Z, Y, X) -> (1, 1, Z, Y, X)
        batch_tensor = torch.from_numpy(batch_data.astype(np.float32))[
            None, None, :, :, :
        ]
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()

        # Run prediction
        with torch.no_grad():
            pred = model(batch_tensor)  # (1, 2, Z, Y, X)

        # Process output: (2, Z, Y, X) -> (Z, 2, Y, X)
        pred_np = pred[0].cpu().numpy().transpose(1, 0, 2, 3)[:actual_size]
        all_predictions.append(pred_np)

        # Free memory
        del batch_data, batch_tensor, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all predictions: (Z, 2, Y, X)
    full_prediction = np.concatenate(all_predictions, axis=0)

    return full_prediction
