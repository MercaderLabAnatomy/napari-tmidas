# processing_functions/careamics_denoising.py
"""
Processing functions for denoising images using CAREamics.

This module provides functionality to denoise images using various models from CAREamics,
including Noise2Void (N2V) and CARE models. The functions support both 2D and 3D data.

The functions will automatically create and manage a dedicated environment for CAREamics
if it's not already installed in the main environment.
"""
import os

import numpy as np

from napari_tmidas._registry import BatchProcessingRegistry

# Import the environment manager for CAREamics
from napari_tmidas.processing_functions.careamics_env_manager import (
    create_careamics_env,
    is_env_created,
    run_careamics_in_env,
)

# Check if CAREamics is directly available in current environment
try:
    from careamics import CAREamist

    CAREAMICS_AVAILABLE = True
    USE_DEDICATED_ENV = False
    print("CAREamics found in current environment, using direct import")
except ImportError:
    CAREAMICS_AVAILABLE = False
    USE_DEDICATED_ENV = True
    print(
        "CAREamics not found in current environment, will use dedicated environment"
    )


@BatchProcessingRegistry.register(
    name="CAREamics Denoise (N2V/CARE)",
    suffix="_denoised",
    description="Denoise images using CAREamics (Noise2Void or CARE model). Supports YX (2D), ZYX (3D), TYX (2D+time), and TZYX (3D+time).",
    parameters={
        "checkpoint_path": {
            "type": str,
            "default": "",
            "description": "Path to CAREamics model: checkpoint (.ckpt), BMZ archive (.zip), or model identifier (e.g., 'careamics/N2V_SEM_demo'). Leave empty to see help.",
        },
        "tile_size": {
            "type": str,
            "default": "128,128,32",
            "description": "Tile size as 'X,Y,Z' (e.g., '128,128,32' for 3D or '128,128' for 2D)",
        },
        "tile_overlap": {
            "type": str,
            "default": "48,48,8",
            "description": "Tile overlap as 'X,Y,Z' (e.g., '48,48,8' for 3D or '48,48' for 2D)",
        },
        "batch_size": {
            "type": int,
            "default": 1,
            "min": 1,
            "max": 16,
            "description": "Batch size for prediction (default: 1, from Flywing tutorial)",
        },
        "use_tta": {
            "type": bool,
            "default": False,
            "description": "Use test-time augmentation (default: False, from Flywing tutorial)",
        },
        "force_dedicated_env": {
            "type": bool,
            "default": False,
            "description": "Force using dedicated environment even if CAREamics is available",
        },
    },
)
def careamics_denoise(
    image: np.ndarray,
    checkpoint_path: str = "",
    tile_size: str = "128,128,32",
    tile_overlap: str = "48,48,8",
    batch_size: int = 1,
    use_tta: bool = False,
    force_dedicated_env: bool = False,
) -> np.ndarray:
    """
    Denoise images using CAREamics models (Noise2Void or CARE).

    This function loads a CAREamics model from a checkpoint file or pretrained model
    identifier and uses it to denoise the input image. The function supports YX (2D), 
    ZYX (3D), TYX (2D with time), and TZYX (3D with time) data formats. For data with 
    time dimension, the function iterates through each timepoint, similar to other 
    processing functions.

    Implementation follows CAREamics tutorials:
    - 2D (Mouse Nuclei): https://careamics.github.io/0.1/applications/Noise2Void/Mouse_Nuclei/
    - 3D (Flywing): https://careamics.github.io/0.1/applications/Noise2Void/Flywing/

    If CAREamics is not installed in the main environment, a dedicated virtual environment
    will be automatically created and managed.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to denoise. Supported formats: YX, ZYX, TYX, TZYX
    checkpoint_path : str
        Path to trained Noise2Void model checkpoint:
        - Local checkpoint: '/path/to/experiment/checkpoints/last.ckpt'
        - BMZ archive: '/path/to/model.zip'
        - Model identifier: 'careamics/N2V_SEM_demo' (from HuggingFace)
        Required. Leave empty to see help message.
    tile_size : str
        Tile size as comma-separated values 'X,Y,Z' or 'X,Y'. Default: '128,128,32'
        Examples:
            '128,128,32' for 3D data (X,Y,Z tile sizes)
            '128,128' for 2D data (X,Y tile sizes)
    tile_overlap : str
        Tile overlap as comma-separated values 'X,Y,Z' or 'X,Y'. Default: '48,48,8'
        Examples:
            '48,48,8' for 3D data (X,Y,Z overlaps)
            '48,48' for 2D data (X,Y overlaps)
    batch_size : int
        Batch size for prediction. Default: 1 (safe for most GPUs)
    use_tta : bool
        Use test-time augmentation for slightly better results (slower). Default: False
    force_dedicated_env : bool
        Force using dedicated environment even if CAREamics is available

    Returns:
    --------
    numpy.ndarray
        Denoised image with the same dimensions as the input

    Notes:
    ------
    - For 2D data (YX): Uses axes="YX", tile_size=(tile_size_y, tile_size_x)
    - For 3D data (ZYX): Uses axes="ZYX", tile_size=(tile_size_z, tile_size_y, tile_size_x)
    - For TYX data: Iterates through T dimension, processing each timepoint as YX
    - For TZYX data: Iterates through T dimension, processing each timepoint as ZYX
    - Pretrained models are automatically downloaded from HuggingFace/BioImage Model Zoo
    """
    # Parse tile size and overlap from comma-separated strings
    try:
        tile_parts = [int(x.strip()) for x in tile_size.split(',')]
        if len(tile_parts) == 2:
            tile_size_x, tile_size_y = tile_parts
            tile_size_z = 32  # Default Z for 3D
        elif len(tile_parts) == 3:
            tile_size_x, tile_size_y, tile_size_z = tile_parts
        else:
            raise ValueError("tile_size must be 'X,Y' or 'X,Y,Z'")
    except (ValueError, AttributeError) as e:
        print(f"Error parsing tile_size '{tile_size}': {e}")
        print("Using defaults: 128,128,32")
        tile_size_x, tile_size_y, tile_size_z = 128, 128, 32
    
    try:
        overlap_parts = [int(x.strip()) for x in tile_overlap.split(',')]
        if len(overlap_parts) == 2:
            tile_overlap_x, tile_overlap_y = overlap_parts
            tile_overlap_z = 8  # Default Z for 3D
        elif len(overlap_parts) == 3:
            tile_overlap_x, tile_overlap_y, tile_overlap_z = overlap_parts
        else:
            raise ValueError("tile_overlap must be 'X,Y' or 'X,Y,Z'")
    except (ValueError, AttributeError) as e:
        print(f"Error parsing tile_overlap '{tile_overlap}': {e}")
        print("Using defaults: 48,48,8")
        tile_overlap_x, tile_overlap_y, tile_overlap_z = 48, 48, 8
    
    # Check if checkpoint/model identifier is provided
    if not checkpoint_path:
        print("=" * 70)
        print("ERROR: No model checkpoint provided!")
        print("\nYou must train a Noise2Void model first using CAREamics.")
        print("See: https://careamics.github.io/0.1/applications/Noise2Void/")
        print("\nAfter training, provide the checkpoint path:")
        print("  checkpoint_path = '/path/to/experiment/checkpoints/last.ckpt'")
        print("\nAlternatively, use a pretrained model:")
        print("  checkpoint_path = 'careamics/N2V_SEM_demo'")
        print("  (Available at: https://huggingface.co/careamics)")
        print("\nNote: napari-tmidas currently supports Noise2Void models only.")
        print("=" * 70)
        return image

    # Determine whether to use dedicated environment
    use_env = force_dedicated_env or USE_DEDICATED_ENV

    careamics_denoise.thread_safe = False

    if use_env:
        print("Using dedicated CAREamics environment...")

        # First check if the environment exists, create if not
        if not is_env_created():
            print(
                "Creating dedicated CAREamics environment (this may take a few minutes)..."
            )
            create_careamics_env()
            print("Environment created successfully.")

        # Prepare arguments for the CAREamics function
        args = {
            "image": image,
            "checkpoint_path": checkpoint_path,
            "tile_size_z": tile_size_z,
            "tile_size_y": tile_size_y,
            "tile_size_x": tile_size_x,
            "tile_overlap_z": tile_overlap_z,
            "tile_overlap_y": tile_overlap_y,
            "tile_overlap_x": tile_overlap_x,
            "batch_size": batch_size,
            "use_tta": use_tta,
        }

        # Run CAREamics in the dedicated environment
        print("Running CAREamics in dedicated environment...")
        return run_careamics_in_env("predict", args)

    else:
        print("Running CAREamics in current environment...")
        # Use CAREamics directly in the current environment
        try:
            # Load model using correct CAREamics API
            print(f"Loading model: {checkpoint_path}")
            try:
                careamist = CAREamist(checkpoint_path)
                print("✓ Model loaded successfully\n")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("\nTroubleshooting:")
                print("- Ensure checkpoint file exists and is not corrupted")
                print("- Check that model was trained with compatible CAREamics version")
                print("- For pretrained models, verify identifier is correct")
                return image

            # Determine data format and process accordingly
            ndim = len(image.shape)
            print(f"Processing: {image.shape}")

            if ndim == 2:
                # YX format - 2D image
                print("Format: 2D (YX)")
                axes = "YX"
                tile_size = (tile_size_y, tile_size_x)
                tile_overlap = (tile_overlap_y, tile_overlap_x)
                
                prediction = careamist.predict(
                    source=image,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    batch_size=batch_size,
                    tta=use_tta,
                )
                
                # Squeeze output to match input shape
                prediction = np.squeeze(prediction)
                print(f"✓ Denoising complete. Output: {prediction.shape}\n")
                return prediction

            elif ndim == 3:
                # Could be ZYX (3D) or TYX (2D with time)
                # Check if first dimension is small (likely T) or large (likely Z)
                if image.shape[0] <= 10:
                    # Likely TYX - iterate through time
                    print(f"Format: TYX (2D+time, T={image.shape[0]})")
                    t_size = image.shape[0]
                    axes = "YX"
                    tile_size = (tile_size_y, tile_size_x)
                    tile_overlap = (tile_overlap_y, tile_overlap_x)
                    
                    # Initialize result array
                    result = np.zeros_like(image)
                    
                    # Process each timepoint
                    for t in range(t_size):
                        print(f"  Timepoint {t+1}/{t_size}...", end=" ")
                        
                        prediction = careamist.predict(
                            source=image[t],
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            batch_size=batch_size,
                            tta=use_tta,
                        )
                        
                        # Squeeze and store result
                        result[t] = np.squeeze(prediction)
                        print("✓")
                    
                    print(f"✓ Denoising complete. Output: {result.shape}\n")
                    return result
                else:
                    # ZYX - 3D image
                    print(f"Format: 3D (ZYX, Z={image.shape[0]})")
                    axes = "ZYX"
                    tile_size = (tile_size_z, tile_size_y, tile_size_x)
                    tile_overlap = (tile_overlap_z, tile_overlap_y, tile_overlap_x)
                    
                    prediction = careamist.predict(
                        source=image,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        batch_size=batch_size,
                        tta=use_tta,
                    )
                    
                    # Squeeze output to match input shape
                    prediction = np.squeeze(prediction)
                    print(f"✓ Denoising complete. Output: {prediction.shape}\n")
                    return prediction

            elif ndim == 4:
                # TZYX - 3D with time, iterate through time
                print(f"Format: TZYX (3D+time, T={image.shape[0]}, Z={image.shape[1]})")
                t_size = image.shape[0]
                axes = "ZYX"
                tile_size = (tile_size_z, tile_size_y, tile_size_x)
                tile_overlap = (tile_overlap_z, tile_overlap_y, tile_overlap_x)
                
                # Initialize result array
                result = np.zeros_like(image)
                
                # Process each timepoint
                for t in range(t_size):
                    print(f"  Timepoint {t+1}/{t_size}...", end=" ")
                    
                    prediction = careamist.predict(
                        source=image[t],
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        batch_size=batch_size,
                        tta=use_tta,
                    )
                    
                    # Squeeze and store result
                    result[t] = np.squeeze(prediction)
                    print("✓")
                
                print(f"✓ Denoising complete. Output: {result.shape}\n")
                return result

            else:
                print(f"✗ Error: Unsupported image dimensionality: {ndim}D")
                print("Supported formats: YX (2D), ZYX (3D), TYX (2D+time), TZYX (3D+time)")
                return image

        except (RuntimeError, ValueError, ImportError) as e:
            import traceback

            print(
                f"Error during CAREamics denoising in current environment: {str(e)}"
            )
            traceback.print_exc()

            # If we haven't already tried using the dedicated environment, try that as a fallback
            if not force_dedicated_env:
                print(
                    "Attempting fallback to dedicated CAREamics environment..."
                )
                args = {
                    "image": image,
                    "checkpoint_path": checkpoint_path,
                    "tile_size_z": tile_size_z,
                    "tile_size_y": tile_size_y,
                    "tile_size_x": tile_size_x,
                    "tile_overlap_z": tile_overlap_z,
                    "tile_overlap_y": tile_overlap_y,
                    "tile_overlap_x": tile_overlap_x,
                    "batch_size": batch_size,
                    "use_tta": use_tta,
                }

                if not is_env_created():
                    create_careamics_env()

                return run_careamics_in_env("predict", args)

            return None
