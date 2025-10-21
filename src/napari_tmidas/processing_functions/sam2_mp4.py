import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile


def tif_to_mp4(
    input_path, fps=10, cleanup_temp=True, use_ffmpeg=False, crf=17
):
    """
    Convert a TIF stack to MP4 with optimized performance.

    Parameters:
    -----------
    input_path : str or Path
        Path to the input TIF file

    fps : int, optional
        Frames per second for the video. Default is 10.

    cleanup_temp : bool, optional
        Whether to clean up temporary files (only used if use_ffmpeg=True). Default is True.

    use_ffmpeg : bool, optional
        Whether to use FFmpeg for encoding (slower but higher quality).
        If False (default), uses OpenCV VideoWriter which is much faster.

    crf : int, optional
        Constant Rate Factor for quality (0-51, lower is better). Default is 17.
        Only used when use_ffmpeg=True.

    Returns:
    --------
    str
        Path to the created MP4 file
    """
    input_path = Path(input_path)

    # Generate output MP4 path in the same folder
    output_path = input_path.with_suffix(".mp4")

    # Use fast OpenCV-based encoding by default
    if not use_ffmpeg:
        return _tif_to_mp4_opencv(input_path, output_path, fps)

    # Otherwise use FFmpeg-based encoding (slower but potentially higher quality)
    return _tif_to_mp4_ffmpeg(input_path, output_path, fps, crf, cleanup_temp)


def _load_tiff_stack(input_path):
    """
    Load TIFF stack and normalize to uint8 format.

    Returns:
    --------
    tuple
        (frames, is_grayscale) where frames is a numpy array and is_grayscale is a bool
    """
    print(f"Reading {input_path}...")

    try:
        # Try using tifffile which handles scientific imaging formats better
        with tifffile.TiffFile(input_path) as tif:
            # Check if it's a multi-page TIFF (Z stack or time series)
            if len(tif.pages) > 1:
                # Read as a stack - this will handle TYX or ZYX format
                stack = tifffile.imread(input_path)
                print(f"Stack shape: {stack.shape}, dtype: {stack.dtype}")

                # Check dimensions
                if len(stack.shape) == 3:
                    # We have a 3D stack (T/Z, Y, X)
                    print(f"Detected 3D stack with shape {stack.shape}")
                    frames = stack
                    is_grayscale = True
                elif len(stack.shape) == 4:
                    if stack.shape[3] == 3:  # (T/Z, Y, X, 3) - color
                        print(
                            f"Detected 4D color stack with shape {stack.shape}"
                        )
                        frames = stack
                        is_grayscale = False
                    else:
                        # We have a 4D stack (likely T, Z, Y, X)
                        print(
                            f"Detected 4D stack with shape {stack.shape}. Flattening first two dimensions."
                        )
                        # Flatten first two dimensions
                        t_dim, z_dim = stack.shape[0], stack.shape[1]
                        height, width = stack.shape[2], stack.shape[3]
                        frames = stack.reshape(t_dim * z_dim, height, width)
                        is_grayscale = True
                else:
                    raise ValueError(f"Unsupported TIFF shape: {stack.shape}")
            else:
                # Single page TIFF
                frame = tifffile.imread(input_path)
                print(f"Detected single frame with shape {frame.shape}")
                if len(frame.shape) == 2:  # (Y, X) - grayscale
                    frames = np.array([frame])
                    is_grayscale = True
                elif (
                    len(frame.shape) == 3 and frame.shape[2] == 3
                ):  # (Y, X, 3) - color
                    frames = np.array([frame])
                    is_grayscale = False
                else:
                    raise ValueError(f"Unsupported frame shape: {frame.shape}")

            # Print min/max/mean values to help diagnose
            sample_frame = frames[0]
            print(
                f"Sample frame - min: {np.min(sample_frame)}, max: {np.max(sample_frame)}, "
                f"mean: {np.mean(sample_frame):.2f}, dtype: {sample_frame.dtype}"
            )

    except (
        OSError,
        tifffile.TiffFileError,
        ValueError,
        FileNotFoundError,
        MemoryError,
    ) as e:
        print(f"Error reading with tifffile: {e}")
        print("Falling back to OpenCV...")

        # Try with OpenCV as fallback
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(
                f"Could not open file {input_path} with either tifffile or OpenCV"
            ) from e

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        frames = np.array(frames)
        is_grayscale = len(frames[0].shape) == 2 or frames[0].shape[2] == 1
        cap.release()

    return frames, is_grayscale


def _normalize_frame_to_uint8(frame):
    """
    Normalize a frame to uint8 format.

    Parameters:
    -----------
    frame : numpy.ndarray
        Input frame to normalize

    Returns:
    --------
    numpy.ndarray
        Normalized uint8 frame
    """
    if frame.dtype == np.uint8:
        return frame

    # Get actual data range
    min_val, max_val = np.min(frame), np.max(frame)

    # For float32 and other types, convert directly to uint8
    if np.issubdtype(frame.dtype, np.floating) or min_val < max_val:
        # Scale to full uint8 range [0, 255] with proper handling of min/max
        frame = np.clip(
            (frame - min_val) * 255.0 / (max_val - min_val + 1e-10),
            0,
            255,
        ).astype(np.uint8)
    else:
        # If min equals max (constant image), create a mid-gray image
        frame = np.full_like(frame, 128, dtype=np.uint8)

    return frame


def _tif_to_mp4_opencv(input_path, output_path, fps=10):
    """
    Fast MP4 conversion using OpenCV VideoWriter.
    This is much faster than the FFmpeg approach as it writes directly to MP4.

    This method is 10-50x faster than the FFmpeg approach because:
    1. No intermediate files (PNG/JP2) are created
    2. Frames are written directly to the video stream
    3. Uses hardware acceleration when available
    """
    # Load the TIFF stack
    frames, is_grayscale = _load_tiff_stack(input_path)
    num_frames = len(frames)
    print(f"Processing {num_frames} frames with OpenCV VideoWriter...")

    # Get frame dimensions
    first_frame = _normalize_frame_to_uint8(frames[0].copy())

    # Convert grayscale to BGR for OpenCV
    if is_grayscale and len(first_frame.shape) == 2:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)

    height, width = first_frame.shape[:2]

    # Ensure dimensions are even (required for some codecs)
    if height % 2 != 0:
        height += 1
    if width % 2 != 0:
        width += 1

    # Define the codec and create VideoWriter object
    # Try different codecs in order of preference
    codecs = [
        ("mp4v", ".mp4"),  # MPEG-4, widely compatible
        ("avc1", ".mp4"),  # H.264, better quality
    ]

    writer = None
    for codec, _ext in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        if writer.isOpened():
            print(f"Using codec: {codec}")
            break
        else:
            writer.release()
            writer = None

    if writer is None:
        raise RuntimeError(
            "Could not initialize VideoWriter with any supported codec"
        )

    try:
        # Process and write frames
        for i in range(num_frames):
            # Get and normalize frame
            frame = frames[i].copy()
            frame = _normalize_frame_to_uint8(frame)

            # Convert grayscale to BGR if needed
            if is_grayscale and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Resize if dimensions were adjusted
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            # Write frame
            writer.write(frame)

            # Report progress
            if (i + 1) % 50 == 0 or i == 0 or i == num_frames - 1:
                print(f"Processed {i+1}/{num_frames} frames")

        print(f"Successfully created MP4: {output_path}")
        return str(output_path)

    finally:
        writer.release()


def _tif_to_mp4_ffmpeg(
    input_path, output_path, fps=10, crf=17, cleanup_temp=True
):
    """
    MP4 conversion using FFmpeg with high quality settings.
    This method is slower but provides more control over encoding parameters.
    """
    # Create a temporary directory for frame files
    temp_dir = Path(tempfile.mkdtemp(prefix="tif_to_mp4_"))

    try:
        # Load the TIFF stack
        frames, is_grayscale = _load_tiff_stack(input_path)
        num_frames = len(frames)
        print(f"Processing {num_frames} frames with FFmpeg...")

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required but was not found.")

        # Process each frame and save as PNG (simpler and faster than JP2)
        for i in range(num_frames):
            # Get and normalize frame
            frame = frames[i].copy()
            frame = _normalize_frame_to_uint8(frame)

            # Convert grayscale to RGB if needed for compatibility
            if is_grayscale and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Save frame as PNG
            png_path = temp_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(png_path), frame)

            # Report progress
            if (i + 1) % 50 == 0 or i == 0 or i == num_frames - 1:
                print(f"Saved {i+1}/{num_frames} frames")

        # Use FFmpeg to create MP4 from PNG frames
        print(f"Creating MP4 file from {num_frames} frames...")

        cmd = [
            "ffmpeg",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",  # Compatible colorspace
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Successfully created MP4: {output_path}")
        except subprocess.CalledProcessError as e:
            print(
                f"FFmpeg MP4 creation error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
            )
            raise

        return str(output_path)

    finally:
        # Clean up temporary directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)
        else:
            print(f"Temporary files saved in: {temp_dir}")
