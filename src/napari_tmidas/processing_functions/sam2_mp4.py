import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile


def tif_to_mp4(input_path, fps=10, cleanup_temp=True):
    """
    Convert a TIF stack to MP4 using JPEG2000 lossless as an intermediate format.

    Parameters:
    -----------
    input_path : str or Path
        Path to the input TIF file

    fps : int, optional
        Frames per second for the video. Default is 10.

    cleanup_temp : bool, optional
        Whether to clean up temporary JP2 files. Default is True.

    Returns:
    --------
    str
        Path to the created MP4 file
    """
    input_path = Path(input_path)

    # Generate output MP4 path in the same folder
    output_path = input_path.with_suffix(".mp4")

    # Create a temporary directory for JP2 files
    temp_dir = Path(tempfile.mkdtemp(prefix="tif_to_jp2_"))

    try:
        # Read the TIFF file
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
                            frames = stack.reshape(
                                t_dim * z_dim, height, width
                            )
                            is_grayscale = True
                    else:
                        raise ValueError(
                            f"Unsupported TIFF shape: {stack.shape}"
                        )
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
                        raise ValueError(
                            f"Unsupported frame shape: {frame.shape}"
                        )

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

        # Get the number of frames
        num_frames = len(frames)
        print(f"Processing {num_frames} frames...")

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required but was not found.")

        # Process each frame and save as lossless JP2
        jp2_paths = []

        for i in range(num_frames):
            # Get the frame
            frame = frames[i].copy()

            # For analysis and debugging
            if i == 0 or i == num_frames - 1:
                print(f"Frame {i} shape: {frame.shape}, dtype: {frame.dtype}")
                print(
                    f"Frame {i} stats - min: {np.min(frame)}, max: {np.max(frame)}, mean: {np.mean(frame):.2f}"
                )

            # Improved handling for float32 and other types - prioritize conversion to uint8
            if frame.dtype != np.uint8:
                # Get actual data range
                min_val, max_val = np.min(frame), np.max(frame)

                # For float32 and other types, convert directly to uint8
                if (
                    np.issubdtype(frame.dtype, np.floating)
                    or min_val < max_val
                ):
                    # Scale to full uint8 range [0, 255] with proper handling of min/max
                    frame = np.clip(
                        (frame - min_val)
                        * 255.0
                        / (max_val - min_val + 1e-10),
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    # If min equals max (constant image), create a mid-gray image
                    frame = np.full_like(frame, 128, dtype=np.uint8)

                # Report conversion stats for debugging
                if i == 0 or i == num_frames - 1:
                    print(
                        f"After conversion - min: {np.min(frame)}, max: {np.max(frame)}, "
                        f"mean: {np.mean(frame):.2f}, dtype: {frame.dtype}"
                    )

            # Convert grayscale to RGB if needed for compatibility
            if is_grayscale and len(frame.shape) == 2:
                # For uint8, we can use cv2.cvtColor
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = frame

            # Save frame as intermediate PNG
            png_path = temp_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(png_path), rgb_frame)

            # Use FFmpeg to convert PNG to lossless JPEG2000
            jp2_path = temp_dir / f"frame_{i:06d}.jp2"
            jp2_paths.append(jp2_path)

            # FFmpeg command for lossless JP2 conversion
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(png_path),
                "-codec",
                "jpeg2000",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # width and height are required to be even numbers
                "-pix_fmt",
                (
                    "rgb24"
                    if not is_grayscale or len(rgb_frame.shape) == 3
                    else "gray"
                ),
                "-compression_level",
                "0",  # Lossless setting
                str(jp2_path),
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"FFmpeg JP2 encoding error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
                )
                # Fallback to PNG if JP2 encoding fails
                print(f"Falling back to PNG for frame {i}")
                jp2_paths[-1] = png_path

            # Delete the PNG file if JP2 was successful and not the same as fallback
            if jp2_paths[-1] != png_path and png_path.exists():
                png_path.unlink()

            # Report progress
            if (i + 1) % 50 == 0 or i == 0 or i == num_frames - 1:
                print(f"Processed {i+1}/{num_frames} frames")

        # Use FFmpeg to create MP4 from JP2/PNG frames
        print(f"Creating MP4 file from {len(jp2_paths)} frames...")

        # Get the extension of the first frame to determine input pattern
        ext = jp2_paths[0].suffix

        cmd = [
            "ffmpeg",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / f"frame_%06d{ext}"),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            "17",  # High quality
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

    return str(output_path)
