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

            # Prepare frame for JP2 encoding - OpenCV's JP2 encoder requires uint8 or uint16
            if frame.dtype not in [np.uint8, np.uint16]:
                # Get actual range
                min_val, max_val = np.min(frame), np.max(frame)

                # Special handling for different bit depths
                if np.issubdtype(frame.dtype, np.floating):
                    # For floating point, scale to full uint16 range for better precision
                    if min_val < max_val:
                        frame = (
                            (frame - min_val) * 65535 / (max_val - min_val)
                        ).astype(np.uint16)
                    else:
                        frame = np.full_like(frame, 32768, dtype=np.uint16)
                else:
                    # For other integer types, also convert to uint16
                    if min_val < max_val:
                        frame = (
                            (frame - min_val) * 65535 / (max_val - min_val)
                        ).astype(np.uint16)
                    else:
                        frame = np.full_like(frame, 32768, dtype=np.uint16)

                # Report min/max after conversion for debugging
                if i == 0 or i == num_frames - 1:
                    print(
                        f"After conversion - min: {np.min(frame)}, max: {np.max(frame)}, "
                        f"mean: {np.mean(frame):.2f}, dtype: {frame.dtype}"
                    )

            # Convert grayscale to RGB if needed for compatibility
            if is_grayscale and len(frame.shape) == 2:
                if frame.dtype == np.uint16:
                    # For uint16, we need to maintain the bit depth during color conversion
                    # OpenCV doesn't have a direct grayscale to RGB for uint16, so we'll do it manually
                    h, w = frame.shape
                    rgb_frame = np.stack((frame, frame, frame), axis=2)
                else:
                    # For uint8, we can use cv2.cvtColor
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = frame

            # Save frame as intermediate PNG (OpenCV's JP2 encoder can be inconsistent)
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


# # Example usage
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description='Convert TIF stack to MP4 via JPEG2000 lossless intermediate')
#     parser.add_argument('input_path', help='Path to input TIF file')
#     parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video')
#     parser.add_argument('--keep-temp', action='store_true', help='Keep temporary JP2 files')

#     args = parser.parse_args()

#     # Convert to MP4
#     output_path = tif_to_mp4_via_jp2(
#         args.input_path,
#         fps=args.fps,
#         cleanup_temp=not args.keep_temp
#     )
#     print(f"Video saved to: {output_path}")
