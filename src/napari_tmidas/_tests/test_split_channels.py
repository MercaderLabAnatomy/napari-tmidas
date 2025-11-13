"""Test for split_channels function with various image formats"""

import os
import sys

import numpy as np
import pytest

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from napari_tmidas.processing_functions.basic import split_channels


class TestSplitChannels:
    """Test the split_channels function with various input formats"""

    def test_split_tcyx_python_format(self):
        """Test splitting TCYX image (Time, Channel, Y, X) with python format"""
        # Create a TCYX image: 5 timepoints, 3 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 3, 100, 100)

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Result should be (3, 5, 100, 100): 3 channels, each with shape (5, 100, 100)
        assert result.shape == (
            3,
            5,
            100,
            100,
        ), f"Expected shape (3, 5, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_tcyx_fiji_format(self):
        """Test splitting TCYX image with Fiji format"""
        # Create a TCYX image: 5 timepoints, 3 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 3, 100, 100)

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="fiji"
        )

        # Result should be (3, 5, 100, 100): 3 channels, each with shape (5, 100, 100)
        assert result.shape == (
            3,
            5,
            100,
            100,
        ), f"Expected shape (3, 5, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_yxc_image(self):
        """Test splitting standard RGB image (YXC)"""
        # Create a YXC image: 100x100 pixels, 3 channels
        yxc_image = np.random.rand(100, 100, 3)

        result = split_channels(
            yxc_image, num_channels=3, time_steps=0, output_format="python"
        )

        # Result should be (3, 100, 100): 3 channels, each with shape (100, 100)
        assert result.shape == (
            3,
            100,
            100,
        ), f"Expected shape (3, 100, 100), got {result.shape}"

        # Each channel should have shape (100, 100)
        for i in range(3):
            assert result[i].shape == (
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_zyxc_image(self):
        """Test splitting 3D color image (ZYXC)"""
        # Create a ZYXC image: 10 z-slices, 100x100 pixels, 3 channels
        zyxc_image = np.random.rand(10, 100, 100, 3)

        result = split_channels(
            zyxc_image, num_channels=3, time_steps=0, output_format="python"
        )

        # Result should be (3, 10, 100, 100): 3 channels, each with shape (10, 100, 100)
        assert result.shape == (
            3,
            10,
            100,
            100,
        ), f"Expected shape (3, 10, 100, 100), got {result.shape}"

        # Each channel should have shape (10, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                10,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_tzyxc_image(self):
        """Test splitting 4D time-series color Z-stack (TZYXC)"""
        # Create a TZYXC image: 5 timepoints, 10 z-slices, 100x100 pixels, 3 channels
        tzyxc_image = np.random.rand(5, 10, 100, 100, 3)

        result = split_channels(
            tzyxc_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Result should be (3, 5, 10, 100, 100): 3 channels, each with shape (5, 10, 100, 100)
        assert result.shape == (
            3,
            5,
            10,
            100,
            100,
        ), f"Expected shape (3, 5, 10, 100, 100), got {result.shape}"

        # Each channel should have shape (5, 10, 100, 100)
        for i in range(3):
            assert result[i].shape == (
                5,
                10,
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_channels_with_4_channels(self):
        """Test splitting image with 4 channels (RGBA)"""
        # Create a YXC image: 100x100 pixels, 4 channels
        yxc_image = np.random.rand(100, 100, 4)

        result = split_channels(
            yxc_image, num_channels=4, time_steps=0, output_format="python"
        )

        # Result should be (4, 100, 100): 4 channels, each with shape (100, 100)
        assert result.shape == (
            4,
            100,
            100,
        ), f"Expected shape (4, 100, 100), got {result.shape}"

        # Each channel should have shape (100, 100)
        for i in range(4):
            assert result[i].shape == (
                100,
                100,
            ), f"Channel {i} has incorrect shape {result[i].shape}"

    def test_split_channels_verifies_data_integrity(self):
        """Test that split channels contain the correct data"""
        # Create a simple test image where we can verify the data
        tcyx_image = np.zeros((2, 3, 10, 10))  # 2 timepoints, 3 channels

        # Set distinct values for each channel
        tcyx_image[:, 0, :, :] = 1.0  # Channel 0
        tcyx_image[:, 1, :, :] = 2.0  # Channel 1
        tcyx_image[:, 2, :, :] = 3.0  # Channel 2

        result = split_channels(
            tcyx_image, num_channels=3, time_steps=2, output_format="python"
        )

        # Verify shape
        assert result.shape == (3, 2, 10, 10)

        # Verify data integrity
        assert np.allclose(result[0], 1.0), "Channel 0 data incorrect"
        assert np.allclose(result[1], 2.0), "Channel 1 data incorrect"
        assert np.allclose(result[2], 3.0), "Channel 2 data incorrect"

    def test_split_channels_auto_detect_mismatch(self):
        """Test that function handles mismatch between specified and actual channel count"""
        # Create a TCYX image: 5 timepoints, 4 channels, 100x100 pixels
        tcyx_image = np.random.rand(5, 4, 100, 100)

        # Specify 3 channels when there are actually 4
        result = split_channels(
            tcyx_image, num_channels=3, time_steps=5, output_format="python"
        )

        # Should auto-detect and use 4 channels
        assert result.shape == (
            4,
            5,
            100,
            100,
        ), f"Expected shape (4, 5, 100, 100), got {result.shape}"

    def test_split_channels_dimension_error(self):
        """Test that function raises error for invalid input"""
        # Create a 2D image (should fail)
        image_2d = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="at least 3 dimensions"):
            split_channels(image_2d, num_channels=3, time_steps=0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
