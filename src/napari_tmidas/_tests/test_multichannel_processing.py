"""Test multichannel processing with channel selection"""
import numpy as np
import pytest
import tifffile

from napari_tmidas._file_selector import (
    detect_channels_for_file,
    detect_channels_in_image,
)


class TestChannelDetection:
    """Test channel detection in various image formats"""

    def test_detect_single_channel_2d(self):
        """Test detection in single channel 2D image"""
        image = np.random.rand(100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 1
        assert channel_axis is None

    def test_detect_cyx_format(self):
        """Test detection in CYX format (3 channels)"""
        image = np.random.rand(3, 100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 3
        assert channel_axis == 0

    def test_detect_czyx_format(self):
        """Test detection in CZYX format (2 channels)"""
        image = np.random.rand(2, 10, 100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 2
        assert channel_axis == 0

    def test_detect_tcyx_format(self):
        """Test detection in TCYX format"""
        image = np.random.rand(10, 3, 100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 3
        assert channel_axis == 1

    def test_detect_tczyx_format(self):
        """Test detection in TCZYX format"""
        image = np.random.rand(10, 2, 5, 100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 2
        assert channel_axis == 1

    def test_detect_tzcyx_format(self):
        """Test detection in TZCYX format from ImageJ-generated TIFF stacks."""
        image = np.random.rand(47, 25, 2, 1024, 1024)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 2
        assert channel_axis == 2

    def test_detect_tiff_axes_metadata_tzcyx(self, tmp_path):
        """Prefer TIFF axes metadata over shape heuristics when available."""
        image = np.random.randint(
            0, 255, size=(47, 25, 2, 32, 32), dtype=np.uint8
        )
        path = tmp_path / "imagej_tzcyx.tif"
        tifffile.imwrite(
            path,
            image,
            imagej=True,
            metadata={"axes": "TZCYX"},
        )

        num_channels, channel_axis = detect_channels_for_file(str(path))
        assert num_channels == 2
        assert channel_axis == 2

    def test_no_channel_dimension_3d(self):
        """Test 3D image without channel dimension (ZYX)"""
        image = np.random.rand(50, 100, 100)
        num_channels, channel_axis = detect_channels_in_image(image)
        # Should not detect channels (50 is too large for channel count)
        assert num_channels == 1
        assert channel_axis is None

    def test_detect_rgb_image(self):
        """Test detection in RGB image (3 channels)"""
        image = np.random.rand(3, 512, 512)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 3
        assert channel_axis == 0

    def test_detect_four_channel(self):
        """Test detection in 4-channel image (RGBIR)"""
        image = np.random.rand(4, 256, 256)
        num_channels, channel_axis = detect_channels_in_image(image)
        assert num_channels == 4
        assert channel_axis == 0

    def test_detect_multi_layer_data(self):
        """Test detection with multi-layer napari data"""
        # Simulate napari-ome-zarr layer data format
        image_data = np.random.rand(3, 100, 100)
        layer_data = [
            (image_data, {}, "image"),
            (np.random.rand(100, 100), {}, "labels"),
        ]
        num_channels, channel_axis = detect_channels_in_image(layer_data)
        # Should extract first image layer and detect channels
        assert num_channels == 3
        assert channel_axis == 0


class TestChannelExtraction:
    """Test channel extraction in processing"""

    def test_extract_single_channel(self):
        """Test extracting a single channel from CYX image"""
        image = np.random.rand(3, 100, 100)
        # Extract channel 1
        channel_1 = np.take(image, 1, axis=0)
        assert channel_1.shape == (100, 100)

    def test_extract_all_channels(self):
        """Test extracting all channels separately"""
        image = np.random.rand(4, 100, 100)
        channels = []
        for i in range(4):
            channel = np.take(image, i, axis=0)
            channels.append(channel)
            assert channel.shape == (100, 100)
        assert len(channels) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
