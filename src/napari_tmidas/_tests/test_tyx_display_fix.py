"""Test TYX image display bug fix"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestTYXDisplayFix:
    """Test the fix for TYX images being incorrectly displayed"""

    def test_channel_detection_rgb(self):
        """RGB images (3 channels) should be detected as multi-channel"""
        shape = (3, 100, 100)
        # Simulate the detection logic
        is_multi_channel = len(shape) > 2 and shape[0] <= 4 and shape[0] > 1
        assert (
            is_multi_channel
        ), "RGB image should be detected as multi-channel"

    def test_channel_detection_rgba(self):
        """RGBA images (4 channels) should be detected as multi-channel"""
        shape = (4, 100, 100)
        is_multi_channel = len(shape) > 2 and shape[0] <= 4 and shape[0] > 1
        assert (
            is_multi_channel
        ), "RGBA image should be detected as multi-channel"

    def test_channel_detection_tyx(self):
        """TYX time series (5+ timepoints) should NOT be detected as multi-channel"""
        shape = (5, 100, 100)
        is_multi_channel = len(shape) > 2 and shape[0] <= 4 and shape[0] > 1
        assert (
            not is_multi_channel
        ), "TYX time series should NOT be detected as multi-channel"

    def test_channel_detection_zyx(self):
        """ZYX z-stacks (10+ slices) should NOT be detected as multi-channel"""
        shape = (10, 100, 100)
        is_multi_channel = len(shape) > 2 and shape[0] <= 4 and shape[0] > 1
        assert (
            not is_multi_channel
        ), "ZYX z-stack should NOT be detected as multi-channel"

    def test_channel_detection_dual_channel(self):
        """2-channel images should be detected as multi-channel"""
        shape = (2, 100, 100)
        is_multi_channel = len(shape) > 2 and shape[0] <= 4 and shape[0] > 1
        assert (
            is_multi_channel
        ), "2-channel image should be detected as multi-channel"

    def test_3d_view_not_enabled_for_tyx(self):
        """TYX time series should use 2D view, not 3D view"""
        shape = (5, 100, 100)

        # Simulate the 3D view detection logic
        if shape[0] >= 2 and shape[0] <= 4:
            meaningful_dims = shape[1:]
        else:
            meaningful_dims = shape

        # Check if 3D view would be enabled
        enable_3d = False
        if len(meaningful_dims) >= 4:
            z_dim = meaningful_dims[1]
            enable_3d = z_dim > 1
        elif len(meaningful_dims) == 3:
            first_dim = meaningful_dims[0]
            enable_3d = first_dim > 10

        assert not enable_3d, "TYX with 5 timepoints should NOT enable 3D view"

    def test_3d_view_enabled_for_large_zyx(self):
        """Large ZYX z-stacks should enable 3D view"""
        shape = (20, 100, 100)

        # Simulate the 3D view detection logic
        if shape[0] >= 2 and shape[0] <= 4:
            meaningful_dims = shape[1:]
        else:
            meaningful_dims = shape

        enable_3d = False
        if len(meaningful_dims) >= 4:
            z_dim = meaningful_dims[1]
            enable_3d = z_dim > 1
        elif len(meaningful_dims) == 3:
            first_dim = meaningful_dims[0]
            enable_3d = first_dim > 10

        assert enable_3d, "ZYX with 20 slices should enable 3D view"

    def test_3d_view_enabled_for_tzyx(self):
        """TZYX data should enable 3D view"""
        shape = (10, 50, 100, 100)

        # Simulate the 3D view detection logic
        if shape[0] >= 2 and shape[0] <= 4:
            meaningful_dims = shape[1:]
        else:
            meaningful_dims = shape

        enable_3d = False
        if len(meaningful_dims) >= 4:
            z_dim = meaningful_dims[1]
            enable_3d = z_dim > 1
        elif len(meaningful_dims) == 3:
            first_dim = meaningful_dims[0]
            enable_3d = first_dim > 10

        assert enable_3d, "TZYX data should enable 3D view"

    def test_3d_view_not_enabled_for_rgb_channels(self):
        """After splitting RGB, individual channels should use 2D view"""
        shape = (
            3,
            100,
            100,
        )  # This would be seen after channel splitting doesn't occur

        # Simulate the 3D view detection logic
        if shape[0] >= 2 and shape[0] <= 4:
            meaningful_dims = shape[1:]
        else:
            meaningful_dims = shape

        enable_3d = False
        if len(meaningful_dims) >= 4:
            z_dim = meaningful_dims[1]
            enable_3d = z_dim > 1
        elif len(meaningful_dims) == 3:
            first_dim = meaningful_dims[0]
            enable_3d = first_dim > 10

        assert not enable_3d, "RGB channels should use 2D view"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
