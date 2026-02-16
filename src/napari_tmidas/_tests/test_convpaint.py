# src/napari_tmidas/_tests/test_convpaint.py
import numpy as np
import pytest

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import discover_and_load_processing_functions


class TestConvpaintPrediction:
    """Test convpaint prediction functionality."""

    def test_convpaint_function_registered(self):
        """Test that convpaint function is registered."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()
        
        functions = BatchProcessingRegistry.list_functions()
        assert "Convpaint Prediction" in functions

    def test_convpaint_parameters(self):
        """Test convpaint function has correct parameters."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()
        
        func_info = BatchProcessingRegistry.get_function_info(
            "Convpaint Prediction"
        )
        assert func_info is not None

        params = func_info["parameters"]
        assert "model_path" in params
        assert "image_downsample" in params
        assert "output_type" in params
        assert "background_label" in params
        assert "use_cpu" in params
        assert "force_dedicated_env" in params
        assert "z_batch_size" in params
        # Check parameter defaults
        assert params["image_downsample"]["default"] == 2
        assert params["output_type"]["default"] == "semantic"
        assert params["background_label"]["default"] == 1
        assert params["use_cpu"]["default"] is False
        assert params["z_batch_size"]["default"] == 20

    def test_convpaint_output_type_options(self):
        """Test output_type has correct options."""
        # Ensure processing functions are loaded
        discover_and_load_processing_functions()
        
        func_info = BatchProcessingRegistry.get_function_info(
            "Convpaint Prediction"
        )
        params = func_info["parameters"]
        assert "options" in params["output_type"]
        assert params["output_type"]["options"] == ["semantic", "instance"]

    def test_convpaint_missing_model_path(self):
        """Test that missing model_path raises ValueError."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            convpaint_predict,
        )

        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="model_path"):
            convpaint_predict(image, model_path="")

    def test_convpaint_invalid_model_path(self):
        """Test that invalid model_path raises ValueError."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            convpaint_predict,
        )

        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="not found"):
            convpaint_predict(image, model_path="/nonexistent/model.pkl")

    def test_convpaint_invalid_output_type(self):
        """Test that invalid output_type raises ValueError."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            convpaint_predict,
        )

        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Create a temporary model file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            with pytest.raises(ValueError, match="output_type"):
                convpaint_predict(
                    image, model_path=model_path, output_type="invalid"
                )
        finally:
            import os

            os.unlink(model_path)

    def test_convpaint_invalid_downsample(self):
        """Test that invalid image_downsample raises ValueError."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            convpaint_predict,
        )

        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Create a temporary model file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            with pytest.raises(ValueError, match="image_downsample"):
                convpaint_predict(
                    image, model_path=model_path, image_downsample=0
                )
        finally:
            import os

            os.unlink(model_path)

    def test_semantic_to_instance_conversion_2d(self):
        """Test semantic to instance conversion for 2D images."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _convert_semantic_to_instance,
        )

        # Create a simple 2D semantic mask with two classes
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:20, 10:20] = 1  # Class 1 object
        image[30:40, 30:40] = 2  # Class 2 object

        result = _convert_semantic_to_instance(image)

        # Should have 2 unique labels plus background
        unique_labels = np.unique(result)
        assert len(unique_labels) == 3  # 0 (background), 1, 2
        assert 0 in unique_labels

    def test_semantic_to_instance_conversion_3d(self):
        """Test semantic to instance conversion for 3D images."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _convert_semantic_to_instance,
        )

        # Create a simple 3D semantic mask (small Z stack)
        image = np.zeros((5, 50, 50), dtype=np.uint8)
        image[1:3, 10:20, 10:20] = 1  # 3D object class 1
        image[2:4, 30:40, 30:40] = 2  # 3D object class 2

        result = _convert_semantic_to_instance(image)

        # Should process as 3D volume
        assert result.shape == image.shape
        unique_labels = np.unique(result)
        assert len(unique_labels) >= 2  # At least background and 1+ objects

    def test_background_label_removal(self):
        """Test that background label is correctly removed."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _convert_semantic_to_instance,
        )

        # Create semantic mask with specific background label
        image = np.ones((50, 50), dtype=np.uint8)  # Background = 1
        image[10:20, 10:20] = 2  # Foreground = 2
        image[30:40, 30:40] = 3  # Foreground = 3

        # Simulate background removal (this happens in main function)
        image[image == 1] = 0

        result = _convert_semantic_to_instance(image)

        # Background should be 0
        assert result[0, 0] == 0
        assert result[49, 49] == 0
        # Objects should have non-zero labels
        assert result[15, 15] > 0
        assert result[35, 35] > 0


class TestConvpaintEnvManager:
    """Test convpaint environment manager."""

    def test_env_manager_exists(self):
        """Test that environment manager module exists."""
        from napari_tmidas.processing_functions import convpaint_env_manager

        assert convpaint_env_manager is not None

    def test_env_manager_functions(self):
        """Test that required functions exist."""
        from napari_tmidas.processing_functions.convpaint_env_manager import (
            create_convpaint_env,
            get_env_python_path,
            is_convpaint_installed,
            is_env_created,
        )

        # Just check they exist and are callable
        assert callable(is_convpaint_installed)
        assert callable(is_env_created)
        assert callable(get_env_python_path)
        assert callable(create_convpaint_env)

    def test_convpaint_not_installed_initially(self):
        """Test convpaint detection."""
        from napari_tmidas.processing_functions.convpaint_env_manager import (
            is_convpaint_installed,
        )

        # This will return True or False depending on environment
        result = is_convpaint_installed()
        assert isinstance(result, bool)


class TestConvpaintBatching:
    """Test convpaint Z-stack batching functionality for memory management."""

    def test_z_batch_size_parameter_validation(self):
        """Test z_batch_size parameter validation."""
        func_info = BatchProcessingRegistry.get_function_info(
            "Convpaint Prediction"
        )
        params = func_info["parameters"]
        
        assert "z_batch_size" in params
        assert params["z_batch_size"]["min"] == 1
        assert params["z_batch_size"]["max"] == 200
        assert params["z_batch_size"]["default"] == 20

    def test_process_zyx_in_batches_function_exists(self):
        """Test that the batching function exists."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_zyx_in_batches,
        )
        
        assert callable(_process_zyx_in_batches)

    def test_batching_output_shape(self):
        """Test that batching preserves output shape."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_zyx_in_batches,
        )
        from unittest.mock import MagicMock, patch
        
        # Create a mock 3D image (50 Z-planes)
        image = np.random.randint(0, 255, (50, 100, 100), dtype=np.uint8)
        
        # Mock the segmentation functions to return simple labels
        def mock_segment(img, *args, **kwargs):
            return np.ones(img.shape, dtype=np.uint32)
        
        with patch(
            'napari_tmidas.processing_functions.convpaint_prediction.run_convpaint_in_env',
            side_effect=mock_segment
        ):
            result = _process_zyx_in_batches(
                image,
                model_path="dummy.pkl",
                image_downsample=2,
                use_dedicated=True,
                use_cpu=False,
                z_batch_size=10
            )
        
        # Output shape should match input shape
        assert result.shape == image.shape
        assert result.dtype == np.uint32

    def test_batching_correct_number_of_batches(self):
        """Test that correct number of batches are created."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_zyx_in_batches,
        )
        from unittest.mock import MagicMock, patch
        
        # Create image with 98 Z-planes (like user's case)
        image = np.random.randint(0, 255, (98, 100, 100), dtype=np.uint8)
        
        call_count = 0
        def mock_segment(img, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return np.ones(img.shape, dtype=np.uint32)
        
        with patch(
            'napari_tmidas.processing_functions.convpaint_prediction.run_convpaint_in_env',
            side_effect=mock_segment
        ):
            result = _process_zyx_in_batches(
                image,
                model_path="dummy.pkl",
                image_downsample=2,
                use_dedicated=True,
                use_cpu=False,
                z_batch_size=20
            )
        
        # 98 planes / 20 batch_size = 5 batches (ceil)
        expected_batches = 5
        assert call_count == expected_batches

    def test_batching_with_different_batch_sizes(self):
        """Test batching with various batch sizes."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_zyx_in_batches,
        )
        from unittest.mock import patch
        
        image = np.random.randint(0, 255, (50, 64, 64), dtype=np.uint8)
        
        def mock_segment(img, *args, **kwargs):
            return np.ones(img.shape, dtype=np.uint32)
        
        # Test different batch sizes
        for batch_size in [5, 10, 25]:
            with patch(
                'napari_tmidas.processing_functions.convpaint_prediction.run_convpaint_in_env',
                side_effect=mock_segment
            ):
                result = _process_zyx_in_batches(
                    image,
                    model_path="dummy.pkl",
                    image_downsample=2,
                    use_dedicated=True,
                    use_cpu=False,
                    z_batch_size=batch_size
                )
            
            assert result.shape == image.shape
            assert result.dtype == np.uint32

    def test_time_series_uses_batching_for_large_zstacks(self):
        """Test that TZYX data with large Z-stacks triggers batching."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_time_series,
        )
        from unittest.mock import patch
        
        # Create TZYX data (2 timepoints, 50 Z-planes each)
        image = np.random.randint(0, 255, (2, 50, 100, 100), dtype=np.uint8)
        
        batch_function_called = False
        def mock_batch_processing(*args, **kwargs):
            nonlocal batch_function_called
            batch_function_called = True
            z_size = args[0].shape[0]
            return np.ones((z_size, 100, 100), dtype=np.uint32)
        
        with patch(
            'napari_tmidas.processing_functions.convpaint_prediction._process_zyx_in_batches',
            side_effect=mock_batch_processing
        ):
            result = _process_time_series(
                image,
                model_path="dummy.pkl",
                image_downsample=2,
                use_dedicated=True,
                use_cpu=False,
                is_3d=True,
                z_batch_size=20  # Trigger batching for 50 > 20
            )
        
        # Batching should have been called for large Z-stacks
        assert batch_function_called
        assert result.shape == image.shape

    def test_small_zstack_skips_batching(self):
        """Test that small Z-stacks don't trigger batching."""
        from napari_tmidas.processing_functions.convpaint_prediction import (
            _process_time_series,
        )
        from unittest.mock import patch
        
        # Create TZYX data with small Z-stack (15 planes < 20 batch_size)
        image = np.random.randint(0, 255, (2, 15, 100, 100), dtype=np.uint8)
        
        batch_function_called = False
        def mock_batch_processing(*args, **kwargs):
            nonlocal batch_function_called
            batch_function_called = True
            return np.ones(args[0].shape, dtype=np.uint32)
        
        def mock_regular_processing(img, *args, **kwargs):
            return np.ones(img.shape, dtype=np.uint32)
        
        with patch(
            'napari_tmidas.processing_functions.convpaint_prediction._process_zyx_in_batches',
            side_effect=mock_batch_processing
        ), patch(
            'napari_tmidas.processing_functions.convpaint_prediction.run_convpaint_in_env',
            side_effect=mock_regular_processing
        ):
            result = _process_time_series(
                image,
                model_path="dummy.pkl",
                image_downsample=2,
                use_dedicated=True,
                use_cpu=False,
                is_3d=True,
                z_batch_size=20
            )
        
        # Batching should NOT be called for small Z-stacks
        assert not batch_function_called
        assert result.shape == image.shape
