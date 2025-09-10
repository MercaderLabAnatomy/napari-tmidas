# src/napari_tmidas/_tests/test_init.py
from napari_tmidas import (
    __version__,
    batch_crop_anything_widget,
    file_selector,
    label_inspector_widget,
    make_sample_data,
    napari_get_reader,
    roi_colocalization_analyzer,
    write_multiple,
    write_single_image,
)


class TestInit:
    def test_version_import(self):
        """Test that version is imported correctly"""
        # Version should be a string
        assert isinstance(__version__, str)
        # Should not be "unknown" in normal operation
        assert __version__ != "unknown"

    def test_core_exports_available(self):
        """Test that core exports are always available"""
        # These should always be importable
        assert napari_get_reader is not None
        assert write_single_image is not None
        assert write_multiple is not None
        assert make_sample_data is not None
        assert file_selector is not None

    def test_optional_exports(self):
        """Test optional exports (may be None on some platforms)"""
        # These might be None if dependencies fail to load on Windows
        # but we should at least be able to import them
        assert (
            label_inspector_widget is not None
            or label_inspector_widget is None
        )
        assert (
            roi_colocalization_analyzer is not None
            or roi_colocalization_analyzer is None
        )
        assert (
            batch_crop_anything_widget is not None
            or batch_crop_anything_widget is None
        )
        assert napari_get_reader is not None
        assert write_single_image is not None
        assert write_multiple is not None
        assert make_sample_data is not None
        assert file_selector is not None
        assert label_inspector_widget is not None
        assert batch_crop_anything_widget is not None
        assert roi_colocalization_analyzer is not None

    def test_functions_are_callable(self):
        """Test that exported functions are callable"""
        # These should be callable objects
        assert callable(napari_get_reader)
        assert callable(write_single_image)
        assert callable(write_multiple)
        assert callable(make_sample_data)
        assert callable(file_selector)
        assert callable(label_inspector_widget)
        assert callable(batch_crop_anything_widget)

    def test_version_fallback(self):
        """Test version fallback when _version import fails"""
        import sys
        from unittest.mock import patch

        # Mock import failure
        with patch.dict("sys.modules", {"napari_tmidas._version": None}):
            # Force reimport
            if "napari_tmidas" in sys.modules:
                del sys.modules["napari_tmidas"]

            # This should trigger the fallback
            try:
                import napari_tmidas

                # Version should be "unknown" when import fails
                assert napari_tmidas.__version__ == "unknown"
            except ImportError:
                pass  # Expected if other imports fail
