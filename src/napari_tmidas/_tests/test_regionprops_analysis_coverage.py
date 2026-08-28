"""
Additional coverage for ``processing_functions.regionprops_analysis``.

The module has three layers that the original test file leaves untouched:

* the loader / discovery helpers (``load_label_image``,
  ``find_label_images``, ``parse_dimensions_from_shape``),
* the optional per-region properties and every ``except`` arm around them,
* the two registered batch entry points, which recover their own file path
  by walking the call stack and whose only output is a CSV on disk.

Every test here builds hand-made label arrays whose areas, centroids and
bounding boxes are known by construction, so the assertions pin real
numbers rather than "something was written".
"""

import builtins
import importlib.util
import inspect
import os
import sys
import threading

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_tmidas._registry import BatchProcessingRegistry
from napari_tmidas.processing_functions import regionprops_analysis as rpa

# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_csv_cache():
    """``_REGIONPROPS_CSV_FILES`` is a module global shared by every test."""
    rpa.reset_regionprops_cache()
    yield
    rpa.reset_regionprops_cache()


@pytest.fixture
def registry_guard():
    """``BatchProcessingRegistry._processing_functions`` is process-wide."""
    saved = dict(BatchProcessingRegistry._processing_functions)
    yield
    BatchProcessingRegistry._processing_functions.clear()
    BatchProcessingRegistry._processing_functions.update(saved)


def _call_with_filepath(func, filepath, image, **kwargs):
    """Invoke ``func`` from a frame that owns a ``filepath`` local.

    That is exactly how the batch worker calls the registered functions,
    and it is the only way they can discover which file they are given.
    """
    assert filepath  # keeps the local alive for the stack walk
    return func(image, **kwargs)


def _rectangle_labels():
    """A 30x30 image holding one 10x20 rectangle labelled 1."""
    image = np.zeros((30, 30), dtype=np.uint16)
    image[5:15, 5:25] = 1
    return image


def _two_step_intensity():
    """Intensity matching ``_rectangle_labels``: 100 px of 10, 100 px of 20."""
    intensity = np.zeros((30, 30), dtype=np.uint16)
    intensity[5:10, 5:25] = 10
    intensity[10:15, 5:25] = 20
    return intensity


ALL_PROPERTIES = [
    "label",
    "area",
    "centroid",
    "bbox",
    "perimeter",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "extent",
    "mean_intensity",
    "median_intensity",
    "std_intensity",
    "max_intensity",
    "min_intensity",
]


class _RaisingRegion:
    """A region whose every optional property is unavailable.

    ``regionprops`` genuinely behaves like this for some property/dimension
    combinations, and the module wraps each access in its own ``try``.
    """

    label = 3
    area = 12
    centroid = (1.0, 2.0)
    bbox = (0, 0, 3, 4)

    @property
    def perimeter(self):
        raise NotImplementedError("perimeter")

    @property
    def eccentricity(self):
        raise NotImplementedError("eccentricity")

    @property
    def solidity(self):
        raise NotImplementedError("solidity")

    @property
    def major_axis_length(self):
        raise AttributeError("major_axis_length")

    @property
    def minor_axis_length(self):
        raise AttributeError("minor_axis_length")

    @property
    def orientation(self):
        raise AttributeError("orientation")

    @property
    def extent(self):
        raise NotImplementedError("extent")

    @property
    def intensity_image(self):
        raise AttributeError("intensity_image")

    @property
    def image(self):
        raise AttributeError("image")

    @property
    def mean_intensity(self):
        raise AttributeError("mean_intensity")

    @property
    def max_intensity(self):
        raise NotImplementedError("max_intensity")

    @property
    def min_intensity(self):
        raise NotImplementedError("min_intensity")


class _NonNumericRegion:
    """Intensity pixels that numpy cannot reduce (median/std must survive)."""

    label = 4
    area = 4
    centroid = (0.5, 0.5)
    bbox = (0, 0, 2, 2)
    mean_intensity = 1.5
    max_intensity = 2.0
    min_intensity = 1.0
    intensity_image = np.array([["a", "b"], ["c", "d"]])
    image = np.ones((2, 2), dtype=bool)


# ---------------------------------------------------------------------------


class TestOptionalPandasFallback:
    """The module must import, and fail loudly, when pandas is missing."""

    def test_module_body_survives_missing_pandas(
        self, monkeypatch, registry_guard
    ):
        """The ``except ImportError`` arm sets ``pd = None``."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas is unavailable in this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        spec = importlib.util.spec_from_file_location(
            "_rpa_without_pandas", rpa.__file__
        )
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)

        assert fresh._HAS_PANDAS is False
        assert fresh.pd is None
        # the real module is untouched
        assert rpa._HAS_PANDAS is True
        assert rpa.pd is pd

        with pytest.raises(ImportError, match="pandas is required"):
            fresh.analyze_folder_regionprops("nowhere", "nowhere.csv")

    def test_registered_metadata_matches_the_signatures(self):
        """Both entry points advertise the defaults they actually use.

        The registry drives the widget, so a default declared there that
        disagrees with the function signature silently ships a different
        behaviour than the GUI promises.
        """
        entries = (
            (
                "Extract Regionprops to CSV",
                rpa.extract_regionprops_folder,
                "_regionprops",
            ),
            (
                "Regionprops Summary Statistics",
                rpa.extract_regionprops_summary_folder,
                "_regionprops_summary",
            ),
        )
        for name, func, suffix in entries:
            info = BatchProcessingRegistry.get_function_info(name)
            assert info["func"] is func
            assert info["suffix"] == suffix
            assert info["parameters"]

            signature = inspect.signature(func)
            for param, meta in info["parameters"].items():
                declared = signature.parameters[param]
                assert declared.default == meta["default"], param
                assert isinstance(declared.default, meta["type"]), param

    def test_analyze_folder_requires_pandas(self, monkeypatch, tmp_path):
        """``analyze_folder_regionprops`` refuses to run without pandas."""
        monkeypatch.setattr(rpa, "_HAS_PANDAS", False)
        with pytest.raises(ImportError, match="pip install pandas"):
            rpa.analyze_folder_regionprops(
                str(tmp_path), str(tmp_path / "out.csv")
            )


class TestLoadLabelImage:
    """Every extension branch of ``load_label_image``."""

    def test_npy_round_trip(self, tmp_path):
        image = _rectangle_labels()
        path = tmp_path / "labels.npy"
        np.save(path, image)

        loaded = rpa.load_label_image(str(path))

        assert loaded.dtype == np.uint16
        np.testing.assert_array_equal(loaded, image)

    def test_tiff_uses_tifffile(self, tmp_path):
        image = _rectangle_labels()
        path = tmp_path / "labels.TIF"  # extension match is case-insensitive
        tifffile.imwrite(path, image)

        loaded = rpa.load_label_image(str(path))

        assert loaded.dtype == np.uint16
        np.testing.assert_array_equal(loaded, image)

    def test_tiff_falls_back_to_skimage_without_tifffile(
        self, tmp_path, monkeypatch
    ):
        """Without tifffile, TIFFs must be routed to ``skimage.io.imread``.

        skimage's own TIFF plugin is backed by tifffile too, so the reader
        itself is replaced here and the assertion is on the delegation:
        which file the fallback asked for, and that its result is returned
        unchanged.
        """
        import skimage.io

        path = tmp_path / "labels.tif"
        path.write_bytes(b"placeholder - never actually decoded")
        expected = _rectangle_labels()
        seen = []

        def fake_imread(fname, *args, **kwargs):
            seen.append(fname)
            return expected

        monkeypatch.setitem(sys.modules, "tifffile", None)
        monkeypatch.setattr(skimage.io, "imread", fake_imread)

        loaded = rpa.load_label_image(str(path))

        assert seen == [str(path)]
        assert loaded is expected

    def test_other_extension_uses_skimage(self, tmp_path):
        from skimage.io import imsave

        image = np.zeros((8, 8), dtype=np.uint8)
        image[2:5, 2:6] = 3
        path = tmp_path / "labels.png"
        imsave(str(path), image, check_contrast=False)

        loaded = rpa.load_label_image(str(path))

        np.testing.assert_array_equal(loaded, image)


class TestFindLabelImages:
    """Discovery guards: missing folder, suffix filter, nothing found."""

    def test_missing_folder_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="Folder does not exist"):
            rpa.find_label_images(str(missing))

    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No label image files found"):
            rpa.find_label_images(str(tmp_path))

    def test_suffix_filter_drops_intensity_images(self, tmp_path):
        image = _rectangle_labels()
        for name in ("a_lbl.tif", "a.tif", "b_lbl.tif", "b.tif"):
            tifffile.imwrite(tmp_path / name, image)

        found = rpa.find_label_images(
            str(tmp_path), intensity_suffix="_lbl.tif"
        )

        assert [os.path.basename(p) for p in found] == [
            "a_lbl.tif",
            "b_lbl.tif",
        ]

    def test_suffix_matching_nothing_raises(self, tmp_path):
        tifffile.imwrite(tmp_path / "plain.tif", _rectangle_labels())
        with pytest.raises(ValueError, match="No label image files found"):
            rpa.find_label_images(
                str(tmp_path), intensity_suffix="_missing.tif"
            )

    def test_explicit_extensions_and_sorting(self, tmp_path):
        np.save(tmp_path / "z.npy", _rectangle_labels())
        np.save(tmp_path / "a.npy", _rectangle_labels())
        tifffile.imwrite(tmp_path / "m.tif", _rectangle_labels())

        found = rpa.find_label_images(str(tmp_path), extensions=[".npy"])

        assert [os.path.basename(p) for p in found] == ["a.npy", "z.npy"]


class TestParseDimensionsFromShape:
    """The 5D and fallback arms of the shape parser."""

    def test_five_dimensions_are_tczyx(self):
        dims = rpa.parse_dimensions_from_shape((2, 3, 4, 5, 6), 5)
        assert dims == {"T": 2, "C": 3, "Z": 4, "Y": 5, "X": 6}

    def test_six_dimensions_are_numbered(self):
        dims = rpa.parse_dimensions_from_shape((1, 2, 3, 4, 5, 6), 6)
        assert dims == {
            "dim_0": 1,
            "dim_1": 2,
            "dim_2": 3,
            "dim_3": 4,
            "dim_4": 5,
            "dim_5": 6,
        }

    def test_one_dimension_falls_through_to_numbering(self):
        assert rpa.parse_dimensions_from_shape((7,), 1) == {"dim_0": 7}


class TestOptionalTwoDimensionalProperties:
    """The 2D-only shape descriptors, with values fixed by construction."""

    def test_all_shape_properties_for_a_rectangle(self):
        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            properties=[
                "label",
                "area",
                "centroid",
                "bbox",
                "perimeter",
                "eccentricity",
                "solidity",
                "major_axis_length",
                "minor_axis_length",
                "orientation",
                "extent",
            ],
            max_spatial_dims=2,
        )

        assert len(results) == 1
        row = results[0]
        assert row["label"] == 1
        assert row["size"] == 200
        assert row["centroid_y"] == pytest.approx(9.5)
        assert row["centroid_x"] == pytest.approx(14.5)
        assert (
            row["bbox_min_y"],
            row["bbox_min_x"],
            row["bbox_max_y"],
            row["bbox_max_x"],
        ) == (5, 5, 15, 25)
        # a filled rectangle exactly fills its bounding box and hull
        assert row["extent"] == pytest.approx(1.0)
        assert row["solidity"] == pytest.approx(1.0)
        assert row["perimeter"] == pytest.approx(56.0)
        assert row["major_axis_length"] > row["minor_axis_length"] > 0
        assert 0.0 < row["eccentricity"] < 1.0
        assert abs(row["orientation"]) == pytest.approx(np.pi / 2)

    def test_three_dimensional_regions_skip_the_2d_properties(self):
        volume = np.zeros((4, 10, 10), dtype=np.uint16)
        volume[1:3, 2:6, 2:7] = 1

        results = rpa.extract_regionprops_recursive(
            volume, properties=ALL_PROPERTIES, max_spatial_dims=3
        )

        assert len(results) == 1
        row = results[0]
        assert row["size"] == 40
        assert row["centroid_z"] == pytest.approx(1.5)
        assert row["bbox_min_z"] == 1
        assert row["bbox_max_x"] == 7
        # perimeter/eccentricity/... are guarded behind a 2D check
        for key in (
            "perimeter",
            "eccentricity",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
        ):
            assert key not in row
        # extent is computed for 3D as well
        assert row["extent"] == pytest.approx(1.0)

    def test_area_can_be_switched_off(self):
        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            properties=["label"],
            max_spatial_dims=2,
        )
        assert results == [{"label": 1}]


class TestIntensityMeasurements:
    """Intensity pairing produces exact statistics for a known image."""

    def test_all_intensity_statistics(self):
        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            intensity_image=_two_step_intensity(),
            properties=[
                "label",
                "area",
                "mean_intensity",
                "median_intensity",
                "std_intensity",
                "max_intensity",
                "min_intensity",
            ],
            max_spatial_dims=2,
        )

        assert len(results) == 1
        row = results[0]
        assert row["size"] == 200
        assert row["mean_intensity"] == pytest.approx(15.0)
        assert row["median_intensity"] == pytest.approx(15.0)
        assert row["std_intensity"] == pytest.approx(5.0)
        assert row["max_intensity"] == pytest.approx(20.0)
        assert row["min_intensity"] == pytest.approx(10.0)

    def test_no_intensity_image_means_no_intensity_columns(self):
        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            intensity_image=None,
            properties=[
                "label",
                "area",
                "mean_intensity",
                "max_intensity",
            ],
            max_spatial_dims=2,
        )
        assert set(results[0]) == {"label", "size"}

    def test_float_labels_are_cast_before_measuring(self):
        image = _rectangle_labels().astype(np.float32)
        results = rpa.extract_regionprops_recursive(
            image, properties=["label", "area"], max_spatial_dims=2
        )
        assert results == [{"label": 1, "size": 200}]


class TestRegionPropertyFailures:
    """Each ``except`` arm around a property access, exercised for real."""

    def test_unavailable_properties_are_skipped_not_fatal(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            rpa.measure,
            "regionprops",
            lambda *a, **k: [_RaisingRegion()],
        )

        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            intensity_image=_two_step_intensity(),
            properties=ALL_PROPERTIES,
            max_spatial_dims=2,
        )

        assert len(results) == 1
        row = results[0]
        # only the always-available bits survive
        assert row["label"] == 3
        assert row["size"] == 12
        assert row["centroid_y"] == pytest.approx(1.0)
        assert row["bbox_max_x"] == 4
        for key in (
            "perimeter",
            "eccentricity",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "extent",
            "mean_intensity",
            "median_intensity",
            "std_intensity",
            "max_intensity",
            "min_intensity",
        ):
            assert key not in row

        out = capsys.readouterr().out
        assert "Could not read region intensities" in out
        assert "Could not extract mean_intensity" in out
        assert "Could not extract max_intensity" in out
        assert "Could not extract min_intensity" in out

    def test_non_numeric_intensities_only_break_median_and_std(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            rpa.measure,
            "regionprops",
            lambda *a, **k: [_NonNumericRegion()],
        )

        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(),
            intensity_image=_two_step_intensity(),
            properties=[
                "label",
                "area",
                "mean_intensity",
                "median_intensity",
                "std_intensity",
                "max_intensity",
                "min_intensity",
            ],
            max_spatial_dims=2,
        )

        row = results[0]
        assert row["mean_intensity"] == pytest.approx(1.5)
        assert row["max_intensity"] == pytest.approx(2.0)
        assert row["min_intensity"] == pytest.approx(1.0)
        assert "median_intensity" not in row
        assert "std_intensity" not in row

        out = capsys.readouterr().out
        assert "Could not extract median_intensity" in out
        assert "Could not extract std_intensity" in out

    def test_regionprops_blowing_up_returns_no_rows(self, monkeypatch, capsys):
        def explode(*args, **kwargs):
            raise RuntimeError("regionprops exploded")

        monkeypatch.setattr(rpa.measure, "regionprops", explode)

        results = rpa.extract_regionprops_recursive(
            _rectangle_labels(), max_spatial_dims=2
        )

        assert results == []
        assert "Error extracting regionprops" in capsys.readouterr().out

    def test_empty_slice_short_circuits_before_regionprops(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("regionprops must not be called")

        monkeypatch.setattr(rpa.measure, "regionprops", explode)

        assert (
            rpa.extract_regionprops_recursive(
                np.zeros((5, 5), dtype=np.uint16), max_spatial_dims=2
            )
            == []
        )


class TestDimensionNaming:
    """How the recursion names the axes it peels off."""

    @staticmethod
    def _one_pixel_stack(shape):
        image = np.zeros(shape, dtype=np.uint16)
        image[..., 1:3, 1:4] = 1
        return image

    def test_explicit_order_labels_each_axis(self):
        image = self._one_pixel_stack((2, 2, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image,
            max_spatial_dims=2,
            dimension_order="TCYX",
            properties=["label", "area"],
        )
        assert len(results) == 4
        assert {(r["T"], r["C"]) for r in results} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        assert all(r["size"] == 6 for r in results)

    def test_non_tcz_character_becomes_a_numbered_axis(self):
        image = self._one_pixel_stack((2, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image,
            max_spatial_dims=2,
            dimension_order="YXZ",
            properties=["label"],
        )
        assert len(results) == 2
        assert sorted(r["dim_0"] for r in results) == [0, 1]

    def test_order_string_shorter_than_the_array(self):
        image = self._one_pixel_stack((2, 2, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image,
            max_spatial_dims=2,
            dimension_order="T",
            properties=["label"],
        )
        assert len(results) == 4
        assert {r["T"] for r in results} == {0, 1}
        assert {r["dim_1"] for r in results} == {0, 1}

    def test_auto_three_dimensions_assume_time(self):
        image = self._one_pixel_stack((3, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image, max_spatial_dims=2, properties=["label"]
        )
        assert sorted(r["T"] for r in results) == [0, 1, 2]

    def test_auto_five_dimensions_are_t_c_then_numbered(self):
        image = self._one_pixel_stack((2, 2, 2, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image, max_spatial_dims=2, properties=["label"]
        )
        assert len(results) == 8
        assert set(results[0]) == {"T", "C", "dim_2", "label"}

    def test_auto_six_dimensions_fall_through_to_numbering(self):
        image = self._one_pixel_stack((2, 1, 1, 1, 6, 6))
        results = rpa.extract_regionprops_recursive(
            image, max_spatial_dims=2, properties=["label"]
        )
        assert len(results) == 2
        assert set(results[0]) == {
            "dim_0",
            "dim_1",
            "dim_2",
            "dim_3",
            "label",
        }


class TestAnalyzeFolderRegionprops:
    """The folder-level entry point: pairing, failures, column order."""

    def test_dimension_order_is_announced(self, tmp_path, capsys):
        np.save(tmp_path / "a.npy", _rectangle_labels())
        out_csv = tmp_path / "out.csv"

        df = rpa.analyze_folder_regionprops(
            str(tmp_path),
            str(out_csv),
            max_spatial_dims=2,
            dimension_order="YX",
        )

        assert "Using dimension order: YX" in capsys.readouterr().out
        # the announcement must not be the only thing that happened
        assert list(df["filename"]) == ["a.npy"]
        assert list(df["size"]) == [200]
        assert list(pd.read_csv(out_csv)["size"]) == [200]

    def test_intensity_pairing_adds_intensity_columns(self, tmp_path):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(folder / "s_lbl.tif", _rectangle_labels())
        tifffile.imwrite(folder / "s.tif", _two_step_intensity())
        out_csv = tmp_path / "out.csv"

        df = rpa.analyze_folder_regionprops(
            str(folder),
            str(out_csv),
            max_spatial_dims=2,
            intensity_suffix="_lbl.tif",
        )

        assert len(df) == 1
        assert df.iloc[0]["mean_intensity"] == pytest.approx(15.0)
        assert df.iloc[0]["size"] == 200
        assert out_csv.exists()

    def test_shape_mismatch_drops_the_intensity_image(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(folder / "s_lbl.tif", _rectangle_labels())
        tifffile.imwrite(folder / "s.tif", np.zeros((4, 4), dtype=np.uint16))

        df = rpa.analyze_folder_regionprops(
            str(folder),
            str(tmp_path / "out.csv"),
            max_spatial_dims=2,
            intensity_suffix="_lbl.tif",
        )

        assert "mean_intensity" not in df.columns
        assert "skipping intensity" in capsys.readouterr().out

    def test_unreadable_intensity_image_is_reported(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(folder / "s_lbl.tif", _rectangle_labels())
        (folder / "s.tif").write_bytes(b"this is not a tiff")

        df = rpa.analyze_folder_regionprops(
            str(folder),
            str(tmp_path / "out.csv"),
            max_spatial_dims=2,
            intensity_suffix="_lbl.tif",
        )

        assert len(df) == 1
        assert "mean_intensity" not in df.columns
        assert "could not load intensity image" in capsys.readouterr().out

    def test_missing_intensity_image_is_reported(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(folder / "s_lbl.tif", _rectangle_labels())

        df = rpa.analyze_folder_regionprops(
            str(folder),
            str(tmp_path / "out.csv"),
            max_spatial_dims=2,
            intensity_suffix="_lbl.tif",
        )

        assert len(df) == 1
        assert "intensity image not found: s.tif" in capsys.readouterr().out

    def test_empty_and_broken_files_yield_an_empty_csv(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(
            folder / "empty.tif", np.zeros((6, 6), dtype=np.uint16)
        )
        (folder / "broken.tif").write_bytes(b"not a tiff at all")
        out_csv = tmp_path / "out.csv"

        df = rpa.analyze_folder_regionprops(
            str(folder), str(out_csv), max_spatial_dims=2
        )

        assert list(df.columns) == ["filename", "label", "area"]
        assert len(df) == 0
        assert out_csv.exists()
        out = capsys.readouterr().out
        assert "empty, skipped" in out
        assert "Error processing broken.tif" in out
        assert "No regions found in any label images" in out

    def test_identifier_columns_come_first(self, tmp_path):
        image = np.zeros((2, 2, 2, 8, 8), dtype=np.uint16)
        image[..., 1:4, 1:5] = 1
        np.save(tmp_path / "stack.npy", image)
        out_csv = tmp_path / "out.csv"

        df = rpa.analyze_folder_regionprops(
            str(tmp_path),
            str(out_csv),
            max_spatial_dims=2,
            dimension_order="TCZYX",
        )

        assert list(df.columns)[:5] == [
            "filename",
            "T",
            "C",
            "Z",
            "label",
        ]
        assert len(df) == 8
        assert set(df["size"]) == {12}
        assert len(pd.read_csv(out_csv)) == 8


class TestGetCurrentFilepath:
    """The stack walk that gives the batch functions their file path."""

    def test_finds_the_nearest_filepath_local(self):
        def caller(filepath):
            assert filepath
            return rpa.get_current_filepath()

        assert caller("/data/sample.tif") == "/data/sample.tif"

    def test_returns_none_when_no_frame_declares_one(self):
        # a fresh thread has a stack of its own, with no ``filepath`` local
        captured = {}

        def worker():
            captured["value"] = rpa.get_current_filepath()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert captured["value"] is None


class TestExtractRegionpropsFolder:
    """The "Extract Regionprops to CSV" entry point."""

    @staticmethod
    def _csv(folder):
        return folder.parent / f"{folder.name}_regionprops.csv"

    def test_without_a_filepath_it_bails_out(self, monkeypatch, capsys):
        monkeypatch.setattr(rpa, "get_current_filepath", lambda: None)

        assert rpa.extract_regionprops_folder(_rectangle_labels()) is None
        assert "Could not determine current file path" in (
            capsys.readouterr().out
        )

    def test_files_without_the_label_suffix_are_ignored(self, tmp_path):
        folder = tmp_path / "labels"
        folder.mkdir()

        result = _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "intensity.tif"),
            _rectangle_labels(),
            label_suffix="_lbl.tif",
            overwrite_existing=True,
        )

        assert result is None
        assert not self._csv(folder).exists()

        # Positive control: the identical call on a name that *does* carry
        # the suffix writes the CSV.  Without this the test above would
        # also pass for a function whose body was simply ``return None``.
        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "sample_lbl.tif"),
            _rectangle_labels(),
            label_suffix="_lbl.tif",
            overwrite_existing=True,
        )

        df = pd.read_csv(self._csv(folder))
        assert list(df["filename"]) == ["sample_lbl.tif"]
        assert list(df["size"]) == [200]

    def test_every_property_flag_reaches_the_csv(self, tmp_path):
        folder = tmp_path / "labels"
        folder.mkdir()
        tifffile.imwrite(folder / "s.tif", _two_step_intensity())

        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "s_lbl.tif"),
            _rectangle_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
            label_suffix="_lbl.tif",
            perimeter=True,
            eccentricity=True,
            extent=True,
            solidity=True,
            major_axis=True,
            minor_axis=True,
            orientation=True,
            max_intensity=True,
            min_intensity=True,
        )

        df = pd.read_csv(self._csv(folder))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["filename"] == "s_lbl.tif"
        assert row["size"] == 200
        assert row["perimeter"] == pytest.approx(56.0)
        assert row["extent"] == pytest.approx(1.0)
        assert row["solidity"] == pytest.approx(1.0)
        assert row["mean_intensity"] == pytest.approx(15.0)
        assert row["median_intensity"] == pytest.approx(15.0)
        assert row["std_intensity"] == pytest.approx(5.0)
        assert row["max_intensity"] == pytest.approx(20.0)
        assert row["min_intensity"] == pytest.approx(10.0)
        # a solid 10x20 rectangle: the axis lengths are 4*sqrt(variance)
        major = row["major_axis_length"]
        minor = row["minor_axis_length"]
        assert major == pytest.approx(4 * np.sqrt((20**2 - 1) / 12))
        assert minor == pytest.approx(4 * np.sqrt((10**2 - 1) / 12))
        assert row["eccentricity"] == pytest.approx(
            np.sqrt(1 - (minor / major) ** 2)
        )
        assert abs(row["orientation"]) == pytest.approx(np.pi / 2)

    def test_missing_intensity_image_is_announced(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()

        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "s_lbl.tif"),
            _rectangle_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
            label_suffix="_lbl.tif",
        )

        out = capsys.readouterr().out
        assert "Intensity image not found: s.tif" in out
        df = pd.read_csv(self._csv(folder))
        assert "mean_intensity" not in df.columns

    def test_second_image_is_appended_without_a_header(self, tmp_path):
        folder = tmp_path / "labels"
        folder.mkdir()

        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "one.tif"),
            _rectangle_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
        )
        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "two.tif"),
            _rectangle_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
        )

        df = pd.read_csv(self._csv(folder))
        assert list(df["filename"]) == ["one.tif", "two.tif"]
        assert list(df["size"]) == [200, 200]

    def test_empty_image_writes_nothing(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()

        _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "blank.tif"),
            np.zeros((10, 10), dtype=np.uint16),
            max_spatial_dims=2,
            overwrite_existing=True,
        )

        assert "No regions found in blank.tif" in capsys.readouterr().out
        assert not self._csv(folder).exists()

    def test_extraction_errors_are_swallowed(
        self, tmp_path, monkeypatch, capsys
    ):
        folder = tmp_path / "labels"
        folder.mkdir()

        def explode(*args, **kwargs):
            raise RuntimeError("extraction failed")

        monkeypatch.setattr(rpa, "extract_regionprops_recursive", explode)

        result = _call_with_filepath(
            rpa.extract_regionprops_folder,
            str(folder / "s.tif"),
            _rectangle_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
        )

        assert result is None
        assert "Error processing s.tif" in capsys.readouterr().out
        assert not self._csv(folder).exists()


class TestExtractRegionpropsSummaryFolder:
    """The "Regionprops Summary Statistics" entry point."""

    @staticmethod
    def _csv(folder):
        return folder.parent / f"{folder.name}_regionprops_summary.csv"

    @staticmethod
    def _two_labels():
        image = np.zeros((30, 30), dtype=np.uint16)
        image[5:15, 5:25] = 1  # 200 px
        image[20:25, 5:15] = 2  # 50 px
        return image

    def test_min_and_max_intensity_statistics(self, tmp_path):
        folder = tmp_path / "labels"
        folder.mkdir()
        np.save(folder / "s.npy", _two_step_intensity())

        _call_with_filepath(
            rpa.extract_regionprops_summary_folder,
            str(folder / "s_lbl.npy"),
            self._two_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
            label_suffix="_lbl.npy",
            max_intensity=True,
            min_intensity=True,
        )

        df = pd.read_csv(self._csv(folder))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["label_count"] == 2
        assert row["size_sum"] == 250
        assert row["size_mean"] == pytest.approx(125.0)
        assert row["size_median"] == pytest.approx(125.0)
        assert row["size_std"] == pytest.approx(106.0660, rel=1e-4)
        # label 1 spans the 10/20 step, label 2 lies on background (0)
        assert row["mean_int_sum"] == pytest.approx(15.0)
        assert row["max_int_sum"] == pytest.approx(20.0)
        assert row["min_int_sum"] == pytest.approx(10.0)
        assert row["median_int_mean"] == pytest.approx(7.5)
        assert row["std_int_sum"] == pytest.approx(5.0)

    def test_intensity_image_not_found_is_announced(self, tmp_path, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()

        _call_with_filepath(
            rpa.extract_regionprops_summary_folder,
            str(folder / "s_lbl.npy"),
            self._two_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
            label_suffix="_lbl.npy",
        )

        assert "Intensity image not found: s.npy" in capsys.readouterr().out
        df = pd.read_csv(self._csv(folder))
        assert "mean_int_sum" not in df.columns

    def test_grouping_by_a_single_dimension(self, tmp_path):
        image = np.zeros((2, 20, 20), dtype=np.uint16)
        image[0, 2:6, 2:7] = 1  # 20 px
        image[1, 2:6, 2:7] = 1  # 20 px
        image[1, 10:12, 10:15] = 2  # 10 px
        folder = tmp_path / "labels"
        folder.mkdir()

        _call_with_filepath(
            rpa.extract_regionprops_summary_folder,
            str(folder / "s.npy"),
            image,
            max_spatial_dims=2,
            overwrite_existing=True,
            group_by_dimensions=True,
            dimension_order="TYX",
            mean_intensity=False,
            median_intensity=False,
            std_intensity=False,
        )

        df = pd.read_csv(self._csv(folder))
        assert len(df) == 2
        assert "T" in df.columns
        # T must hold plain integers, not the pandas 1-tuple groupby key
        # ("(0,)"/"(1,)") that a single-element grouping list produces on
        # modern pandas -- and the counts/sums/T triples must stay paired
        # to their own group, not just independently sorted.
        assert sorted(df["T"]) == [0, 1]
        rows = sorted(
            zip(
                (int(t) for t in df["T"]),
                (int(c) for c in df["label_count"]),
                (int(v) for v in df["size_sum"]),
            )
        )
        assert rows == [(0, 1, 20), (1, 2, 30)]

    def test_grouping_by_two_dimensions_with_intensities(self, tmp_path):
        labels = np.zeros((2, 2, 10, 10), dtype=np.uint16)
        labels[:, :, 1:4, 1:5] = 1  # 12 px per (T, Z)
        intensity = np.full((2, 2, 10, 10), 4, dtype=np.uint16)
        intensity[1] = 8

        folder = tmp_path / "labels"
        folder.mkdir()
        np.save(folder / "s.npy", intensity)

        _call_with_filepath(
            rpa.extract_regionprops_summary_folder,
            str(folder / "s_lbl.npy"),
            labels,
            max_spatial_dims=2,
            overwrite_existing=True,
            label_suffix="_lbl.npy",
            group_by_dimensions=True,
            dimension_order="TZYX",
        )

        df = pd.read_csv(self._csv(folder)).sort_values(["T", "Z"])
        assert len(df) == 4
        assert list(df["T"]) == [0, 0, 1, 1]
        assert list(df["Z"]) == [0, 1, 0, 1]
        assert set(df["size_sum"]) == {12}
        assert list(df["mean_int_mean"]) == [4.0, 4.0, 8.0, 8.0]
        assert list(df["median_int_sum"]) == [4.0, 4.0, 8.0, 8.0]
        assert set(df["std_int_sum"]) == {0.0}

    def test_summary_errors_are_swallowed(self, tmp_path, monkeypatch, capsys):
        folder = tmp_path / "labels"
        folder.mkdir()

        def explode(*args, **kwargs):
            raise RuntimeError("summary failed")

        monkeypatch.setattr(rpa, "extract_regionprops_recursive", explode)

        result = _call_with_filepath(
            rpa.extract_regionprops_summary_folder,
            str(folder / "s.npy"),
            self._two_labels(),
            max_spatial_dims=2,
            overwrite_existing=True,
        )

        assert result is None
        assert "Error processing s.npy" in capsys.readouterr().out
        assert not self._csv(folder).exists()
