"""
Edge cases of ome_output_utils: writing root attributes back to either zarr
layout, and reading a frame interval from files that carry none, or carry
one that cannot be used.
"""

import json

import numpy as np
import pytest
import tifffile

from napari_tmidas.processing_functions.ome_output_utils import (
    extract_source_time_interval,
    save_root_attrs,
)


class TestSaveRootAttrs:
    def test_a_v2_group_gets_its_zattrs_replaced(self, tmp_path):
        (tmp_path / ".zattrs").write_text(json.dumps({"old": 1}))

        assert save_root_attrs(str(tmp_path), {"new": 2}) is True

        assert json.loads((tmp_path / ".zattrs").read_text()) == {"new": 2}

    def test_a_v3_group_keeps_everything_but_its_attributes(self, tmp_path):
        doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": 1}}
        (tmp_path / "zarr.json").write_text(json.dumps(doc))

        assert save_root_attrs(str(tmp_path), {"b": 2}) is True

        written = json.loads((tmp_path / "zarr.json").read_text())
        assert written == {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"b": 2},
        }

    def test_a_malformed_zarr_json_is_left_alone(self, tmp_path):
        (tmp_path / "zarr.json").write_text("[1, 2, 3]")

        assert save_root_attrs(str(tmp_path), {"b": 2}) is False

        assert (tmp_path / "zarr.json").read_text() == "[1, 2, 3]"

    def test_a_folder_without_zarr_metadata_gets_none_written(self, tmp_path):
        assert save_root_attrs(str(tmp_path), {"b": 2}) is False

        assert list(tmp_path.iterdir()) == []


def _zarr_with_attrs(tmp_path, attrs):
    store = tmp_path / "img.zarr"
    store.mkdir()
    (store / ".zattrs").write_text(json.dumps(attrs))
    return str(store)


def _multiscales(axes, scale=None):
    dataset = {"path": "0"}
    if scale is not None:
        dataset["coordinateTransformations"] = [
            {"type": "scale", "scale": scale}
        ]
    return {"multiscales": [{"axes": axes, "datasets": [dataset]}]}


class TestNoUsableFrameInterval:
    @pytest.mark.parametrize("path", [None, ""])
    def test_no_path_means_no_interval(self, path):
        assert extract_source_time_interval(path) is None

    def test_an_unreadable_tiff_has_no_interval(self, tmp_path):
        bad = tmp_path / "broken.tif"
        bad.write_bytes(b"this is not a TIFF file")

        assert extract_source_time_interval(str(bad)) is None

    def test_an_ome_tiff_without_time_increment_has_no_interval(self, tmp_path):
        path = tmp_path / "plain.ome.tif"
        tifffile.imwrite(
            str(path),
            np.zeros((3, 4, 5), dtype=np.uint8),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "TYX"},
        )

        assert extract_source_time_interval(str(path)) is None

    def test_a_zarr_without_multiscales_has_no_interval(self, tmp_path):
        assert extract_source_time_interval(_zarr_with_attrs(tmp_path, {})) is None

    def test_a_zarr_without_a_t_axis_has_no_interval(self, tmp_path):
        attrs = _multiscales(
            [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            scale=[0.5, 0.5],
        )

        assert extract_source_time_interval(_zarr_with_attrs(tmp_path, attrs)) is None

    def test_a_t_axis_whose_scale_does_not_cover_every_axis_is_ignored(
        self, tmp_path
    ):
        # Two scale values for three axes cannot be keyed to "t" reliably.
        attrs = _multiscales(
            [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            scale=[0.5, 0.5],
        )

        assert extract_source_time_interval(_zarr_with_attrs(tmp_path, attrs)) is None

    @pytest.mark.parametrize(
        "interval, unit",
        [
            ("soon", "second"),  # not a number
            (None, "second"),
            (5, "fortnight"),  # unknown unit
            (0, "second"),  # not a positive interval
            (-2, "minute"),
        ],
    )
    def test_a_recorded_interval_that_is_not_usable_is_dropped(
        self, tmp_path, interval, unit
    ):
        attrs = {
            "napari_tmidas": {
                "time_interval": interval,
                "time_interval_unit": unit,
            }
        }

        assert extract_source_time_interval(_zarr_with_attrs(tmp_path, attrs)) is None


def test_a_recorded_interval_in_minutes_is_returned_in_seconds(tmp_path):
    attrs = {"napari_tmidas": {"time_interval": 1.5, "time_interval_unit": "min"}}

    assert extract_source_time_interval(
        _zarr_with_attrs(tmp_path, attrs)
    ) == pytest.approx(90.0)
