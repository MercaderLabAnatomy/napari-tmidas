"""
Tests for ROI colocalization analysis (``_roi_colocalization``).

This is quantification code: the numbers it produces end up in a CSV that
somebody reports.  Almost all of it is pure array work behind a Qt shell,
so these tests drive the statistics directly on small hand-built label
images whose expected counts and sizes can be worked out by eye, and then
check that the same numbers survive the full ``process_colocalization``
pipeline (which crops each ROI to its bounding box before measuring it).
"""

import csv
import os
import sys

import numpy as np
import pytest
import tifffile

from napari_tmidas import _roi_colocalization as rc

pytest.importorskip("pytestqt")

requires_gui = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


# --------------------------------------------------------------------------
# A small, fully hand-checkable scene.
#
#   channel 1 (ROIs)     label 1 = rows 0:6,  cols 0:6   (36 px)
#                        label 2 = rows 6:12, cols 6:12  (36 px)
#   channel 2 (nuclei)   10 and 11 inside ROI 1, 12 inside ROI 2 (4 px each)
#   channel 3 (spots)    20 inside nucleus 10, 21 in ROI 1 but in no
#                        nucleus, 22 inside nucleus 12
# --------------------------------------------------------------------------


def make_c1():
    image = np.zeros((12, 12), dtype=np.uint16)
    image[0:6, 0:6] = 1
    image[6:12, 6:12] = 2
    return image


def make_c2():
    image = np.zeros((12, 12), dtype=np.uint16)
    image[1:3, 1:3] = 10
    image[4:6, 4:6] = 11
    image[7:9, 7:9] = 12
    return image


def make_c3():
    image = np.zeros((12, 12), dtype=np.uint16)
    image[1, 1] = 20
    image[0, 3] = 21
    image[7, 7] = 22
    return image


@pytest.fixture
def worker_factory(qapp):
    def build(**kwargs):
        kwargs.setdefault("file_pairs", [])
        kwargs.setdefault("channel_names", ["CH1", "CH2", "CH3"])
        return rc.ColocalizationWorker(**kwargs)

    return build


@pytest.fixture
def worker(worker_factory):
    return worker_factory()


class TestSemanticToInstanceLabels:
    def test_none_passes_through(self):
        assert rc.convert_semantic_to_instance_labels(None) is None

    def test_empty_image_passes_through(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        np.testing.assert_array_equal(
            rc.convert_semantic_to_instance_labels(image), image
        )

    def test_one_semantic_value_becomes_separate_instances(self):
        image = np.zeros((6, 6), dtype=np.uint8)
        image[0:2, 0:2] = 7
        image[4:6, 4:6] = 7

        result = rc.convert_semantic_to_instance_labels(image)

        assert sorted(np.unique(result[result != 0]).tolist()) == [1, 2]
        assert result[0, 0] != result[5, 5]

    def test_disconnected_regions_keep_their_own_label(self):
        image = np.zeros((6, 6), dtype=np.uint8)
        image[0:2, 0:2] = 3
        image[4:6, 4:6] = 9

        result = rc.convert_semantic_to_instance_labels(image)
        assert len(np.unique(result[result != 0])) == 2

    def test_touching_regions_merge_into_one_instance(self):
        # Connected components works on the binary foreground, so two
        # differently-labelled but adjacent blobs become one object. That
        # is what "convert semantic to instance" is asked to do, but it is
        # worth pinning: ticking the box on already-instanced labels is
        # lossy.
        image = np.zeros((4, 6), dtype=np.uint8)
        image[1:3, 1:3] = 5
        image[1:3, 3:5] = 6

        result = rc.convert_semantic_to_instance_labels(image)
        assert len(np.unique(result[result != 0])) == 1

    def test_connectivity_argument_is_honoured(self):
        # Two blobs touching only at a corner: separate under connectivity
        # 1, joined under connectivity 2.
        image = np.zeros((4, 4), dtype=np.uint8)
        image[0:2, 0:2] = 1
        image[2:4, 2:4] = 1

        assert (
            len(
                np.unique(
                    rc.convert_semantic_to_instance_labels(
                        image, connectivity=1
                    )[image != 0]
                )
            )
            == 2
        )
        assert (
            len(
                np.unique(
                    rc.convert_semantic_to_instance_labels(
                        image, connectivity=2
                    )[image != 0]
                )
            )
            == 1
        )


class TestLongestCommonSubstring:
    def test_finds_shared_run(self):
        assert (
            rc.longest_common_substring("embryo7_a.tif", "embryo7_b.tif")
            == "embryo7_"
        )

    def test_a_shared_suffix_can_beat_the_sample_identifier(self):
        # 11 characters of convention against 10 of identity. This is why
        # grouping strips shared affixes before comparing.
        assert (
            rc.longest_common_substring(
                "sample01_c1_labels.tif", "sample01_c2_labels.tif"
            )
            == "_labels.tif"
        )

    def test_no_overlap_yields_empty(self):
        assert rc.longest_common_substring("abc", "xyz") == ""

    def test_identical_strings(self):
        assert rc.longest_common_substring("abc", "abc") == "abc"


class TestGroupFilesByCommonSubstring:
    def test_pairs_two_channels(self):
        file_lists = {
            "CH1": ["/a/sample01_c1.tif", "/a/sample02_c1.tif"],
            "CH2": ["/b/sample02_c2.tif", "/b/sample01_c2.tif"],
        }
        groups = rc.group_files_by_common_substring(file_lists, ["CH1", "CH2"])

        pairs = {
            tuple(os.path.basename(p) for p in paths)
            for paths in groups.values()
        }
        assert pairs == {
            ("sample01_c1.tif", "sample01_c2.tif"),
            ("sample02_c1.tif", "sample02_c2.tif"),
        }

    def test_groups_three_channels(self):
        file_lists = {
            "CH1": ["/a/embryo7_c1.tif"],
            "CH2": ["/b/embryo7_c2.tif"],
            "CH3": ["/c/embryo7_c3.tif"],
        }
        groups = rc.group_files_by_common_substring(
            file_lists, ["CH1", "CH2", "CH3"]
        )

        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 3

    def test_channel_without_candidates_yields_no_group(self):
        file_lists = {"CH1": ["/a/sample01_c1.tif"], "CH2": []}
        assert (
            rc.group_files_by_common_substring(file_lists, ["CH1", "CH2"])
            == {}
        )


class TestGroupingWithASharedSuffix:
    """
    The widget's own default pattern is ``*_labels.tif``, so every file
    ends in a run longer than most sample identifiers.  Matching on the
    raw filename latched onto that shared suffix instead of the sample:
    three samples went in and one mismatched pair came out.
    """

    def _lists(self, samples, channels=("c1", "c2"), suffix="_labels.tif"):
        return {
            f"CH{i + 1}": [
                f"/{channel}/{sample}_{channel}{suffix}" for sample in samples
            ]
            for i, channel in enumerate(channels)
        }

    def test_every_sample_survives(self):
        lists = self._lists(["sample01", "sample02", "sample03"])
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])
        assert len(groups) == 3

    def test_pairs_stay_within_their_sample(self):
        lists = self._lists(["sample01", "sample02", "sample03"])
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])

        for paths in groups.values():
            samples = {os.path.basename(path).split("_")[0] for path in paths}
            assert len(samples) == 1, f"mismatched pair: {paths}"

    def test_three_channels_stay_within_their_sample(self):
        lists = self._lists(["embryo7", "embryo8"], ("c1", "c2", "c3"))
        groups = rc.group_files_by_common_substring(
            lists, ["CH1", "CH2", "CH3"]
        )

        assert len(groups) == 2
        for paths in groups.values():
            assert len(paths) == 3
            samples = {os.path.basename(path).split("_")[0] for path in paths}
            assert len(samples) == 1

    def test_group_key_stays_readable(self):
        lists = self._lists(["sample01", "sample02"])
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])
        # This key is what the results table shows as the row identifier.
        assert sorted(groups) == ["sample01", "sample02"]

    def test_identifier_at_the_end_of_the_name(self):
        lists = {
            "CH1": ["/a/labels_c1_s01.tif", "/a/labels_c1_s02.tif"],
            "CH2": ["/b/labels_c2_s01.tif", "/b/labels_c2_s02.tif"],
        }
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])

        assert len(groups) == 2
        for paths in groups.values():
            ids = {os.path.basename(p).rsplit("_", 1)[1] for p in paths}
            assert len(ids) == 1

    def test_confusable_identifiers_do_not_cross(self):
        lists = self._lists(["s01", "s10"])
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])

        assert len(groups) == 2
        for paths in groups.values():
            samples = {os.path.basename(path).split("_")[0] for path in paths}
            assert len(samples) == 1

    def test_single_file_per_channel_still_pairs(self):
        lists = self._lists(["only"])
        groups = rc.group_files_by_common_substring(lists, ["CH1", "CH2"])
        assert len(groups) == 1


class TestSharedAffixHelpers:
    def test_shared_affixes(self):
        prefix, suffix = rc._shared_affixes(
            ["sample01_c1_labels.tif", "sample02_c1_labels.tif"]
        )
        assert prefix == "sample0"
        assert suffix == "_c1_labels.tif"

    def test_single_name_has_no_shared_affixes(self):
        assert rc._shared_affixes(["only.tif"]) == ("", "")

    def test_strip_affixes(self):
        assert (
            rc._strip_affixes("sample01_c1.tif", "sample0", "_c1.tif") == "1"
        )

    def test_strip_affixes_falls_back_when_nothing_is_left(self):
        # Identical names leave nothing to distinguish them by.
        assert rc._strip_affixes("a.tif", "a.tif", "a.tif") == "a.tif"


class TestLabelHelpers:
    def test_get_nonzero_labels(self, worker):
        assert worker.get_nonzero_labels(make_c2()) == [10, 11, 12]

    def test_get_nonzero_labels_of_empty_image(self, worker):
        assert worker.get_nonzero_labels(np.zeros((4, 4), np.uint8)) == []

    def test_count_unique_nonzero(self, worker):
        image_c2 = make_c2()
        mask = make_c1() == 1
        assert worker.count_unique_nonzero(image_c2, mask) == 2

    def test_count_unique_nonzero_excludes_background(self, worker):
        # The mask covers background as well as two labels.
        array = np.array([[0, 0], [5, 6]], dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)
        assert worker.count_unique_nonzero(array, mask) == 2

    def test_count_unique_nonzero_of_empty_selection(self, worker):
        array = np.zeros((2, 2), dtype=np.uint8)
        assert worker.count_unique_nonzero(array, array.astype(bool)) == 0

    def test_calculate_all_rois_size(self, worker):
        assert worker.calculate_all_rois_size(make_c1()) == {1: 36, 2: 36}

    def test_calculate_all_rois_size_of_empty_array(self, worker):
        assert worker.calculate_all_rois_size(np.array([], np.uint8)) == {}

    def test_calculate_all_rois_size_survives_unsupported_dtype(self, worker):
        # bincount rejects floats; the helper degrades to {} rather than
        # taking the whole run down.
        assert (
            worker.calculate_all_rois_size(np.zeros((4, 4), np.float32)) == {}
        )


class TestLabelsInRoi:
    def test_groups_voxels_by_label(self, worker):
        labels, groups, selector = rc.ColocalizationWorker._labels_in_roi(
            make_c2(), make_c1() == 1
        )

        assert labels.tolist() == [10, 11]
        assert np.bincount(groups).tolist() == [4, 4]
        assert selector.sum() == 8

    def test_selector_indexes_other_channels(self, worker):
        image_c1, image_c2, image_c3 = make_c1(), make_c2(), make_c3()
        mask = image_c1 == 1
        labels, groups, selector = rc.ColocalizationWorker._labels_in_roi(
            image_c2, mask
        )

        # Spot 20 sits inside nucleus 10, which is group 0.
        c3_values = image_c3[mask][selector]
        assert c3_values[groups == 0].max() == 20
        assert c3_values[groups == 1].max() == 0

    def test_empty_roi_returns_empty_arrays(self, worker):
        labels, groups, selector = rc.ColocalizationWorker._labels_in_roi(
            np.zeros((4, 4), np.uint16), np.ones((4, 4), bool)
        )
        assert labels.size == 0
        assert groups.size == 0


class TestCalculateColocSize:
    def test_plain_overlap(self, worker):
        # All of channel 2 that falls inside ROI 1: nuclei 10 and 11.
        assert worker.calculate_coloc_size(make_c1(), make_c2(), 1) == 8

    def test_overlap_for_second_roi(self, worker):
        assert worker.calculate_coloc_size(make_c1(), make_c2(), 2) == 4

    def test_where_c2_present_with_third_channel(self, worker):
        size = worker.calculate_coloc_size(
            make_c1(), make_c2(), 1, mask_c2=True, image_c3=make_c3()
        )
        assert size == 1  # spot 20 only

    def test_where_c2_absent_with_third_channel(self, worker):
        size = worker.calculate_coloc_size(
            make_c1(), make_c2(), 1, mask_c2=False, image_c3=make_c3()
        )
        assert size == 1  # spot 21 only

    def test_where_c2_absent_without_third_channel_counts_area(self, worker):
        # ROI 1 is 36 px, 8 of them covered by channel 2.
        size = worker.calculate_coloc_size(
            make_c1(), make_c2(), 1, mask_c2=False
        )
        assert size == 28

    def test_missing_label_has_no_overlap(self, worker):
        assert worker.calculate_coloc_size(make_c1(), make_c2(), 99) == 0


class TestIntensityStats:
    def test_stats_of_masked_region(self, worker):
        intensity = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0:4] = True  # values 0,1,2,3

        stats = worker.calculate_intensity_stats(intensity, mask)

        assert stats["mean"] == pytest.approx(1.5)
        assert stats["median"] == pytest.approx(1.5)
        assert stats["std"] == pytest.approx(np.std([0, 1, 2, 3]))
        assert stats["max"] == 3.0
        assert stats["min"] == 0.0

    def test_empty_mask_yields_zeros(self, worker):
        stats = worker.calculate_intensity_stats(
            np.ones((4, 4)), np.zeros((4, 4), dtype=bool)
        )
        assert stats == {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
        }


class TestGroupedIntensityStats:
    def test_matches_a_naive_per_group_computation(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=200)
        groups = rng.integers(0, 5, size=200)

        stats = rc.ColocalizationWorker._grouped_intensity_stats(
            values, groups, 5
        )

        for index in range(5):
            block = values[groups == index]
            assert stats["mean"][index] == pytest.approx(np.mean(block))
            assert stats["median"][index] == pytest.approx(np.median(block))
            assert stats["std"][index] == pytest.approx(np.std(block))

    def test_empty_group_reports_zero(self):
        values = np.array([1.0, 2.0, 3.0])
        groups = np.array([0, 0, 2])

        stats = rc.ColocalizationWorker._grouped_intensity_stats(
            values, groups, 3
        )

        assert stats["mean"][1] == 0.0
        assert stats["median"][1] == 0.0
        assert stats["std"][1] == 0.0
        assert stats["mean"][0] == pytest.approx(1.5)


class TestIndividualC2Measurements:
    def test_sizes_per_nucleus(self, worker):
        sizes = worker.calculate_individual_c2_sizes(make_c2(), make_c1() == 1)
        assert sizes == {10: 4, 11: 4}

    def test_sizes_restricted_to_third_channel(self, worker):
        sizes = worker.calculate_individual_c2_sizes(
            make_c2(), make_c1() == 1, image_c3=make_c3()
        )
        # Only nucleus 10 contains a spot, and only one voxel of it.
        assert sizes == {10: 1, 11: 0}

    def test_intensities_per_nucleus(self, worker):
        intensity = np.zeros((12, 12), dtype=np.float32)
        intensity[1:3, 1:3] = 8.0  # all of nucleus 10
        intensity[4, 4] = 4.0  # one voxel of nucleus 11

        means = worker.calculate_individual_c2_intensities(
            make_c2(), intensity, make_c1() == 1
        )

        assert means[10] == pytest.approx(8.0)
        assert means[11] == pytest.approx(1.0)  # 4.0 over 4 voxels

    def test_empty_roi_yields_empty_mapping(self, worker):
        empty = np.zeros((12, 12), dtype=bool)
        assert worker.calculate_individual_c2_sizes(make_c2(), empty) == {}
        assert (
            worker.calculate_individual_c2_intensities(
                make_c2(), np.zeros((12, 12), np.float32), empty
            )
            == {}
        )


class TestCountPositiveObjects:
    def _intensity(self):
        intensity = np.zeros((12, 12), dtype=np.float32)
        intensity[1:3, 1:3] = 10.0  # nucleus 10 is bright
        intensity[4:6, 4:6] = 1.0  # nucleus 11 is dim
        return intensity

    def test_absolute_threshold(self, worker):
        counts = worker.count_positive_objects(
            make_c2(),
            self._intensity(),
            make_c1() == 1,
            threshold_method="absolute",
            threshold_value=5.0,
        )

        assert counts["total_c2_objects"] == 2
        assert counts["positive_c2_objects"] == 1
        assert counts["negative_c2_objects"] == 1
        assert counts["percent_positive"] == pytest.approx(50.0)
        assert counts["threshold_used"] == 5.0

    def test_percentile_threshold_is_taken_over_voxels_under_c2(self, worker):
        counts = worker.count_positive_objects(
            make_c2(),
            self._intensity(),
            make_c1() == 1,
            threshold_method="percentile",
            threshold_value=75.0,
        )
        # Voxels under channel 2 are four 1.0s and four 10.0s.
        assert counts["threshold_used"] == pytest.approx(
            np.percentile([1.0] * 4 + [10.0] * 4, 75.0)
        )
        assert counts["positive_c2_objects"] == 1

    def test_threshold_at_the_value_counts_as_positive(self, worker):
        counts = worker.count_positive_objects(
            make_c2(),
            self._intensity(),
            make_c1() == 1,
            threshold_method="absolute",
            threshold_value=1.0,
        )
        assert counts["positive_c2_objects"] == 2

    def test_empty_roi(self, worker):
        counts = worker.count_positive_objects(
            np.zeros((12, 12), np.uint16),
            self._intensity(),
            np.ones((12, 12), bool),
        )
        assert counts["total_c2_objects"] == 0
        assert counts["percent_positive"] == 0.0


class TestCountC2PositiveForC3Labels:
    def test_counts_nuclei_containing_a_spot(self, worker):
        counts = worker.count_c2_positive_for_c3_labels(
            make_c2(), make_c3(), make_c1() == 1
        )

        assert counts["total_c2_objects"] == 2
        assert counts["c2_positive_for_c3_count"] == 1
        assert counts["c2_negative_for_c3_count"] == 1
        assert counts["c2_percent_positive_for_c3"] == pytest.approx(50.0)

    def test_no_spots_means_none_positive(self, worker):
        counts = worker.count_c2_positive_for_c3_labels(
            make_c2(), np.zeros((12, 12), np.uint16), make_c1() == 1
        )
        assert counts["c2_positive_for_c3_count"] == 0
        assert counts["c2_percent_positive_for_c3"] == 0.0

    def test_empty_roi(self, worker):
        counts = worker.count_c2_positive_for_c3_labels(
            np.zeros((12, 12), np.uint16),
            make_c3(),
            np.ones((12, 12), bool),
        )
        assert counts["total_c2_objects"] == 0


class TestProcessColocalization:
    def test_two_label_channels_without_sizes(self, worker_factory):
        worker = worker_factory(channel_names=["CH1", "CH2"])
        output = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2()
        )

        assert output["filename"] == "sample.tif"
        assert output["csv_rows"] == [
            ["sample.tif", 1, 2],
            ["sample.tif", 2, 1],
        ]
        assert [r["label_id"] for r in output["results"]] == [1, 2]
        assert [r["ch2_in_ch1_count"] for r in output["results"]] == [2, 1]

    def test_two_label_channels_with_sizes(self, worker_factory):
        worker = worker_factory(channel_names=["CH1", "CH2"], get_sizes=True)
        output = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2()
        )

        # filename, label, count, roi size, overlap size
        assert output["csv_rows"] == [
            ["sample.tif", 1, 2, 36, 8],
            ["sample.tif", 2, 1, 36, 4],
        ]
        assert output["results"][0]["ch1_size"] == 36
        assert output["results"][0]["ch2_in_ch1_size"] == 8

    def test_three_label_channels_with_sizes(self, worker_factory):
        worker = worker_factory(get_sizes=True)
        output = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2(), make_c3()
        )

        assert output["csv_rows"] == [
            ["sample.tif", 1, 2, 36, 8, 1, 1, 1, 1],
            ["sample.tif", 2, 1, 36, 4, 1, 0, 1, 0],
        ]
        first = output["results"][0]
        assert first["ch3_in_ch2_in_ch1_count"] == 1
        assert first["ch3_not_in_ch2_but_in_ch1_count"] == 1
        assert first["ch3_in_ch2_in_ch1_size"] == 1
        assert first["ch3_not_in_ch2_but_in_ch1_size"] == 1

    def test_cropping_to_the_bounding_box_does_not_change_the_numbers(
        self, worker_factory
    ):
        # Each ROI is measured inside its own bbox; padding the scene must
        # not move any count.
        worker = worker_factory(get_sizes=True)
        pad = ((5, 7), (3, 9))
        padded = worker.process_colocalization(
            "sample.tif",
            np.pad(make_c1(), pad),
            np.pad(make_c2(), pad),
            np.pad(make_c3(), pad),
        )
        tight = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2(), make_c3()
        )
        assert padded["csv_rows"] == tight["csv_rows"]

    def test_channel3_as_intensity(self, worker_factory):
        worker = worker_factory(channel3_is_labels=False)
        intensity = np.zeros((12, 12), dtype=np.float32)
        intensity[1:3, 1:3] = 10.0  # under nucleus 10

        output = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2(), intensity
        )

        row = output["csv_rows"][0]
        # filename, label, c2 count, then 4 stats inside c2 + 4 outside
        assert row[:3] == ["sample.tif", 1, 2]
        assert len(row) == 11
        in_c2_mean, _, _, in_c2_max = row[3:7]
        assert in_c2_mean == pytest.approx(5.0)  # four 10s, four 0s
        assert in_c2_max == pytest.approx(10.0)
        assert row[7:11] == [0.0, 0.0, 0.0, 0.0]  # nothing outside c2

    def test_channel2_as_intensity(self, worker_factory):
        worker = worker_factory(
            channel_names=["CH1", "CH2"], channel2_is_labels=False
        )
        intensity = np.zeros((12, 12), dtype=np.float32)
        intensity[0:6, 0:6] = 4.0

        output = worker.process_colocalization(
            "sample.tif", make_c1(), intensity
        )

        row = output["csv_rows"][0]
        assert row[0:2] == ["sample.tif", 1]
        assert row[2] == pytest.approx(4.0)  # mean over ROI 1
        assert row[5] == pytest.approx(4.0)  # max over ROI 1

    def test_counting_c2_positive_for_c3(self, worker_factory):
        worker = worker_factory(count_c2_positive_for_c3=True)
        output = worker.process_colocalization(
            "sample.tif", make_c1(), make_c2(), make_c3()
        )

        # trailing three columns: positive, negative, percent
        assert output["csv_rows"][0][-3:] == [1, 1, pytest.approx(50.0)]

    def test_semantic_conversion_is_applied_when_requested(
        self, worker_factory
    ):
        image_c2 = np.zeros((12, 12), dtype=np.uint16)
        image_c2[1:3, 1:3] = 5  # one semantic value...
        image_c2[4:6, 4:6] = 5  # ...for two separate blobs

        plain = worker_factory(channel_names=["CH1", "CH2"])
        assert (
            plain.process_colocalization("s.tif", make_c1(), image_c2)[
                "csv_rows"
            ][0][2]
            == 1
        )

        converting = worker_factory(
            channel_names=["CH1", "CH2"], convert_to_instances_c2=True
        )
        assert (
            converting.process_colocalization("s.tif", make_c1(), image_c2)[
                "csv_rows"
            ][0][2]
            == 2
        )

    def test_empty_roi_channel_yields_no_rows(self, worker_factory):
        worker = worker_factory(channel_names=["CH1", "CH2"])
        output = worker.process_colocalization(
            "sample.tif", np.zeros((12, 12), np.uint16), make_c2()
        )
        assert output["csv_rows"] == []
        assert output["results"] == []

    def test_non_contiguous_label_ids(self, worker_factory):
        image_c1 = np.zeros((12, 12), dtype=np.uint16)
        image_c1[0:6, 0:6] = 4
        image_c1[6:12, 6:12] = 9

        worker = worker_factory(channel_names=["CH1", "CH2"])
        output = worker.process_colocalization(
            "sample.tif", image_c1, make_c2()
        )

        assert [row[1] for row in output["csv_rows"]] == [4, 9]
        assert [row[2] for row in output["csv_rows"]] == [2, 1]


class TestProcessFilePair:
    def _write(self, tmp_path, name, image):
        path = tmp_path / name
        tifffile.imwrite(path, image)
        return str(path)

    def test_two_channel_pair(self, worker_factory, tmp_path):
        pair = [
            self._write(tmp_path, "s_c1.tif", make_c1()),
            self._write(tmp_path, "s_c2.tif", make_c2()),
        ]
        worker = worker_factory(
            file_pairs=[pair], channel_names=["CH1", "CH2"]
        )

        output = worker.process_file_pair(pair)

        assert output["filename"] == "s_c1.tif"
        assert [row[2] for row in output["csv_rows"]] == [2, 1]

    def test_three_channel_pair(self, worker_factory, tmp_path):
        pair = [
            self._write(tmp_path, "s_c1.tif", make_c1()),
            self._write(tmp_path, "s_c2.tif", make_c2()),
            self._write(tmp_path, "s_c3.tif", make_c3()),
        ]
        worker = worker_factory(file_pairs=[pair])

        output = worker.process_file_pair(pair)
        assert output["csv_rows"][0] == ["s_c1.tif", 1, 2, 1, 1]

    def test_mismatched_shapes_raise(self, worker_factory, tmp_path):
        pair = [
            self._write(tmp_path, "s_c1.tif", make_c1()),
            self._write(tmp_path, "s_c2.tif", np.zeros((8, 8), np.uint16)),
        ]
        worker = worker_factory(channel_names=["CH1", "CH2"])

        with pytest.raises(ValueError, match="shapes don't match"):
            worker.process_file_pair(pair)

    def test_mismatched_third_channel_raises(self, worker_factory, tmp_path):
        pair = [
            self._write(tmp_path, "s_c1.tif", make_c1()),
            self._write(tmp_path, "s_c2.tif", make_c2()),
            self._write(tmp_path, "s_c3.tif", np.zeros((8, 8), np.uint16)),
        ]
        worker = worker_factory()

        with pytest.raises(ValueError, match="shapes don't match"):
            worker.process_file_pair(pair)

    def test_missing_file_raises(self, worker_factory, tmp_path):
        worker = worker_factory(channel_names=["CH1", "CH2"])
        with pytest.raises(ValueError):
            worker.process_file_pair(
                [str(tmp_path / "gone_c1.tif"), str(tmp_path / "gone_c2.tif")]
            )


class TestSaveOutputImage:
    def _pair(self, tmp_path):
        c1 = tmp_path / "s_c1.tif"
        c2 = tmp_path / "s_c2.tif"
        c3 = tmp_path / "s_c3.tif"
        tifffile.imwrite(c1, make_c1())
        tifffile.imwrite(c2, make_c2())
        tifffile.imwrite(c3, make_c3())
        return [str(c1), str(c2), str(c3)]

    def test_writes_three_plane_visualisation(self, worker_factory, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        pair = self._pair(tmp_path)
        worker = worker_factory(output_folder=str(out))

        results = worker.process_colocalization(
            "s_c1.tif", make_c1(), make_c2(), make_c3()
        )
        worker.save_output_image(results, pair)

        written = tifffile.imread(results["output_path"])
        assert written.shape == (3,) + make_c1().shape
        # plane 0 is channel 1 verbatim
        np.testing.assert_array_equal(written[0], make_c1())
        # both ROIs overlap channel 2, and both contain a channel 3 spot
        assert set(np.unique(written[1])) == {0, 1, 2}
        assert set(np.unique(written[2])) == {0, 1, 2}

    def test_output_path_uses_channel_names(self, worker_factory, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        pair = self._pair(tmp_path)
        worker = worker_factory(
            output_folder=str(out), channel_names=["nuc", "ki67", "spot"]
        )

        results = worker.process_colocalization(
            "s_c1.tif", make_c1(), make_c2(), make_c3()
        )
        worker.save_output_image(results, pair)

        assert os.path.basename(results["output_path"]) == (
            "s_c1_nuc_ki67_spot_coloc.tif"
        )

    def test_no_output_folder_writes_nothing(self, worker_factory, tmp_path):
        pair = self._pair(tmp_path)
        worker = worker_factory(output_folder=None)
        results = {"results": []}

        worker.save_output_image(results, pair)
        assert "output_path" not in results

    def test_save_images_disabled_writes_nothing(
        self, worker_factory, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        pair = self._pair(tmp_path)
        worker = worker_factory(output_folder=str(out), save_images=False)
        results = {"results": []}

        worker.save_output_image(results, pair)
        assert "output_path" not in results
        assert list(out.iterdir()) == []

    def test_rois_without_overlap_stay_empty(self, worker_factory, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        c1 = tmp_path / "e_c1.tif"
        c2 = tmp_path / "e_c2.tif"
        tifffile.imwrite(c1, make_c1())
        tifffile.imwrite(c2, np.zeros((12, 12), np.uint16))
        worker = worker_factory(
            output_folder=str(out), channel_names=["CH1", "CH2"]
        )

        results = worker.process_colocalization(
            "e_c1.tif", make_c1(), np.zeros((12, 12), np.uint16)
        )
        worker.save_output_image(results, [str(c1), str(c2)])

        written = tifffile.imread(results["output_path"])
        assert not written[1].any()


class TestWorkerRun:
    def _pair(self, tmp_path):
        c1 = tmp_path / "s_c1.tif"
        c2 = tmp_path / "s_c2.tif"
        tifffile.imwrite(c1, make_c1())
        tifffile.imwrite(c2, make_c2())
        return [str(c1), str(c2)]

    def test_emits_a_result_per_pair_and_writes_csv(
        self, worker_factory, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        worker = worker_factory(
            file_pairs=[self._pair(tmp_path)],
            channel_names=["CH1", "CH2"],
            output_folder=str(out),
        )
        results = []
        worker.file_processed.connect(results.append)

        worker.run()

        assert len(results) == 1
        assert results[0]["filename"] == "s_c1.tif"

        csv_files = sorted(out.glob("*.csv"))
        assert csv_files, f"no csv written into {list(out.iterdir())}"
        with open(csv_files[0], newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) >= 3  # header plus one row per ROI

    def test_stop_prevents_processing(self, worker_factory, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        worker = worker_factory(
            file_pairs=[self._pair(tmp_path)],
            channel_names=["CH1", "CH2"],
            output_folder=str(out),
        )
        results = []
        worker.file_processed.connect(results.append)

        worker.stop()
        worker.run()

        assert results == []

    def test_unreadable_pair_reports_an_error(self, worker_factory, tmp_path):
        worker = worker_factory(
            file_pairs=[[str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]],
            channel_names=["CH1", "CH2"],
        )
        errors = []
        worker.error_occurred.connect(
            lambda path, msg: errors.append((path, msg))
        )
        results = []
        worker.file_processed.connect(results.append)

        worker.run()

        assert results == []
        assert errors


@requires_gui
class TestResultsWidget:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        return rc.ColocalizationResultsWidget(
            make_napari_viewer(), ["CH1", "CH2"]
        )

    def test_add_result_populates_a_row(self, widget):
        widget.add_result(
            {
                "filename": "sample_c1.tif",
                "common_substring": "sample",
                "csv_rows": [["sample_c1.tif", 1, 2], ["sample_c1.tif", 2, 0]],
            }
        )

        assert widget.table.rowCount() == 1
        assert widget.table.item(0, 0).text() == "sample"
        # one of the two ROIs has a non-zero channel-2 count
        assert widget.table.item(0, 1).text().strip() == "1"

    def test_result_without_rows_reports_zero(self, widget):
        widget.add_result({"filename": "sample_c1.tif", "csv_rows": []})
        assert widget.table.item(0, 1).text().strip() == "0"

    def test_identifier_falls_back_to_the_filename(self, widget):
        widget.add_result({"filename": "sample_c1.tif", "csv_rows": []})
        assert widget.table.item(0, 0).text() == "sample_c1.tif"

    def test_extract_identifier_uses_the_common_substring(self, widget):
        widget.add_result(
            {
                "filename": "sample_c1.tif",
                "common_substring": "sample",
                "csv_rows": [],
            }
        )
        assert widget._extract_identifier("sample_c1.tif") == "sample"

    def test_extract_identifier_of_unknown_file_strips_extension(self, widget):
        assert widget._extract_identifier("/a/b/other_c1.tif") == "other_c1"


@requires_gui
class TestAnalysisWidget:
    @pytest.fixture
    def widget(self, make_napari_viewer):
        return rc.ColocalizationAnalysisWidget(make_napari_viewer())

    def test_size_methods_are_mutually_exclusive(self, widget):
        widget.size_method_sum.setChecked(True)
        widget._update_size_method_checkboxes("sum", True)
        assert widget.size_method_median.isChecked() is False
        assert widget.size_method_individual.isChecked() is False

        widget.size_method_individual.setChecked(True)
        widget._update_size_method_checkboxes("individual", True)
        assert widget.size_method_sum.isChecked() is False

    def test_unchecking_does_not_clear_the_others(self, widget):
        widget.size_method_median.setChecked(True)
        widget._update_size_method_checkboxes("sum", False)
        assert widget.size_method_median.isChecked() is True

    def test_size_method_controls_follow_get_sizes(self, widget):
        widget.get_sizes_checkbox.setChecked(False)
        widget._update_size_method_controls_state()
        assert widget.size_method_median.isEnabled() is False

        widget.get_sizes_checkbox.setChecked(True)
        widget._update_size_method_controls_state()
        assert widget.size_method_median.isEnabled() is True

    def test_positive_counting_needs_labels_then_intensity(self, widget):
        widget.ch2_is_labels_checkbox.setChecked(True)
        widget.ch3_is_labels_checkbox.setChecked(False)
        widget.update_positive_counting_state()
        assert widget.count_positive_checkbox.isEnabled() is True

    @pytest.mark.parametrize(
        "ch2_labels, ch3_labels", [(True, True), (False, False), (False, True)]
    )
    def test_positive_counting_is_disabled_otherwise(
        self, widget, ch2_labels, ch3_labels
    ):
        widget.count_positive_checkbox.setChecked(True)
        widget.ch2_is_labels_checkbox.setChecked(ch2_labels)
        widget.ch3_is_labels_checkbox.setChecked(ch3_labels)
        widget.update_positive_counting_state()

        assert widget.count_positive_checkbox.isEnabled() is False
        assert widget.count_positive_checkbox.isChecked() is False

    def test_c2_positive_needs_a_third_channel_folder(self, widget, tmp_path):
        widget.ch2_is_labels_checkbox.setChecked(True)
        widget.ch3_is_labels_checkbox.setChecked(True)

        widget.ch3_folder.setText("")
        widget.update_c2_positive_state()
        assert widget.count_c2_positive_checkbox.isEnabled() is False

        widget.ch3_folder.setText(str(tmp_path))
        widget.update_c2_positive_state()
        assert widget.count_c2_positive_checkbox.isEnabled() is True

    def test_threshold_controls_follow_positive_counting(self, widget):
        widget.on_count_positive_changed(False)
        assert widget.threshold_value_input.isEnabled() is False

        widget.on_count_positive_changed(True)
        assert widget.threshold_value_input.isEnabled() is True

    def test_find_matching_files_pairs_two_folders(self, widget, tmp_path):
        ch1 = tmp_path / "ch1"
        ch2 = tmp_path / "ch2"
        ch1.mkdir()
        ch2.mkdir()
        for folder, suffix in ((ch1, "c1"), (ch2, "c2")):
            for sample in ("sample01", "sample02"):
                tifffile.imwrite(
                    folder / f"{sample}_{suffix}_labels.tif",
                    np.zeros((4, 4), np.uint16),
                )

        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        widget.ch3_folder.setText("")
        widget.find_matching_files()

        assert len(widget.file_pairs) == 2
        for pair in widget.file_pairs:
            assert len(pair) == 2
            assert os.path.basename(pair[0]).startswith(
                os.path.basename(pair[1]).split("_c2")[0]
            )

    def test_find_matching_files_requires_channel_one(self, widget, tmp_path):
        widget.ch1_folder.setText(str(tmp_path / "missing"))
        widget.ch2_folder.setText(str(tmp_path))
        widget.find_matching_files()

        assert "Channel 1" in widget.status_label.text()
        assert widget.file_pairs == []

    def test_find_matching_files_reports_empty_folder(self, widget, tmp_path):
        ch1 = tmp_path / "ch1"
        ch2 = tmp_path / "ch2"
        ch1.mkdir()
        ch2.mkdir()

        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        widget.find_matching_files()

        assert "No files matching" in widget.status_label.text()
        assert widget.analyze_button.isEnabled() is False


@requires_gui
def test_analyzer_factory_builds_a_gui(make_napari_viewer):
    make_napari_viewer()  # a Qt application has to exist first
    gui = rc.roi_colocalization_analyzer()
    assert callable(gui)
    assert "viewer" in gui.__signature__.parameters
