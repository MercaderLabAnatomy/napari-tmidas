"""Extra coverage for ROI colocalization (``_roi_colocalization``).

The existing suite pins the colocalization arithmetic on small hand-built
scenes.  This file goes after everything around it:

* the CSV header the worker emits for each combination of channel modes
  (labels/intensity x aggregate/individual x 2/3 channels),
* the per-C2-object "individual" row builder,
* the error paths that are meant to degrade rather than abort (CSV setup
  failure, CSV append failure, unreadable image on save),
* and the Qt shell: folder browsing, table population and click handling,
  the find/start/cancel/finish state machine and the napari factory.

No modal dialog is ever reached: ``QFileDialog`` is replaced on the module
object (the module imported the name, so patching ``qtpy`` would be too
late), and the analysis worker is replaced by a recording double so no
QThread is ever started.
"""

import csv
import os
import sys

import numpy as np
import pytest
import scipy.ndimage
import tifffile
from qtpy.QtGui import QCloseEvent

from napari_tmidas import _roi_colocalization as rc

pytest.importorskip("pytestqt")

requires_gui = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") == "true",
    reason="Qt widget tests cause segfaults on macOS CI (headless)",
)


# --------------------------------------------------------------------------
# A scene whose every number can be worked out by eye.
#
#   channel 1   label 1 = rows 0:4 (32 px), label 2 = rows 4:8 (32 px)
#   channel 2   label 10 at [0:2, 0:2], label 11 at [0:2, 4:6] (both in
#               ROI 1), label 12 at [5:7, 1:3] (in ROI 2) - 4 px each
#   channel 3   labels: 20 inside nucleus 10 (1 px), 21 inside nucleus 11
#               (2 px), nothing inside nucleus 12
#   channel 3   intensity: 10.0 under nucleus 10, [1, 3, 5, 7] under
#               nucleus 11, 0 everywhere else
# --------------------------------------------------------------------------


def make_c1():
    image = np.zeros((8, 8), dtype=np.uint16)
    image[0:4, :] = 1
    image[4:8, :] = 2
    return image


def make_c2():
    image = np.zeros((8, 8), dtype=np.uint16)
    image[0:2, 0:2] = 10
    image[0:2, 4:6] = 11
    image[5:7, 1:3] = 12
    return image


def make_c3_labels():
    image = np.zeros((8, 8), dtype=np.uint16)
    image[0, 0] = 20
    image[1, 4:6] = 21
    return image


def make_c3_intensity():
    image = np.zeros((8, 8), dtype=np.float32)
    image[0:2, 0:2] = 10.0
    image[0:2, 4:6] = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    return image


def make_c2_intensity():
    image = np.zeros((8, 8), dtype=np.float32)
    image[0:4, :] = 2.0
    image[4:8, :] = 6.0
    return image


@pytest.fixture
def worker_factory(qapp):
    def build(**kwargs):
        kwargs.setdefault("file_pairs", [])
        kwargs.setdefault("channel_names", ["A", "B", "C"])
        return rc.ColocalizationWorker(**kwargs)

    return build


def header_written_by(worker_factory, tmp_path, **kwargs):
    """Run a worker with no file pairs and return the CSV header it wrote."""
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    worker = worker_factory(output_folder=str(out), **kwargs)
    worker.run()
    csv_path = out / ("_".join(worker.channel_names) + "_colocalization.csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class FakeSignal:
    """The slice of ``Signal`` the widget actually uses."""

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class FakeWorker:
    """Stands in for ``ColocalizationWorker`` so no thread ever starts."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.started = False
        self.stopped = False
        self.terminated = False
        self.waits = []
        self.running = False
        self.wait_result = True
        self.progress_updated = FakeSignal()
        self.file_processed = FakeSignal()
        self.processing_finished = FakeSignal()
        self.error_occurred = FakeSignal()

    def start(self):
        self.started = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.stopped = True

    def wait(self, msecs=None):
        self.waits.append(msecs)
        return self.wait_result

    def terminate(self):
        self.terminated = True


class FakeWindow:
    def __init__(self):
        self.docked = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.docked.append((widget, name, area))
        return widget


class FakeViewer:
    """Only ``layers``, ``add_labels``, ``status`` and ``window`` are used."""

    def __init__(self):
        self.window = FakeWindow()
        self.layers = []
        self.status = ""
        self.added = []

    def add_labels(self, data, name=None):
        self.added.append((data, name))


def patch_file_dialog(monkeypatch, folder):
    """Replace ``QFileDialog`` on the module so nothing can block."""
    calls = []

    class Dialog:
        ShowDirsOnly = 1
        DontResolveSymlinks = 2

        @staticmethod
        def getExistingDirectory(*args, **kwargs):
            calls.append(args)
            return folder

    monkeypatch.setattr(rc, "QFileDialog", Dialog)
    return calls


@pytest.fixture
def viewer():
    return FakeViewer()


@pytest.fixture
def widget(qapp, viewer):
    return rc.ColocalizationAnalysisWidget(viewer)


@pytest.fixture
def results_widget(qapp, viewer):
    return rc.ColocalizationResultsWidget(viewer, ["A", "B"])


# --------------------------------------------------------------------------
# CSV header construction
# --------------------------------------------------------------------------


class TestCsvHeader:
    """The header is the contract for every downstream column."""

    def test_two_label_channels_aggregate(self, worker_factory, tmp_path):
        header = header_written_by(
            worker_factory, tmp_path, channel_names=["A", "B"]
        )
        assert header == ["Filename", "A_label_id", "B_in_A_count"]

    def test_channel2_as_intensity_replaces_the_count(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            channel_names=["A", "B"],
            channel2_is_labels=False,
        )
        assert header == [
            "Filename",
            "A_label_id",
            "B_in_A_mean",
            "B_in_A_median",
            "B_in_A_std",
            "B_in_A_max",
        ]

    def test_three_label_channels_with_sizes_and_positive_counting(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            get_sizes=True,
            count_c2_positive_for_c3=True,
        )
        assert header == [
            "Filename",
            "A_label_id",
            "B_in_A_count",
            "A_size",
            "B_in_A_size",
            "C_in_B_in_A_count",
            "C_not_in_B_but_in_A_count",
            "C_in_B_in_A_size",
            "C_not_in_B_but_in_A_size",
            "B_in_A_positive_for_C_count",
            "B_in_A_negative_for_C_count",
            "B_in_A_percent_positive_for_C",
        ]

    def test_individual_mode_adds_a_c2_label_column(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            get_sizes=True,
            size_method="individual",
        )
        # No aggregate "B_in_A_count": individual mode is one row per B
        # object, so the count would always be 1.
        assert header == [
            "Filename",
            "A_label_id",
            "B_label_id",
            "A_size",
            "B_size",
            "C_in_B_in_A_count",
            "C_not_in_B_but_in_A_count",
            "C_in_B_size",
        ]

    def test_individual_mode_with_intensity_third_channel(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            size_method="individual",
            channel3_is_labels=False,
        )
        assert header == [
            "Filename",
            "A_label_id",
            "B_label_id",
            "C_in_B_mean",
            "C_in_B_median",
            "C_in_B_std",
        ]

    def test_aggregate_intensity_third_channel_with_positive_counting(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            channel3_is_labels=False,
            count_positive=True,
        )
        assert header[:3] == ["Filename", "A_label_id", "B_in_A_count"]
        assert header[3:7] == [
            "C_in_B_in_A_mean",
            "C_in_B_in_A_median",
            "C_in_B_in_A_std",
            "C_in_B_in_A_max",
        ]
        assert header[7:11] == [
            "C_not_in_B_but_in_A_mean",
            "C_not_in_B_but_in_A_median",
            "C_not_in_B_but_in_A_std",
            "C_not_in_B_but_in_A_max",
        ]
        assert header[11:] == [
            "B_in_A_positive_for_C_count",
            "B_in_A_negative_for_C_count",
            "B_in_A_percent_positive_for_C",
            "C_threshold_used",
        ]

    def test_intensity_c2_with_label_c3_and_sizes(
        self, worker_factory, tmp_path
    ):
        header = header_written_by(
            worker_factory,
            tmp_path,
            channel2_is_labels=False,
            get_sizes=True,
        )
        assert header == [
            "Filename",
            "A_label_id",
            "B_in_A_mean",
            "B_in_A_median",
            "B_in_A_std",
            "B_in_A_max",
            "A_size",
            "C_in_A_count",
            "C_in_A_size",
        ]

    def test_both_extra_channels_as_intensity(self, worker_factory, tmp_path):
        header = header_written_by(
            worker_factory,
            tmp_path,
            channel2_is_labels=False,
            channel3_is_labels=False,
        )
        assert header[6:] == [
            "C_in_A_mean",
            "C_in_A_median",
            "C_in_A_std",
            "C_in_A_max",
        ]

    def test_an_existing_csv_is_replaced_not_appended_to(
        self, worker_factory, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        stale = out / "A_B_colocalization.csv"
        stale.write_text("stale,rows\n1,2\n3,4\n", encoding="utf-8")

        worker = worker_factory(
            channel_names=["A", "B"], output_folder=str(out)
        )
        worker.run()

        with open(stale, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows == [["Filename", "A_label_id", "B_in_A_count"]]

    def test_a_stale_csv_that_will_not_delete_is_reported(
        self, worker_factory, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        out.mkdir()
        stale = out / "A_B_colocalization.csv"
        stale.write_text("stale\n", encoding="utf-8")
        # A delete that reports success but leaves the file behind (a
        # network share holding an open handle, say).
        monkeypatch.setattr(rc.os, "remove", lambda path: None)

        worker = worker_factory(
            channel_names=["A", "B"], output_folder=str(out)
        )
        errors = []
        worker.error_occurred.connect(
            lambda path, msg: errors.append((path, msg))
        )

        worker.run()

        assert [path for path, _ in errors] == ["CSV file"]
        assert "Failed to remove existing CSV file" in errors[0][1]
        # the old contents are left untouched rather than half-rewritten
        assert stale.read_text(encoding="utf-8") == "stale\n"

    def test_csv_setup_failure_is_reported_and_does_not_raise(
        self, worker_factory, tmp_path
    ):
        # A plain file where the output folder should be: makedirs fails.
        blocked = tmp_path / "blocked"
        blocked.write_text("not a folder", encoding="utf-8")

        worker = worker_factory(
            channel_names=["A", "B"], output_folder=str(blocked)
        )
        errors = []
        worker.error_occurred.connect(
            lambda path, msg: errors.append((path, msg))
        )
        finished = []
        worker.processing_finished.connect(lambda: finished.append(True))

        worker.run()

        assert [path for path, _ in errors] == ["CSV file"]
        assert "Failed to set up CSV file" in errors[0][1]
        assert finished == [True]


class TestCsvAppendFailure:
    """A failed row append must not abort the run."""

    def test_row_write_failure_is_logged_and_processing_continues(
        self, worker_factory, tmp_path, monkeypatch, capsys
    ):
        c1 = tmp_path / "s_c1.tif"
        c2 = tmp_path / "s_c2.tif"
        tifffile.imwrite(c1, make_c1())
        tifffile.imwrite(c2, make_c2())
        out = tmp_path / "out"
        out.mkdir()

        real_writer = csv.writer

        class HeaderOnlyWriter:
            def __init__(self, handle, *args, **kwargs):
                self._real = real_writer(handle, *args, **kwargs)

            def writerow(self, row):
                self._real.writerow(row)

            def writerows(self, rows):
                raise OSError("no space left on device")

        monkeypatch.setattr(rc.csv, "writer", HeaderOnlyWriter)

        worker = worker_factory(
            file_pairs=[[str(c1), str(c2)]],
            channel_names=["A", "B"],
            output_folder=str(out),
            save_images=False,
        )
        results = []
        worker.file_processed.connect(results.append)

        worker.run()

        assert len(results) == 1
        assert "Error writing to CSV file" in capsys.readouterr().out


# --------------------------------------------------------------------------
# process_file_pair / save_output_image edge paths
# --------------------------------------------------------------------------


class TestProcessFilePairWarnings:
    def test_identical_second_and_third_channel_is_flagged(
        self, worker_factory, tmp_path, capsys
    ):
        c1 = tmp_path / "s_c1.tif"
        c2 = tmp_path / "s_c2.tif"
        c3 = tmp_path / "s_c3.tif"
        tifffile.imwrite(c1, make_c1())
        tifffile.imwrite(c2, make_c2())
        tifffile.imwrite(c3, make_c2())  # same data as channel 2

        worker = worker_factory(channel_names=["A", "B", "C"])
        worker.process_file_pair([str(c1), str(c2), str(c3)])

        printed = capsys.readouterr().out
        assert "IDENTICAL" in printed
        assert "s_c2.tif" in printed and "s_c3.tif" in printed


class TestSaveOutputImageFailure:
    def test_unreadable_channel_one_is_reported_not_raised(
        self, worker_factory, tmp_path, capsys
    ):
        out = tmp_path / "out"
        out.mkdir()
        worker = worker_factory(
            channel_names=["A", "B"], output_folder=str(out)
        )
        results = {"results": []}

        worker.save_output_image(results, [str(tmp_path / "missing.tif")])

        assert "output_path" not in results
        assert "Error saving output image" in capsys.readouterr().out
        assert list(out.iterdir()) == []


# --------------------------------------------------------------------------
# Individual (one row per channel-2 object) mode
# --------------------------------------------------------------------------


class TestIndividualMode:
    """One row per C2 object, with that object's own measurements."""

    def _worker(self, worker_factory, **kwargs):
        kwargs.setdefault("channel_names", ["A", "B", "C"])
        kwargs.setdefault("size_method", "individual")
        return worker_factory(**kwargs)

    def test_two_channels_with_sizes(self, worker_factory):
        worker = self._worker(worker_factory, get_sizes=True)
        output = worker.process_colocalization("s.tif", make_c1(), make_c2())

        assert output["csv_rows"] == [
            ["s.tif", 1, 10, 32, 4],
            ["s.tif", 1, 11, 32, 4],
            ["s.tif", 2, 12, 32, 4],
        ]
        # Individual mode does not build the per-ROI visualisation dicts.
        assert output["results"] == []

    def test_third_label_channel_counts_and_sizes_per_object(
        self, worker_factory
    ):
        worker = self._worker(worker_factory, get_sizes=True)
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2(), make_c3_labels()
        )

        # nucleus 10 holds one 1-px spot, nucleus 11 one 2-px spot,
        # nucleus 12 none.
        assert output["csv_rows"] == [
            ["s.tif", 1, 10, 32, 4, 1, 1],
            ["s.tif", 1, 11, 32, 4, 1, 2],
            ["s.tif", 2, 12, 32, 4, 0, 0],
        ]

    def test_third_intensity_channel_gives_per_object_statistics(
        self, worker_factory
    ):
        worker = self._worker(worker_factory, channel3_is_labels=False)
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2(), make_c3_intensity()
        )

        rows = output["csv_rows"]
        assert [row[:3] for row in rows] == [
            ["s.tif", 1, 10],
            ["s.tif", 1, 11],
            ["s.tif", 2, 12],
        ]
        # nucleus 10 sits on a constant 10.0 patch
        assert rows[0][3:] == pytest.approx([10.0, 10.0, 0.0])
        # nucleus 11 sits on 1, 3, 5, 7
        assert rows[1][3:] == pytest.approx([4.0, 4.0, 5.0**0.5])
        # nucleus 12 sits on background
        assert rows[2][3:] == pytest.approx([0.0, 0.0, 0.0])

    def test_roi_without_any_c2_object_contributes_no_row(
        self, worker_factory
    ):
        worker = self._worker(worker_factory, get_sizes=True)
        empty_c2 = np.zeros((8, 8), dtype=np.uint16)
        output = worker.process_colocalization("s.tif", make_c1(), empty_c2)
        assert output["csv_rows"] == []


# --------------------------------------------------------------------------
# Aggregate mode: the remaining channel-mode combinations
# --------------------------------------------------------------------------


class TestAggregateChannelModes:
    def test_label_c2_with_intensity_c3_and_positive_counting(
        self, worker_factory
    ):
        worker = worker_factory(
            channel3_is_labels=False,
            count_positive=True,
            threshold_method="absolute",
            threshold_value=5.0,
        )
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2(), make_c3_intensity()
        )

        row = output["csv_rows"][0]
        assert row[:3] == ["s.tif", 1, 2]
        # C3 under the two nuclei: 10, 10, 10, 10, 1, 3, 5, 7
        assert row[3] == pytest.approx(7.0)
        assert row[4] == pytest.approx(8.5)
        assert row[5] == pytest.approx(11.5**0.5)
        assert row[6] == pytest.approx(10.0)
        # ...and nothing at all outside them
        assert row[7:11] == pytest.approx([0.0, 0.0, 0.0, 0.0])
        # nucleus 10 averages 10.0 (positive), nucleus 11 averages 4.0
        assert row[11:] == pytest.approx([1, 1, 50.0, 5.0])

        first = output["results"][0]
        assert first["label_id"] == 1
        assert first["ch2_in_ch1_count"] == 2
        assert first["ch3_in_ch2_in_ch1_mean"] == pytest.approx(7.0)
        assert first["ch3_in_ch2_in_ch1_max"] == pytest.approx(10.0)
        assert first["ch2_in_ch1_positive_for_ch3_count"] == 1
        assert first["ch2_in_ch1_negative_for_ch3_count"] == 1
        assert first["ch2_in_ch1_percent_positive_for_ch3"] == pytest.approx(
            50.0
        )
        assert first["ch3_threshold_used"] == pytest.approx(5.0)

    def test_intensity_c2_with_label_c3_and_sizes(self, worker_factory):
        worker = worker_factory(channel2_is_labels=False, get_sizes=True)
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2_intensity(), make_c2()
        )

        rows = output["csv_rows"]
        # ROI 1 sits on a constant 2.0, ROI 2 on a constant 6.0
        assert rows[0][:2] == ["s.tif", 1]
        assert rows[0][2:6] == pytest.approx([2.0, 2.0, 0.0, 2.0])
        assert rows[0][6] == 32  # ROI size
        assert rows[0][7] == 2  # two C3 objects inside ROI 1
        assert rows[0][8] == 8  # eight C3 pixels inside ROI 1
        assert rows[1][2:6] == pytest.approx([6.0, 6.0, 0.0, 6.0])
        assert rows[1][7:] == [1, 4]

        # channel2_is_labels=False means row[2:6] holds ch2's own intensity
        # stats rather than a single count, so the result dict must carry
        # those four values rather than the labels-mode "count" key.
        assert output["results"][0] == {
            "label_id": 1,
            "ch2_in_ch1_mean": pytest.approx(2.0),
            "ch2_in_ch1_median": pytest.approx(2.0),
            "ch2_in_ch1_std": pytest.approx(0.0),
            "ch2_in_ch1_max": pytest.approx(2.0),
            "ch1_size": 32,
            "ch3_in_ch1_count": 2,
            "ch3_in_ch1_size": 8,
        }

    def test_both_extra_channels_as_intensity(self, worker_factory):
        worker = worker_factory(
            channel2_is_labels=False, channel3_is_labels=False
        )
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2_intensity(), make_c3_intensity()
        )

        row = output["csv_rows"][0]
        assert row[2:6] == pytest.approx([2.0, 2.0, 0.0, 2.0])
        # C3 over the whole 32-px ROI 1: four 10s, then 1, 3, 5, 7
        assert row[6] == pytest.approx(56.0 / 32)
        assert row[7] == pytest.approx(0.0)
        assert row[9] == pytest.approx(10.0)

        # Both channels are intensity, so the result dict must carry all
        # eight stats -- ch2's own four plus ch3's -- at their real values,
        # not the labels-mode "count" key the buggy offset used to leave
        # every one of these columns shifted under.
        assert output["results"][0] == {
            "label_id": 1,
            "ch2_in_ch1_mean": pytest.approx(2.0),
            "ch2_in_ch1_median": pytest.approx(2.0),
            "ch2_in_ch1_std": pytest.approx(0.0),
            "ch2_in_ch1_max": pytest.approx(2.0),
            "ch3_in_ch1_mean": pytest.approx(1.75),
            "ch3_in_ch1_median": pytest.approx(0.0),
            "ch3_in_ch1_std": pytest.approx(3.473111),
            "ch3_in_ch1_max": pytest.approx(10.0),
        }

    def test_labels_missing_a_bounding_box_are_skipped(
        self, worker_factory, monkeypatch
    ):
        """Defensive branch: find_objects reporting no box for a label."""
        monkeypatch.setattr(
            scipy.ndimage,
            "find_objects",
            lambda image: [None] * int(image.max()),
        )
        worker = worker_factory(channel_names=["A", "B"])
        output = worker.process_colocalization("s.tif", make_c1(), make_c2())

        assert output["csv_rows"] == []
        assert output["results"] == []

    def test_semantic_conversion_of_the_third_channel(self, worker_factory):
        worker = worker_factory(convert_to_instances_c3=True)
        semantic_c3 = np.zeros((8, 8), dtype=np.uint16)
        semantic_c3[0, 0] = 5  # inside nucleus 10
        semantic_c3[1, 4] = 5  # inside nucleus 11, disconnected
        output = worker.process_colocalization(
            "s.tif", make_c1(), make_c2(), semantic_c3
        )

        # Both blobs carried label 5; after conversion they are two objects,
        # so ROI 1 reports two C3 objects inside C2 rather than one.
        assert output["csv_rows"][0][3] == 2


# --------------------------------------------------------------------------
# Results widget
# --------------------------------------------------------------------------


@requires_gui
class TestResultsWidgetTable:
    def _tif(self, tmp_path):
        path = tmp_path / "vis.tif"
        tifffile.imwrite(
            path,
            np.zeros((3, 8, 8), dtype=np.uint32),
            photometric="minisblack",
        )
        return str(path)

    def test_output_path_is_stored_on_every_cell(
        self, results_widget, tmp_path
    ):
        path = self._tif(tmp_path)
        results_widget.add_result(
            {"filename": "s.tif", "csv_rows": [], "output_path": path}
        )

        for column in range(2):
            item = results_widget.table.item(0, column)
            assert item.data(rc.Qt.UserRole + 1) == path

    def test_click_on_an_empty_table_does_nothing(self, results_widget):
        results_widget.on_table_clicked(0, 0)
        assert results_widget.viewer.status == ""

    def test_click_on_an_unknown_filename_does_nothing(self, results_widget):
        results_widget.table.insertRow(0)
        item = rc.QTableWidgetItem("orphan")
        item.setData(rc.Qt.UserRole, "never_seen.tif")
        results_widget.table.setItem(0, 0, item)

        results_widget.on_table_clicked(0, 0)
        assert results_widget.viewer.status == ""

    def test_click_without_a_visualisation_says_so(self, results_widget):
        results_widget.add_result({"filename": "s.tif", "csv_rows": []})
        results_widget.on_table_clicked(0, 0)
        assert (
            results_widget.viewer.status
            == "No visualization available for this result"
        )

    def test_click_with_a_missing_file_says_so(self, results_widget, tmp_path):
        results_widget.add_result(
            {
                "filename": "s.tif",
                "csv_rows": [],
                "output_path": str(tmp_path / "gone.tif"),
            }
        )
        results_widget.on_table_clicked(0, 0)
        assert (
            results_widget.viewer.status
            == "No visualization available for this result"
        )

    def test_click_loads_the_visualisation_into_the_viewer(
        self, results_widget, tmp_path
    ):
        path = self._tif(tmp_path)
        results_widget.viewer.layers.append("stale layer")
        results_widget.add_result(
            {"filename": "s.tif", "csv_rows": [], "output_path": path}
        )

        results_widget.on_table_clicked(0, 0)

        assert results_widget.viewer.layers == []
        assert len(results_widget.viewer.added) == 1
        data, name = results_widget.viewer.added[0]
        assert data.shape == (3, 8, 8)
        assert name == "Colocalization: vis.tif"
        assert results_widget.viewer.status == "Loaded visualization for s.tif"

    def test_unreadable_visualisation_reports_an_error(
        self, results_widget, tmp_path
    ):
        broken = tmp_path / "broken.tif"
        broken.write_bytes(b"not a tiff at all")
        results_widget.add_result(
            {
                "filename": "s.tif",
                "csv_rows": [],
                "output_path": str(broken),
            }
        )

        results_widget.on_table_clicked(0, 0)

        assert results_widget.viewer.status.startswith(
            "Error loading visualization"
        )
        assert results_widget.viewer.added == []


# --------------------------------------------------------------------------
# Analysis widget: construction and folder browsing
# --------------------------------------------------------------------------


@requires_gui
class TestAnalysisWidgetConstruction:
    def test_folders_and_patterns_are_prefilled(self, qapp, viewer):
        built = rc.ColocalizationAnalysisWidget(
            viewer,
            channel_folders=["/a", "/b", "/c"],
            channel_patterns=["*1.tif", "*2.tif", "*3.tif"],
        )

        assert built.ch1_folder.text() == "/a"
        assert built.ch2_folder.text() == "/b"
        assert built.ch3_folder.text() == "/c"
        assert built.ch1_pattern.text() == "*1.tif"
        assert built.ch2_pattern.text() == "*2.tif"
        assert built.ch3_pattern.text() == "*3.tif"
        assert built.channel_names == ["CH1", "CH2", "CH3"]

    def test_two_folders_leave_the_third_empty(self, qapp, viewer):
        built = rc.ColocalizationAnalysisWidget(
            viewer, channel_folders=["/a", "/b"], channel_patterns=["*1.tif"]
        )
        assert built.ch3_folder.text() == ""
        assert built.ch2_pattern.text() == ""
        assert built.channel_names == ["CH1", "CH2"]

    def test_median_selection_clears_the_other_methods(self, widget):
        widget.size_method_sum.setChecked(True)
        widget.size_method_individual.setChecked(True)

        widget._update_size_method_checkboxes("median", True)

        assert widget.size_method_sum.isChecked() is False
        assert widget.size_method_individual.isChecked() is False


@requires_gui
class TestBrowsing:
    def test_browse_fills_each_channel_field(
        self, widget, tmp_path, monkeypatch
    ):
        patch_file_dialog(monkeypatch, str(tmp_path))

        widget.browse_folder(0)
        widget.browse_folder(1)
        assert widget.ch1_folder.text() == str(tmp_path)
        assert widget.ch2_folder.text() == str(tmp_path)

    def test_browsing_channel_three_enables_its_controls(
        self, widget, tmp_path, monkeypatch
    ):
        patch_file_dialog(monkeypatch, str(tmp_path))
        assert widget.ch3_is_labels_checkbox.isEnabled() is False

        widget.browse_folder(2)

        assert widget.ch3_folder.text() == str(tmp_path)
        assert widget.ch3_is_labels_checkbox.isEnabled() is True
        assert widget.convert_c3_checkbox.isEnabled() is True

    def test_cancelled_dialog_leaves_the_field_alone(
        self, widget, monkeypatch
    ):
        widget.ch1_folder.setText("/keep/me")
        patch_file_dialog(monkeypatch, "")

        widget.browse_folder(0)

        assert widget.ch1_folder.text() == "/keep/me"

    def test_browse_output_fills_the_output_field(
        self, widget, tmp_path, monkeypatch
    ):
        calls = patch_file_dialog(monkeypatch, str(tmp_path))

        widget.browse_output()

        assert widget.output_folder.text() == str(tmp_path)
        assert calls[0][1] == "Select Output Folder"

    def test_cancelled_output_dialog_leaves_the_field_alone(
        self, widget, monkeypatch
    ):
        # Pre-fill with a sentinel: the field starts empty by default, so
        # without this the assertion below would hold even if the ``if
        # folder:`` guard in browse_output() were deleted and it called
        # setText("") unconditionally.
        widget.output_folder.setText("/keep/me")
        patch_file_dialog(monkeypatch, "")

        widget.browse_output()

        assert widget.output_folder.text() == "/keep/me"


# --------------------------------------------------------------------------
# Analysis widget: find_matching_files guards
# --------------------------------------------------------------------------


@requires_gui
class TestFindMatchingFiles:
    def _folder(self, tmp_path, name, filenames):
        folder = tmp_path / name
        folder.mkdir()
        for filename in filenames:
            (folder / filename).write_bytes(b"")
        return folder

    def test_second_channel_folder_must_exist(self, widget, tmp_path):
        ch1 = self._folder(tmp_path, "ch1", ["a_labels.tif"])
        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(tmp_path / "nope"))

        widget.find_matching_files()

        assert (
            widget.status_label.text()
            == "Channel 2 folder is required and must exist"
        )

    def test_empty_second_channel_folder_is_reported(self, widget, tmp_path):
        ch1 = self._folder(tmp_path, "ch1", ["a_labels.tif"])
        ch2 = self._folder(tmp_path, "ch2", [])
        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        widget.analyze_button.setEnabled(True)

        widget.find_matching_files()

        assert "Channel 2 folder" in widget.status_label.text()
        assert widget.match_label.text() == "No matching files found"
        assert widget.analyze_button.isEnabled() is False

    def test_empty_third_channel_folder_is_reported(self, widget, tmp_path):
        ch1 = self._folder(tmp_path, "ch1", ["a_labels.tif"])
        ch2 = self._folder(tmp_path, "ch2", ["a_labels.tif"])
        ch3 = self._folder(tmp_path, "ch3", ["a.txt"])
        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        widget.ch3_folder.setText(str(ch3))
        # Start enabled so disabling it below proves find_matching_files()
        # actually flips it, rather than coinciding with its default state.
        widget.analyze_button.setEnabled(True)

        widget.find_matching_files()

        assert "Channel 3 folder" in widget.status_label.text()
        assert widget.analyze_button.isEnabled() is False

    def test_three_channels_are_grouped_into_triplets(self, widget, tmp_path):
        ch1 = self._folder(
            tmp_path, "ch1", ["sampleA_labels.tif", "sampleB_labels.tif"]
        )
        ch2 = self._folder(
            tmp_path, "ch2", ["sampleA_labels.tif", "sampleB_labels.tif"]
        )
        ch3 = self._folder(
            tmp_path, "ch3", ["sampleA_labels.tif", "sampleB_labels.tif"]
        )
        for field, folder in (
            (widget.ch1_folder, ch1),
            (widget.ch2_folder, ch2),
            (widget.ch3_folder, ch3),
        ):
            field.setText(str(folder))

        widget.find_matching_files()

        assert len(widget.file_pairs) == 2
        assert all(len(pair) == 3 for pair in widget.file_pairs)
        assert (
            widget.match_label.text()
            == "Found 2 matching file sets across 3 channels"
        )
        assert widget.status_label.text() == "Ready to analyze"
        assert widget.analyze_button.isEnabled() is True
        stored = {
            entry["common_substring"] for entry in widget.file_results.values()
        }
        assert stored == {"sampleA", "sampleB"}

    def test_names_with_nothing_in_common_produce_no_groups(
        self, widget, tmp_path
    ):
        ch1 = self._folder(tmp_path, "ch1", ["aaa"])
        ch2 = self._folder(tmp_path, "ch2", ["bbb"])
        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        widget.ch1_pattern.setText("*")
        widget.ch2_pattern.setText("*")
        widget.analyze_button.setEnabled(True)

        widget.find_matching_files()

        assert widget.file_pairs == []
        assert widget.match_label.text() == (
            "No matching files found across channels"
        )
        assert widget.status_label.text() == "No files to analyze"
        assert widget.analyze_button.isEnabled() is False


# --------------------------------------------------------------------------
# Analysis widget: start / progress / finish / cancel
# --------------------------------------------------------------------------


@requires_gui
class TestStartAnalysis:
    def _ready(self, widget, tmp_path, channels=2):
        ch1 = tmp_path / "chan1"
        ch2 = tmp_path / "chan2"
        ch3 = tmp_path / "chan3"
        for folder in (ch1, ch2, ch3):
            folder.mkdir(exist_ok=True)
        widget.ch1_folder.setText(str(ch1))
        widget.ch2_folder.setText(str(ch2))
        files = [str(ch1 / "s.tif"), str(ch2 / "s.tif")]
        if channels == 3:
            widget.ch3_folder.setText(str(ch3))
            files.append(str(ch3 / "s.tif"))
        widget.file_pairs = [tuple(files)]
        return widget

    def _patch_worker(self, monkeypatch):
        made = []

        def factory(*args, **kwargs):
            worker = FakeWorker(*args, **kwargs)
            made.append(worker)
            return worker

        monkeypatch.setattr(rc, "ColocalizationWorker", factory)
        return made

    def test_without_file_pairs_nothing_starts(self, widget, monkeypatch):
        made = self._patch_worker(monkeypatch)
        widget.start_analysis()

        assert made == []
        assert widget.status_label.text() == "No file pairs to analyze"

    def test_worker_is_configured_from_the_widget(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        out = tmp_path / "out"
        widget.output_folder.setText(str(out))
        widget.get_sizes_checkbox.setChecked(True)
        widget.save_images_checkbox.setChecked(True)
        widget.thread_count.setValue(1)
        made = self._patch_worker(monkeypatch)

        widget.start_analysis()

        assert len(made) == 1
        worker = made[0]
        args = worker.init_args
        assert args[0] == widget.file_pairs
        # channel names come from the folder basenames
        assert args[1] == ["chan1", "chan2"]
        assert args[2] is True  # get_sizes
        assert args[3] == "median"  # size_method
        assert args[4] == str(out)
        assert args[10] is True  # save_images
        assert worker.thread_count == 1
        assert worker.started is True
        assert out.is_dir()

        # the parent is never shown in tests, so ask about the flag
        assert widget.progress_bar.isHidden() is False
        assert widget.analyze_button.isEnabled() is False
        assert widget.cancel_button.isEnabled() is True
        assert widget.status_label.text() == (
            "Processing 1 file pairs with 1 threads"
        )
        assert widget.results_widget is not None
        assert viewer_docked_names(widget) == ["Colocalization Results"]

    def test_third_channel_name_is_added(self, widget, tmp_path, monkeypatch):
        self._ready(widget, tmp_path, channels=3)
        made = self._patch_worker(monkeypatch)

        widget.start_analysis()

        assert made[0].init_args[1] == ["chan1", "chan2", "chan3"]

    def test_sum_and_individual_size_methods_are_passed_through(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        made = self._patch_worker(monkeypatch)

        widget.size_method_median.setChecked(False)
        widget.size_method_sum.setChecked(True)
        widget.start_analysis()
        assert made[0].init_args[3] == "sum"

        widget.size_method_sum.setChecked(False)
        widget.size_method_individual.setChecked(True)
        widget.start_analysis()
        assert made[1].init_args[3] == "individual"

    def test_no_size_method_selected_falls_back_to_median(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        made = self._patch_worker(monkeypatch)
        widget.size_method_median.setChecked(False)
        widget.size_method_sum.setChecked(False)
        widget.size_method_individual.setChecked(False)

        widget.start_analysis()

        assert made[0].init_args[3] == "median"

    def test_unparseable_threshold_resets_to_the_default(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        made = self._patch_worker(monkeypatch)
        widget.threshold_value_input.setText("not a number")

        widget.start_analysis()

        assert made[0].init_args[9] == 75.0
        assert widget.threshold_value_input.text() == "75.0"

    def test_absolute_threshold_method_is_passed_through(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        made = self._patch_worker(monkeypatch)
        widget.threshold_percentile.setChecked(False)

        widget.start_analysis()

        assert made[0].init_args[8] == "absolute"

    def test_an_output_folder_that_cannot_be_created_stops_the_run(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        blocked = tmp_path / "blocked"
        blocked.write_text("a file, not a folder", encoding="utf-8")
        widget.output_folder.setText(str(blocked))
        made = self._patch_worker(monkeypatch)

        widget.start_analysis()

        assert made == []
        assert widget.status_label.text().startswith(
            "Error creating output folder"
        )

    def test_an_unwritable_output_folder_stops_the_run(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        out = tmp_path / "out"
        widget.output_folder.setText(str(out))
        made = self._patch_worker(monkeypatch)

        def refuse(path):
            raise PermissionError("read-only file system")

        monkeypatch.setattr(rc.os, "remove", refuse)

        widget.start_analysis()

        assert made == []
        assert widget.status_label.text().startswith(
            "Cannot write to output folder"
        )

    def test_an_existing_results_widget_is_reused(
        self, widget, tmp_path, monkeypatch
    ):
        self._ready(widget, tmp_path)
        self._patch_worker(monkeypatch)
        widget.start_analysis()
        first = widget.results_widget

        widget.start_analysis()

        assert widget.results_widget is first
        assert viewer_docked_names(widget) == ["Colocalization Results"]


def viewer_docked_names(widget):
    return [name for _, name, _ in widget.viewer.window.docked]


@requires_gui
class TestProgressAndCompletion:
    def test_update_progress_moves_the_bar(self, widget):
        widget.update_progress(42)
        assert widget.progress_bar.value() == 42

    def test_file_processed_without_a_results_widget_is_a_no_op(self, widget):
        widget.file_processed({"filename": "s.tif", "csv_rows": []})
        assert widget.results_widget is None

    def test_file_processed_forwards_to_the_results_widget(
        self, widget, viewer
    ):
        widget.results_widget = rc.ColocalizationResultsWidget(
            viewer, ["A", "B"]
        )
        widget.file_processed(
            {
                "filename": "s.tif",
                "common_substring": "s",
                "csv_rows": [["s.tif", 1, 3]],
            }
        )
        assert widget.results_widget.table.rowCount() == 1
        assert widget.results_widget.table.item(0, 1).text().strip() == "1"

    def test_processing_finished_restores_the_buttons(self, widget):
        widget.analyze_button.setEnabled(False)
        widget.cancel_button.setEnabled(True)
        widget.worker = FakeWorker()

        widget.processing_finished()

        assert widget.progress_bar.value() == 100
        assert widget.analyze_button.isEnabled() is True
        assert widget.cancel_button.isEnabled() is False
        assert widget.worker is None
        assert widget.status_label.text() == "Analysis complete"

    def test_processing_finished_stops_a_still_running_worker(self, widget):
        worker = FakeWorker()
        worker.running = True
        widget.worker = worker

        widget.processing_finished()

        assert worker.stopped is True
        assert worker.waits == [None]
        assert widget.worker is None

    def test_processing_finished_without_a_worker(self, widget):
        widget.worker = None
        widget.processing_finished()
        assert widget.status_label.text() == "Analysis complete"

    def test_processing_error_is_surfaced(self, widget, capsys):
        widget.processing_error("/some/file.tif", "boom")

        assert widget.status_label.text() == "Error: boom"
        assert (
            "Error processing /some/file.tif: boom" in capsys.readouterr().out
        )


@requires_gui
class TestCancelAnalysis:
    def test_cancelling_without_a_worker_does_nothing(self, widget):
        widget.worker = None
        widget.cancel_analysis()
        assert widget.status_label.text() == ""

    def test_cancelling_an_idle_worker_does_nothing(self, widget):
        worker = FakeWorker()
        widget.worker = worker

        widget.cancel_analysis()

        assert worker.stopped is False
        assert widget.worker is worker

    def test_cancelling_a_running_worker_stops_it(self, widget):
        worker = FakeWorker()
        worker.running = True
        widget.worker = worker
        widget.analyze_button.setEnabled(False)
        widget.cancel_button.setEnabled(True)

        widget.cancel_analysis()

        assert worker.stopped is True
        assert worker.terminated is False
        assert widget.worker is None
        assert widget.analyze_button.isEnabled() is True
        assert widget.cancel_button.isEnabled() is False
        assert widget.status_label.text() == "Analysis cancelled"

    def test_an_unresponsive_worker_is_terminated(self, widget):
        worker = FakeWorker()
        worker.running = True
        worker.wait_result = False
        widget.worker = worker

        widget.cancel_analysis()

        assert worker.terminated is True
        assert worker.waits == [1000, None]


# --------------------------------------------------------------------------
# The napari factory function
# --------------------------------------------------------------------------


@requires_gui
class TestAnalyzerFactory:
    def _run(self, viewer):
        function = rc.roi_colocalization_analyzer.keywords["function"]
        return function(viewer)

    def test_docks_the_analysis_widget(self, qapp, viewer):
        built = self._run(viewer)

        assert isinstance(built, rc.ColocalizationAnalysisWidget)
        widget, name, area = viewer.window.docked[0]
        assert widget is built
        assert name == "ROI Colocalization Analysis"
        assert area == "right"

    def test_closing_stops_a_running_worker(self, qapp, viewer):
        built = self._run(viewer)
        worker = FakeWorker()
        worker.running = True
        worker.wait_result = False
        built.worker = worker

        built.closeEvent(QCloseEvent())

        assert worker.stopped is True
        assert worker.terminated is True
        assert built.worker is None

    def test_closing_without_a_worker_is_harmless(self, qapp, viewer):
        built = self._run(viewer)
        built.worker = None

        built.closeEvent(QCloseEvent())

        assert built.worker is None
