"""
Trackers save their lineage next to the tracked labels.

The Trackastra test runs the real generated script in a subprocess against a
small stand-in ``trackastra`` package (and a ``torch`` stub), so the code
that ends up in the tracker's environment is what gets exercised.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_tmidas.processing_functions.track_qc import (
    ctc_res_track_to_lineage,
    lineage_csv_path,
    load_tracks_for_labels,
    read_lineage,
    track_statistics,
    track_statistics_qc,
    tracks_from_labels,
)

_STUB_TORCH = """
class cuda:
    OutOfMemoryError = MemoryError

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        pass
"""

# A graph like Trackastra's: one node per object per frame, edges forward in
# time. Object 1 moves for frames 0-1 and divides into objects 2 and 3.
_STUB_MODEL = """
import networkx as nx


class Trackastra:
    @classmethod
    def from_pretrained(cls, name, device=None):
        return cls()

    def track(self, img, mask, mode=None, **kwargs):
        g = nx.DiGraph()
        for t, label in [(0, 1), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
            g.add_node((t, label), time=t, label=label)
        g.add_edge((0, 1), (1, 1))
        g.add_edge((1, 1), (2, 2))
        g.add_edge((1, 1), (2, 3))
        g.add_edge((2, 2), (3, 2))
        g.add_edge((2, 3), (3, 3))
        return g
"""

# Tracklets as CTC defines them: chains cut where a node has more than one
# successor. Orderable, with a ``nodes`` attribute, like the real ones.
_STUB_TRACKING = """
from dataclasses import dataclass, field


@dataclass(order=True)
class _Tracklet:
    start: tuple
    nodes: list = field(compare=False)


def ctc_tracklets(G, frame_attribute="time"):
    starts = [
        n for n in G.nodes
        if G.in_degree(n) != 1 or G.out_degree(next(iter(G.predecessors(n)))) > 1
    ]
    tracklets = []
    for s in starts:
        nodes = [s]
        while G.out_degree(nodes[-1]) == 1:
            nodes.append(next(iter(G.successors(nodes[-1]))))
        tracklets.append(_Tracklet((G.nodes[s][frame_attribute], s), nodes))
    return tracklets


def graph_to_ctc(*args, **kwargs):
    raise AssertionError("the streaming path should be used")
"""


@pytest.fixture
def stub_env(tmp_path):
    root = tmp_path / "stubs"
    (root / "trackastra" / "tracking").mkdir(parents=True)
    (root / "torch.py").write_text(_STUB_TORCH)
    (root / "trackastra" / "__init__.py").write_text("")
    (root / "trackastra" / "model.py").write_text(_STUB_MODEL)
    (root / "trackastra" / "tracking" / "__init__.py").write_text(_STUB_TRACKING)
    (root / "trackastra" / "tracking" / "utils.py").write_text(
        "from trackastra.tracking import ctc_tracklets\n"
    )
    return root


def test_trackastra_script_saves_lineage_matching_the_labels(tmp_path, stub_env):
    pytest.importorskip("networkx")
    from napari_tmidas.processing_functions import trackastra_tracking

    mask = np.zeros((4, 16, 16), dtype=np.uint16)
    mask[0:2, 2:5, 2:5] = 1
    mask[2:4, 2:5, 2:5] = 2
    mask[2:4, 10:13, 10:13] = 3
    img = (mask > 0).astype(np.uint16) * 100
    img_path, mask_path = tmp_path / "a.tif", tmp_path / "a_labels.tif"
    for path, data in ((img_path, img), (mask_path, mask)):
        tifffile.imwrite(str(path), data, ome=True, metadata={"axes": "TYX"})
    out_path = tmp_path / "a_tracked.tif"

    script = trackastra_tracking.create_trackastra_script(
        str(img_path), str(mask_path), "general_2d", "greedy", str(out_path)
    )
    script_path = tmp_path / "run.py"
    script_path.write_text(script)
    env = {**os.environ, "PYTHONPATH": str(stub_env)}
    run = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stdout + run.stderr

    lineage = pd.read_csv(lineage_csv_path(str(out_path)))
    assert list(lineage.columns) == [
        "track_id",
        "start",
        "end",
        "parent_track_id",
    ]
    rows = {r.track_id: (r.start, r.end, r.parent_track_id) for r in lineage.itertuples()}
    assert rows == {1: (0, 1, 0), 2: (2, 3, 1), 3: (2, 3, 1)}

    # The ids in the lineage are the ids in the written labels.
    labels = tifffile.imread(str(out_path))
    assert set(np.unique(labels[0])) == {0, 1}
    assert set(np.unique(labels[3])) == {0, 2, 3}

    table, n = tracks_from_labels(str(out_path))
    stats = track_statistics(table, n, lineage=read_lineage(str(out_path)))
    assert stats["divisions"] == 1
    assert stats["tracks_appearing"] == 0


def test_ctc_res_track_conversion(tmp_path):
    res = tmp_path / "res_track.txt"
    res.write_text("1 0 4 0\n2 5 9 1\n3 5 9 1\n4 7 9 0\n")
    labels = tmp_path / "a_hoct_tracked.tif"

    written = ctc_res_track_to_lineage(str(res), str(labels))

    assert written == str(tmp_path / "a_hoct_tracked_lineage.csv")
    assert read_lineage(str(labels)) == {2: 1, 3: 1}


def test_ctc_res_track_conversion_never_raises(tmp_path):
    bad = tmp_path / "res_track.txt"
    bad.write_text("not numbers\n")
    assert ctc_res_track_to_lineage(str(bad), str(tmp_path / "x.tif")) is None
    assert ctc_res_track_to_lineage(str(tmp_path / "missing.txt"), "x.tif") is None


def test_lineage_reaches_statistics_and_tracks_layer(tmp_path):
    movie = np.zeros((4, 16, 16), np.uint32)
    movie[0:2, 2:5, 2:5] = 1
    movie[2:4, 2:5, 2:5] = 2
    movie[2:4, 10:13, 10:13] = 3
    movie[3, 14, 14] = 4  # appears late without a parent
    path = str(tmp_path / "m_tracked.tif")
    tifffile.imwrite(path, movie, ome=True, metadata={"axes": "TYX"})
    pd.DataFrame(
        {"track_id": [1, 2, 3, 4], "start": [0, 2, 2, 3], "end": [1, 3, 3, 3],
         "parent_track_id": [0, 1, 1, 0]}
    ).to_csv(lineage_csv_path(path), index=False)

    stats = track_statistics_qc(None, _source_filepath=path)
    assert stats["divisions"] == 1
    assert stats["tracks_appearing"] == 1  # track 4; 2 and 3 are daughters

    data, graph, _ = load_tracks_for_labels(path, 3)
    assert graph == {2: [1], 3: [1]}


def test_without_lineage_divisions_are_unknown(tmp_path):
    movie = np.zeros((3, 8, 8), np.uint32)
    movie[:, 1, 1] = 1
    path = str(tmp_path / "m_tracked.tif")
    tifffile.imwrite(path, movie, ome=True, metadata={"axes": "TYX"})
    assert read_lineage(path) is None
    stats = track_statistics_qc(None, _source_filepath=path)
    assert stats["divisions"] is None and stats["tracks_appearing"] is None


def test_ultrack_parents_count_as_lineage(tmp_path):
    path = tmp_path / "m_ultrack.tif"
    path.touch()
    pd.DataFrame(
        {"track_id": [1, 2, 3], "t": [0, 1, 1], "y": [0, 0, 0], "x": [0, 0, 0],
         "parent_track_id": [-1, 1, 1]}
    ).to_csv(tmp_path / "m_ultrack_tracks.csv", index=False)
    assert read_lineage(str(path)) == {2: 1, 3: 1}


def test_scripted_stats_without_a_file_have_no_lineage():
    movie = np.zeros((3, 8, 8), np.uint32)
    movie[:, 1, 1] = 1
    stats = track_statistics_qc(movie)
    assert stats["divisions"] is None


@pytest.mark.parametrize(
    "name, expected",
    [("a_tracked.tif", "a_tracked_lineage.csv"), ("b.zarr", "b_lineage.csv")],
)
def test_lineage_path(name, expected):
    assert lineage_csv_path(f"/d/{name}") == f"/d/{expected}"

