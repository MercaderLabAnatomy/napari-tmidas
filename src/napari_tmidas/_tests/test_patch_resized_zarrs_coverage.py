"""Coverage tests for the ``patch_resized_zarrs`` script module.

The module is a stand-alone script that rewrites the ``.zattrs`` of
already-resized OME-Zarr stores in place: it repairs the coordinate
transformations by copying the physical scale from the matching source
store (adjusting Y/X for the new size) and rebuilds the ``omero``
window metadata from the level-0 pixels.

These tests build tiny *real* zarr v2 stores under ``tmp_path`` with
hand-written ``.zattrs`` and drive the axis mapping, the guard
branches, the coordinate-transform maths, the omero window
computation and the ``main()`` CLI entry point (via a patched
``sys.argv``, never a subprocess).
"""

import json
import os

import numpy as np
import pytest
import zarr

from napari_tmidas.processing_functions import patch_resized_zarrs as prz

_AX_TYPES = {"t": "time", "c": "channel"}


def _axes(names):
    """Build an OME axes list from short axis names."""
    return [{"name": n, "type": _AX_TYPES.get(n, "space")} for n in names]


def _make_attrs(
    axes_names,
    scale=None,
    n_levels=1,
    omero=None,
    ctf_type="scale",
):
    """Build a minimal OME-NGFF ``.zattrs`` dict."""
    datasets = []
    for i in range(n_levels):
        ds = {"path": str(i)}
        if scale is not None:
            ds["coordinateTransformations"] = [
                {"type": ctf_type, "scale": list(scale)}
            ]
        datasets.append(ds)
    ms = {"version": "0.4", "axes": _axes(axes_names), "datasets": datasets}
    attrs = {"multiscales": [ms]}
    if omero is not None:
        attrs["omero"] = omero
    return attrs


def _make_store(root, attrs, shape=None, data=None, dtype="uint16"):
    """Create a directory with a ``.zattrs`` and an optional level 0."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".zattrs"), "w") as fh:
        json.dump(attrs, fh)
    if shape is not None:
        arr = zarr.create_array(
            store=os.path.join(root, "0"),
            shape=shape,
            dtype=dtype,
            zarr_format=2,
            chunks=shape,
        )
        arr[:] = (
            np.zeros(shape, dtype=dtype) if data is None else data
        ).astype(dtype)
    return root


def _read_attrs(root):
    """Read back the ``.zattrs`` the module wrote."""
    with open(os.path.join(str(root), ".zattrs")) as fh:
        return json.load(fh)


def _scales(attrs):
    """Return the per-level scale vectors from a patched ``.zattrs``."""
    out = []
    for ds in attrs["multiscales"][0]["datasets"]:
        ctf = ds.get("coordinateTransformations")
        out.append(None if ctf is None else ctf[0]["scale"])
    return out


class TestAxesNameIndices:
    """Pin the OME axis-name -> index mapping helper."""

    def test_maps_all_five_canonical_names(self):
        """t/c/z/y/x dict axes map to their positional index."""
        mapping = prz._axes_name_indices(_axes(["t", "c", "z", "y", "x"]))
        assert mapping == {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}

    def test_axis_type_wins_when_name_is_unknown(self):
        """``type: time`` / ``type: channel`` classify oddly named axes."""
        axes = [
            {"name": "frame", "type": "time"},
            {"name": "stain", "type": "channel"},
            {"name": "y", "type": "space"},
        ]
        assert prz._axes_name_indices(axes) == {"t": 0, "c": 1, "y": 2}

    def test_plain_string_axes_are_accepted(self):
        """Legacy string axes lists are handled without a ``type``."""
        mapping = prz._axes_name_indices(["t", "z", "y", "x"])
        assert mapping == {"t": 0, "z": 1, "y": 2, "x": 3}

    def test_channel_aliases_and_case_insensitivity(self):
        """'C', 'channel' and 'ch' all resolve to the channel axis."""
        assert prz._axes_name_indices(["C"]) == {"c": 0}
        assert prz._axes_name_indices(["channel"]) == {"c": 0}
        assert prz._axes_name_indices(["ch"]) == {"c": 0}
        assert prz._axes_name_indices(["Y", "X"]) == {"y": 0, "x": 1}

    def test_unknown_axes_are_ignored(self):
        """Names the mapping does not recognise are dropped silently."""
        assert prz._axes_name_indices(["q", "w"]) == {}
        assert prz._axes_name_indices([]) == {}

    def test_last_axis_of_a_name_wins(self):
        """A repeated axis name keeps the highest index."""
        assert prz._axes_name_indices(["y", "y"]) == {"y": 1}


class TestPatchZarrGuards:
    """Early-return guards of :func:`patch_zarr`."""

    def test_missing_destination_zattrs_skips(self, tmp_path, capsys):
        """A resized store without ``.zattrs`` is skipped, not written."""
        dst = tmp_path / "a_resized.zarr"
        dst.mkdir()
        src = _make_store(tmp_path / "a.zarr", _make_attrs(["y", "x"]))

        assert prz.patch_zarr(str(dst), src) is None
        assert not (dst / ".zattrs").exists()
        assert "SKIP: no .zattrs found" in capsys.readouterr().out

    def test_missing_source_zattrs_skips(self, tmp_path, capsys):
        """A missing source store leaves the destination untouched."""
        attrs = _make_attrs(["y", "x"], scale=[1.0, 1.0])
        dst = _make_store(tmp_path / "a_resized.zarr", attrs)
        missing = str(tmp_path / "nope.zarr")

        assert prz.patch_zarr(dst, missing) is None
        assert _read_attrs(dst) == attrs
        assert "SKIP: source zarr not found" in capsys.readouterr().out

    def test_empty_dataset_list_skips(self, tmp_path, capsys):
        """A resized ``.zattrs`` with no datasets is skipped."""
        dst_attrs = {"multiscales": [{"axes": _axes(["y", "x"])}]}
        dst = _make_store(tmp_path / "a_resized.zarr", dst_attrs)
        src = _make_store(tmp_path / "a.zarr", _make_attrs(["y", "x"]))

        assert prz.patch_zarr(dst, src) is None
        assert _read_attrs(dst) == dst_attrs
        assert "SKIP: no datasets" in capsys.readouterr().out

    def test_missing_multiscales_key_is_tolerated(self, tmp_path, capsys):
        """A ``.zattrs`` without ``multiscales`` falls back to ``{}``."""
        dst = _make_store(tmp_path / "a_resized.zarr", {"foo": 1})
        src = _make_store(tmp_path / "a.zarr", {"bar": 2})

        assert prz.patch_zarr(dst, src) is None
        assert _read_attrs(dst) == {"foo": 1}
        assert "SKIP: no datasets" in capsys.readouterr().out

    def test_malformed_json_propagates(self, tmp_path):
        """Corrupt ``.zattrs`` raises instead of writing garbage back."""
        dst = tmp_path / "a_resized.zarr"
        dst.mkdir()
        (dst / ".zattrs").write_text("{not json")
        src = _make_store(tmp_path / "a.zarr", _make_attrs(["y", "x"]))

        with pytest.raises(json.JSONDecodeError):
            prz.patch_zarr(str(dst), src)


class TestCoordinateTransforms:
    """The physical-scale repair applied to the resized datasets."""

    def test_yx_scale_is_rescaled_and_pyramid_doubles(self, tmp_path, capsys):
        """Y/X scale is multiplied by src/dst size and by 2**level."""
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "c", "z", "y", "x"], scale=[1.0, 1.0, 2.0, 0.5, 0.5]
            ),
            shape=(2, 2, 3, 8, 8),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "c", "z", "y", "x"], n_levels=2),
            shape=(2, 2, 3, 4, 4),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [
            [1.0, 1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
        ]
        out = capsys.readouterr().out
        assert "Coordinate transforms updated: level-0 Y/X = 1.0000" in out
        assert ".zattrs written" in out

    def test_unnamed_axes_fall_back_to_the_last_two(self, tmp_path):
        """Without y/x names the trailing two axes are treated as Y/X."""
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "c", "z", "dim3", "dim4"],
                scale=[1.0, 1.0, 2.0, 0.25, 0.25],
            ),
            shape=(1, 1, 1, 8, 4),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "c", "z", "dim3", "dim4"]),
            shape=(1, 1, 1, 2, 2),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [[1.0, 1.0, 2.0, 1.0, 0.5]]

    def test_non_scale_transform_skips_the_fix(self, tmp_path, capsys):
        """A source transform that is not ``scale`` leaves dst alone."""
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "z", "y", "x"],
                scale=[1.0, 1.0, 1.0, 1.0],
                ctf_type="translation",
            ),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(1, 1, 4, 4),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [None]
        assert "no source scale found" in capsys.readouterr().out

    def test_source_without_datasets_skips_the_fix(self, tmp_path, capsys):
        """A source ``.zattrs`` with no datasets is not fatal."""
        src = _make_store(
            tmp_path / "a.zarr", {"multiscales": [{"axes": _axes(["y"])}]}
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(1, 1, 4, 4),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [None]
        assert "no source scale found" in capsys.readouterr().out

    def test_trailing_channel_axis_is_dropped_from_the_scale(self, tmp_path):
        """A resized store with fewer dims loses the channel scale entry."""
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "z", "y", "x", "c"], scale=[1.0, 2.0, 0.5, 0.5, 1.0]
            ),
            shape=(2, 3, 8, 8, 2),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(2, 3, 4, 4),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [[1.0, 2.0, 1.0, 1.0]]

    def test_leading_channel_axis_drop_raises_index_error(self, tmp_path):
        """KNOWN BUG pin -- not desired behaviour.

        ``patch_zarr`` probes the *destination* level-0 shape with the
        *source* axis indices (lines 82-85) before the channel-removal
        code at lines 95-100 gets a chance to shift them.  With the
        canonical TCZYX source layout a channel-dropped resized store is
        only 4-D, so ``dst_arr0.shape[x_idx]`` with ``x_idx == 4``
        blows up.  The store SHOULD instead be patched with a 4-entry
        TZYX scale, exactly as
        ``test_trailing_channel_axis_is_dropped_from_the_scale`` gets
        for the (accidentally working) trailing-C layout.

        This test therefore asserts the crash on purpose so the defect
        stays visible; when the source is fixed it will fail and MUST be
        rewritten to assert the correct scale rather than re-pinned.
        """
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "c", "z", "y", "x"], scale=[1.0, 1.0, 2.0, 0.5, 0.5]
            ),
            shape=(2, 2, 3, 8, 8),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(2, 3, 4, 4),
        )

        with pytest.raises(IndexError):
            prz.patch_zarr(dst, src)


    def test_dim_drop_without_a_channel_axis_keeps_every_entry(
        self, tmp_path, capsys
    ):
        """Fewer dst dims but no ``c`` axis leaves the scale length alone.

        Line 97 (``if c_idx is not None``) has a false branch: the
        resized store lost a dimension the axes list never named
        ``c``, so nothing is removed and the emitted scale keeps all
        three entries even though the array is 2-D.  The omero pass
        then trips over the same mismatch and only warns.
        """
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(["y", "x", "z"], scale=[0.5, 0.5, 1.0]),
            shape=(8, 8, 4),
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["y", "x", "z"]),
            shape=(4, 4),
        )

        prz.patch_zarr(dst, src)

        assert _scales(_read_attrs(dst)) == [[1.0, 1.0, 1.0]]
        assert "omero" not in _read_attrs(dst)
        assert "WARNING: omero metadata failed" in capsys.readouterr().out


class TestOmeroMetadata:
    """The rebuilt ``omero`` block and its contrast windows."""

    def _no_scale_source(self, tmp_path, omero=None):
        """A source store whose transform is not a scale."""
        return _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "c", "z", "y", "x"],
                scale=[1.0] * 5,
                ctf_type="translation",
                omero=omero,
            ),
        )

    def test_per_channel_windows_and_source_channel_reuse(self, tmp_path):
        """Each output channel gets its own window; src entries win."""
        src = self._no_scale_source(
            tmp_path,
            omero={
                "version": "0.5",
                "rdefs": {"model": "color"},
                "channels": [{"label": "DAPI", "color": "0000FF"}],
            },
        )
        data = np.zeros((1, 2, 1, 4, 4), dtype="uint16")
        data[0, 0] = np.arange(16, dtype="uint16").reshape(1, 4, 4)
        data[0, 1] = 7
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "c", "z", "y", "x"]),
            shape=(1, 2, 1, 4, 4),
            data=data,
        )

        prz.patch_zarr(dst, src)
        omero = _read_attrs(dst)["omero"]

        assert omero["version"] == "0.5"
        assert omero["rdefs"] == {"model": "color"}
        assert len(omero["channels"]) == 2

        ch0, ch1 = omero["channels"]
        assert ch0["label"] == "DAPI"
        assert ch0["color"] == "0000FF"
        assert ch0["active"] is True
        assert ch0["window"] == {"min": 0, "max": 15, "start": 0, "end": 15}

        assert ch1["label"] == "Channel 1"
        assert ch1["color"] == "00FF00"
        assert ch1["window"] == {"min": 7, "max": 7, "start": 7, "end": 7}

    def test_outlier_maximum_is_clamped_to_ten_times_p99(self, tmp_path):
        """A single hot pixel clamps ``window.end`` to ``10 * p99``."""
        src = self._no_scale_source(tmp_path)
        flat = np.full(100, 5, dtype="uint16")
        flat[-1] = 60000
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "c", "z", "y", "x"]),
            shape=(1, 1, 1, 10, 10),
            data=flat.reshape(1, 1, 1, 10, 10),
        )

        prz.patch_zarr(dst, src)
        window = _read_attrs(dst)["omero"]["channels"][0]["window"]

        # p99 of 99x5 + 1x60000 interpolates to 604.95 -> int 604,
        # so ``end`` is clamped to 6040 instead of the raw max 60000.
        assert window == {
            "min": 5,
            "max": 60000,
            "start": 5,
            "end": 6040,
        }

    def test_four_dimensional_store_gets_one_default_channel(self, tmp_path):
        """TZYX data yields a single white channel labelled ``Channel 0``."""
        src = self._no_scale_source(tmp_path)
        data = np.arange(2 * 3 * 4 * 4, dtype="uint16").reshape(2, 3, 4, 4)
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(2, 3, 4, 4),
            data=data,
        )

        prz.patch_zarr(dst, src)
        channels = _read_attrs(dst)["omero"]["channels"]

        assert len(channels) == 1
        assert channels[0]["label"] == "Channel 0"
        assert channels[0]["color"] == "FFFFFF"
        # t_idxs/z_idxs exhaust both non-YX dims here, so every voxel is
        # sampled and start/end land exactly on the true min/max (no p99
        # clamp kicks in).
        assert channels[0]["window"] == {
            "min": 0,
            "max": int(data.max()),
            "start": 0,
            "end": int(data.max()),
        }

    def test_missing_omero_version_defaults_to_0_3(self, tmp_path):
        """No source omero block at all still yields a versioned one."""
        src = self._no_scale_source(tmp_path)
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(1, 1, 2, 2),
        )

        prz.patch_zarr(dst, src)
        omero = _read_attrs(dst)["omero"]

        assert omero["version"] == "0.3"
        assert [c["label"] for c in omero["channels"]] == ["Channel 0"]

    def test_five_d_store_without_channel_axis(self, tmp_path):
        """A 5-D array whose axes lack ``c`` collapses to one channel."""
        src = self._no_scale_source(tmp_path)
        data = np.arange(2 * 2 * 2 * 3 * 3, dtype="uint16").reshape(
            2, 2, 2, 3, 3
        )
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "q", "z", "y", "x"]),
            shape=(2, 2, 2, 3, 3),
            data=data,
        )

        prz.patch_zarr(dst, src)
        channels = _read_attrs(dst)["omero"]["channels"]

        assert len(channels) == 1
        # t/z sampling plus the fully-included "q" dim exhausts the array,
        # so start/end land exactly on the true min/max (no p99 clamp).
        assert channels[0]["window"] == {
            "min": 0,
            "max": int(data.max()),
            "start": 0,
            "end": int(data.max()),
        }

    def test_missing_level_zero_array_only_warns(self, tmp_path, capsys):
        """A resized store without a level-0 array still gets written."""
        src = self._no_scale_source(tmp_path)
        dst = _make_store(
            tmp_path / "a_resized.zarr", _make_attrs(["t", "z", "y", "x"])
        )

        prz.patch_zarr(dst, src)

        assert "omero" not in _read_attrs(dst)
        captured = capsys.readouterr()
        assert "WARNING: omero metadata failed" in captured.out
        assert ".zattrs written" in captured.out

    def test_source_axes_are_used_when_destination_has_none(self, tmp_path):
        """``dst_axes`` falls back to the source axes list."""
        src = self._no_scale_source(tmp_path)
        dst_attrs = {"multiscales": [{"datasets": [{"path": "0"}]}]}
        data = np.zeros((1, 2, 1, 2, 2), dtype="uint16")
        data[0, 0] = 3
        data[0, 1] = 40
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            dst_attrs,
            shape=(1, 2, 1, 2, 2),
            data=data,
        )

        prz.patch_zarr(dst, src)
        channels = _read_attrs(dst)["omero"]["channels"]

        # Source axes are TCZYX, so the channel axis is found at index 1
        # and each output channel is sampled from its own sub-volume:
        # a broken fallback would give one channel spanning both values.
        assert len(channels) == 2
        assert channels[0]["window"] == {
            "min": 3,
            "max": 3,
            "start": 3,
            "end": 3,
        }
        assert channels[1]["window"] == {
            "min": 40,
            "max": 40,
            "start": 40,
            "end": 40,
        }


class TestMainCli:
    """The ``main()`` entry point, driven through ``sys.argv``."""

    def _record(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            prz, "patch_zarr", lambda r, s: calls.append((r, s))
        )
        return calls

    def test_no_arguments_prints_usage_and_exits(self, monkeypatch, capsys):
        """Calling with no path prints the module docstring and exits 1."""
        monkeypatch.setattr(prz.sys, "argv", ["patch_resized_zarrs.py"])

        with pytest.raises(SystemExit) as exc:
            prz.main()

        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().out

    def test_empty_directory_exits_with_error(
        self, monkeypatch, capsys, tmp_path
    ):
        """A directory holding no ``*.zarr`` exits 1 with a message."""
        empty = tmp_path / "resized"
        empty.mkdir()
        monkeypatch.setattr(prz.sys, "argv", ["prog", str(empty)])

        with pytest.raises(SystemExit) as exc:
            prz.main()

        assert exc.value.code == 1
        assert "No .zarr found" in capsys.readouterr().out

    def test_suffix_variants_are_stripped(self, monkeypatch, capsys, tmp_path):
        """``_yx_resized`` / ``_resized`` / ``_resize`` map to the stem."""
        resized = tmp_path / "resized"
        resized.mkdir()
        names = [
            "a_yx_resized.zarr",
            "b_resized.zarr",
            "c_resize.zarr",
            "d.zarr",
        ]
        for name in names:
            (resized / name).mkdir()
        for stem in ["a.zarr", "b.zarr", "c.zarr", "d.zarr"]:
            (tmp_path / stem).mkdir()

        calls = self._record(monkeypatch)
        monkeypatch.setattr(
            prz.sys, "argv", ["prog", str(resized), str(tmp_path)]
        )
        prz.main()

        assert [os.path.basename(s) for _, s in calls] == [
            "a.zarr",
            "b.zarr",
            "c.zarr",
            "d.zarr",
        ]
        assert [os.path.basename(r) for r, _ in calls] == names
        out = capsys.readouterr().out
        assert "Found 4 resized zarrs to patch" in out
        assert "Done." in out

    def test_source_root_is_inferred_from_the_parent(
        self, monkeypatch, tmp_path
    ):
        """Without a source argument the parent of the dir is used."""
        resized = tmp_path / "resized"
        resized.mkdir()
        (resized / "a_resized.zarr").mkdir()
        (tmp_path / "a.zarr").mkdir()

        calls = self._record(monkeypatch)
        monkeypatch.setattr(prz.sys, "argv", ["prog", str(resized) + "/"])
        prz.main()

        assert calls == [
            (
                str(resized / "a_resized.zarr"),
                str(tmp_path / "a.zarr"),
            )
        ]

    def test_trailing_slash_on_source_argument(self, monkeypatch, tmp_path):
        """A trailing slash on the source argument is stripped."""
        resized = tmp_path / "resized"
        source = tmp_path / "source"
        resized.mkdir()
        source.mkdir()
        (resized / "a_resized.zarr").mkdir()
        (source / "a.zarr").mkdir()

        calls = self._record(monkeypatch)
        monkeypatch.setattr(
            prz.sys, "argv", ["prog", str(resized), str(source) + "/"]
        )
        prz.main()

        assert calls == [
            (
                str(resized / "a_resized.zarr"),
                str(source / "a.zarr"),
            )
        ]

    def test_extensionless_source_directory_is_found(
        self, monkeypatch, tmp_path
    ):
        """A source folder without the ``.zarr`` suffix still matches."""
        resized = tmp_path / "resized"
        resized.mkdir()
        (resized / "a_resized.zarr").mkdir()
        (tmp_path / "a").mkdir()

        calls = self._record(monkeypatch)
        monkeypatch.setattr(
            prz.sys, "argv", ["prog", str(resized), str(tmp_path)]
        )
        prz.main()

        assert calls == [
            (str(resized / "a_resized.zarr"), str(tmp_path / "a"))
        ]

    def test_unmatched_source_keeps_the_stem_path(
        self, monkeypatch, capsys, tmp_path
    ):
        """No candidate directory leaves the (missing) stem path in place."""
        resized = tmp_path / "resized"
        resized.mkdir()
        _make_store(
            resized / "a_resized.zarr", _make_attrs(["t", "z", "y", "x"])
        )

        monkeypatch.setattr(
            prz.sys, "argv", ["prog", str(resized), str(tmp_path)]
        )
        prz.main()

        out = capsys.readouterr().out
        assert "SKIP: source zarr not found" in out
        assert str(tmp_path / "a.zarr") in out

    def test_end_to_end_patches_a_real_store(
        self, monkeypatch, capsys, tmp_path
    ):
        """A full run rewrites scales and omero for a discovered store."""
        resized = tmp_path / "resized"
        resized.mkdir()
        _make_store(
            tmp_path / "sample.zarr",
            _make_attrs(["t", "z", "y", "x"], scale=[1.0, 2.0, 0.325, 0.325]),
            shape=(1, 2, 8, 8),
        )
        data = np.arange(1 * 2 * 4 * 4, dtype="uint16").reshape(1, 2, 4, 4)
        dst = _make_store(
            resized / "sample_yx_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(1, 2, 4, 4),
            data=data,
        )

        monkeypatch.setattr(prz.sys, "argv", ["prog", str(resized)])
        prz.main()

        attrs = _read_attrs(dst)
        assert _scales(attrs) == [[1.0, 2.0, 0.65, 0.65]]
        window = attrs["omero"]["channels"][0]["window"]
        assert window["min"] == 0
        assert window["max"] == int(data.max())
        assert "Done." in capsys.readouterr().out


class TestSamplingAndPixelSafety:
    """Sub-sampling of the window statistics and the no-pixel-write rule."""

    def _no_scale_source(self, tmp_path):
        return _make_store(
            tmp_path / "a.zarr",
            _make_attrs(
                ["t", "c", "z", "y", "x"],
                scale=[1.0] * 5,
                ctf_type="translation",
            ),
        )

    def test_only_five_timepoints_are_sampled(self, tmp_path):
        """Windows are estimated from at most 5 evenly spaced frames.

        ``np.linspace(0, 9, 5)`` truncates to t = 0, 2, 4, 6, 9, so a
        hot pixel that lives only at t = 3 never reaches the stats.
        """
        src = self._no_scale_source(tmp_path)
        data = np.ones((10, 1, 1, 2, 2), dtype="uint16")
        data[3] = 500
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "c", "z", "y", "x"]),
            shape=(10, 1, 1, 2, 2),
            data=data,
        )

        prz.patch_zarr(dst, src)
        window = _read_attrs(dst)["omero"]["channels"][0]["window"]

        assert int(data.max()) == 500
        assert window["max"] == 1
        assert window["min"] == 1

    def test_pixel_data_is_left_untouched(self, tmp_path):
        """Patching rewrites metadata only; level 0 keeps its values."""
        src = _make_store(
            tmp_path / "a.zarr",
            _make_attrs(["t", "z", "y", "x"], scale=[1.0, 1.0, 0.5, 0.5]),
            shape=(1, 1, 8, 8),
        )
        rng = np.random.default_rng(0)
        data = rng.integers(0, 1000, size=(1, 1, 4, 4)).astype("uint16")
        dst = _make_store(
            tmp_path / "a_resized.zarr",
            _make_attrs(["t", "z", "y", "x"]),
            shape=(1, 1, 4, 4),
            data=data,
        )

        prz.patch_zarr(dst, src)

        after = zarr.open_array(os.path.join(dst, "0"), mode="r")[:]
        assert np.array_equal(after, data)
        assert _scales(_read_attrs(dst)) == [[1.0, 1.0, 1.0, 1.0]]

    def test_module_runs_as_a_script(self, monkeypatch, capsys):
        """The ``__main__`` guard dispatches to :func:`main`."""
        import runpy

        monkeypatch.setattr(prz.sys, "argv", ["patch_resized_zarrs.py"])

        with pytest.raises(SystemExit) as exc:
            runpy.run_module(
                "napari_tmidas.processing_functions.patch_resized_zarrs",
                run_name="__main__",
            )

        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().out
