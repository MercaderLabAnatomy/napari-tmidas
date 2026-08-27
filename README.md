
# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/MercaderLabAnatomy/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Supported Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://python.org)
[![Downloads](https://static.pepy.tech/badge/napari-tmidas)](https://pepy.tech/project/napari-tmidas)
[![GitHub stars](https://badgen.net/github/stars/MercaderLabAnatomy/napari-tmidas)](https://github.com/MercaderLabAnatomy/napari-tmidas/stargazers)
[![DOI](https://zenodo.org/badge/943353883.svg)](https://doi.org/10.5281/zenodo.17988815)
[![tests](https://github.com/MercaderLabAnatomy/napari-tmidas/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)](https://github.com/MercaderLabAnatomy/napari-tmidas/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/MercaderLabAnatomy/napari-tmidas/branch/main/graph/badge.svg)](https://codecov.io/gh/MercaderLabAnatomy/napari-tmidas)


**Need fast batch processing for confocal & whole-slide microscopy images of biological cells and tissues?**

This open-source napari plugin integrates state-of-the-art AI + analysis tools in an interactive GUI with side-by-side result comparison! Transform, analyze, and quantify microscopy data at scale including deep learning - from file conversion to segmentation, tracking, and analysis.

![napari-tmidas-interactive-table-example](https://github.com/user-attachments/assets/1330cc6c-18de-46f4-a7ef-e1d7ffc3970e)


## ✨ Key Features

🤖 **AI Methods Built-In**
- Virtual staining (VisCy) • Denoising (CAREamics) • Spot detection (Spotiflow) • Segmentation (Cellpose, Convpaint) • Tracking (Trackastra, HOCT, Ultrack)
- Auto-install in isolated environments • No dependency conflicts • GPU acceleration

🔄 **Universal File Conversion**
- Convert LIF, ND2, CZI, NDPI, Acquifer → TIFF or OME-Zarr
- Preserve spatial metadata automatically

⚡ **Batch Processing**
- Process entire folders with one click • 40+ processing functions • Progress tracking & quality control

� **Interactive Workflow**
- Side-by-side table view of original and processed images • Click to instantly compare results • Quickly iterate parameter values • Real-time visual feedback

�📊 **Complete Analysis Pipeline**
- Segmentation → Tracking → Quantification → Colocalization

## 🚀 Quick Start


Supports Python 3.11+; commands below use Python 3.12.
```sh
# Install napari and the plugin
mamba create -y -n napari-tmidas -c conda-forge python=3.12
mamba activate napari-tmidas
pip install "napari[all]"
pip install napari-tmidas

# Launch napari
napari
```

Then find napari-tmidas in the **Plugins** menu. [Watch video tutorials →](https://www.youtube.com/@macromeer/videos)

> **💡 Tip**: AI methods (SAM2, Cellpose, Spotiflow, etc.) auto-install into isolated environments on first use - no manual setup required!

## 📖 Documentation

### AI-Powered Methods

| Method | Description | Documentation |
|--------|-------------|---------------|
| 🎨 **VisCy** | Virtual staining from phase/DIC | [Guide](docs/viscy_virtual_staining.md) |
| 🔧 **CAREamics** | Noise2Void/CARE denoising | [Guide](docs/careamics_denoising.md) |
| 🎯 **Spotiflow** | Spot/puncta detection | [Guide](docs/spotiflow_detection.md) |
| 🔬 **Cellpose** | Cell/nucleus segmentation | [Guide](docs/cellpose_segmentation.md) |
| 🎨 **Convpaint** | Custom semantic/instance segmentation | [Guide](docs/convpaint_prediction.md) |
| 📈 **Trackastra** | Transformer-based cell tracking | [Guide](docs/trackastra_tracking.md) |
| 🧬 **HOCT** | Transformer-based cell tracking (Higher-Order Cell Tracking Transformer) | [Guide](docs/hoct_tracking.md) |
| 🔗 **Ultrack** | Cell tracking based on segmentation ensemble | [Guide](docs/ultrack_tracking.md) |

### Core Workflows

- **[File Conversion](docs/file_conversion.md)** - Multi-format microscopy file conversion (LIF, ND2, CZI, NDPI, Acquifer)
- **[Batch Processing](docs/all_processing_functions.md)** - All 40+ processing functions in one place
- **[Frame Removal](docs/frame_removal.md)** - Interactive human-in-the-loop frame removal from time series
- **[Label-Based Cropping](docs/label_based_cropping.md)** - Interactive ROI extraction with label expansion
- **[Quality Control](docs/grid_view_overlay.md)** - Visual QC with grid overlay
- **[Quantification](docs/regionprops_analysis.md)** - Extract measurements from labels

### Advanced Features

- [Batch Crop Anything](docs/crop_anything.md) - Interactive object cropping with SAM2
- [Batch Label Inspection](docs/batch_label_inspection.md) - Manual label verification and editing, with one-click delete/relabel across all timepoints, click-to-split for merged objects, and click-to-merge-neighbors for over-segmented ones
- [Multichannel Processing](docs/multichannel_processing.md) - Channel selection and per-channel processing

## 💻 Installation

### Step 1: Install napari

```sh
mamba create -y -n napari-tmidas -c conda-forge python=3.12
mamba activate napari-tmidas
python -m pip install "napari[all]"
```

### Step 2: Install napari-tmidas

| Your Needs | Command |
|----------|---------|
| **Standard installation** | `pip install napari-tmidas` |
| **Want the latest dev features** | `pip install git+https://github.com/MercaderLabAnatomy/napari-tmidas.git` |

## 🖼️ Screenshots

<details>
<summary><b>File Conversion Widget</b></summary>

<img src="https://github.com/user-attachments/assets/e377ca71-2f30-447d-825e-d2feebf7061b" alt="File Conversion" width="600">

Convert proprietary formats to open standards with metadata preservation.
</details>

<details>
<summary><b>Batch Processing Interface</b></summary>

<img src="https://github.com/user-attachments/assets/cfe84828-c1cc-4196-9a53-5dfb82d5bfce" alt="Batch Processing" width="600">

Select files → Choose processing function → Run on entire dataset.
</details>

<details>
<summary><b>Label Inspection</b></summary>

<img src="https://github.com/user-attachments/assets/0bf8c6ae-4212-449d-8183-e91b23ba740e" alt="Label Inspection" width="600">

Inspect and manually correct segmentation results.
</details>

<details>
<summary><b>SAM2 Crop Anything</b></summary>

<img src="https://github.com/user-attachments/assets/6d72c2a2-1064-4a27-b398-a9b86fcbc443" alt="Crop Anything" width="600">

Interactive object selection and cropping with SAM2.
</details>

## 📋 TODO

### Memory-Efficient Streaming

Most of this is done. Batch processing no longer materializes whole stacks: the worker keeps large inputs lazy and streams results back to disk block by block (256 MB budget), 15 functions map their existing body over blocks via the `@chunked` decorator, and 6 more own their I/O outright via `skip_load`. Measured end to end on a real `(31, 2, 57, 2720, 2720)` uint16 acquisition — 52 GB dense — Gamma Correction peaks at **3.15 GB RSS** in 8.8 min, byte-identical to the dense path. Convpaint prediction, Cellpose segmentation and Trackastra tracking all write per-timepoint now. The mechanism is documented in [`_chunked.py`](src/napari_tmidas/processing_functions/_chunked.py), and the behaviour is pinned by `TestZarrOutputStreaming`, `TestSplitChannelsStreaming`, `TestLazyTiffLoading`, `TestCLAHEDaskMemory` and `TestRollingBallPerPlane`.

What is left:

- **CAREamics denoising and VisCy virtual staining still run dense.** Both allocate the full output array and take the input as a NumPy array. Neither has been audited for real — that needs their dedicated virtualenvs installed.
- **~14 registered functions still scale linearly with input size.** That is now a known list rather than an unknown one. Each needs either a `@chunked` conversion or a check that it cannot take one: functions using global statistics (`Convert to 8-bit` rescales by the whole-stack min/max) and functions with cross-block topology (`Mirror Labels`) both resist it.
- **The structural decision.** Dense functions opt into laziness one at a time, by accepting `_source_filepath`. Whether to keep converting them individually or change the worker's contract for all of them at once is still open.

### Other Known Issues

- `Resize Zarr by YX Scale (OME-Zarr native)` imports `zarr.storage.FSStore`, which zarr v3 removed, so the native path raises `ImportError` on every run and silently falls back. Fixing it needs a call on the v3 replacement (`LocalStore`, or `FsspecStore` for remote) and on whether napari-ome-zarr still needs `key_separator="/"` stated.

## 🤝 Contributing

Contributions are welcome! Please ensure tests pass before submitting PRs:

```sh
pip install tox
tox
```

## 📄 License

BSD-3 License - see [LICENSE](LICENSE) for details.

## 🐛 Issues

Found a bug or have a feature request? [Open an issue](https://github.com/MercaderLabAnatomy/napari-tmidas/issues)

## 🙏 Acknowledgments

Built with [napari](https://github.com/napari/napari) and powered by:

**AI/ML Methods:**
- [Cellpose](https://github.com/MouseLand/cellpose) • [Convpaint](https://github.com/guiwitz/napari-convpaint) • [VisCy](https://github.com/mehta-lab/VisCy) • [CAREamics](https://github.com/CAREamics/careamics) • [Spotiflow](https://github.com/weigertlab/spotiflow) • [Trackastra](https://github.com/weigertlab/trackastra) • [HOCT](https://github.com/royerlab/hoct) • [Ultrack](https://github.com/royerlab/ultrack) • [SAM2](https://github.com/facebookresearch/segment-anything-2)

**Core Scientific Stack:**
- [NumPy](https://numpy.org/) • [scikit-image](https://scikit-image.org/) • [PyTorch](https://pytorch.org/)

**File Format Support:**
- [OME-Zarr](https://github.com/ome/ome-zarr-py) • [tifffile](https://github.com/cgohlke/tifffile) • [nd2](https://github.com/tlambert03/nd2) • [pylibCZIrw](https://github.com/ZEISS/pylibczi) • [readlif](https://github.com/nimne/readlif)

---

[PyPI]: https://pypi.org/project/napari-tmidas
[pip]: https://pypi.org/project/pip/
[tox]: https://tox.readthedocs.io/en/latest/

