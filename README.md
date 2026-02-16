# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/macromeer/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tmidas.svg?color=green)](https://python.org)
[![Downloads](https://static.pepy.tech/badge/napari-tmidas)](https://pepy.tech/project/napari-tmidas)
[![Downloads](https://pepy.tech/badge/napari-tmidas/month)](https://pepy.tech/project/napari-tmidas)
[![DOI](https://zenodo.org/badge/943353883.svg)](https://doi.org/10.5281/zenodo.17988815)
[![tests](https://github.com/macromeer/napari-tmidas/workflows/tests/badge.svg)](https://github.com/macromeer/napari-tmidas/actions)


**Automated batch processing for microscopy images**

Transform, analyze, and quantify microscopy data at scale including deep learning - from file conversion to segmentation, tracking, and analysis.

## ✨ Key Features

🤖 **AI Methods Built-In**
- Virtual staining (VisCy) • Denoising (CAREamics) • Spot detection (Spotiflow) • Segmentation (Cellpose, Convpaint) • Tracking (Trackastra)
- Auto-install in isolated environments • No dependency conflicts • GPU acceleration

🔄 **Universal File Conversion**
- Convert LIF, ND2, CZI, NDPI, Acquifer → TIFF or OME-Zarr
- Preserve spatial metadata automatically

⚡ **Batch Processing**
- Process entire folders with one click • 40+ processing functions • Progress tracking & quality control

📊 **Complete Analysis Pipeline**
- Segmentation → Tracking → Quantification → Colocalization

## 🚀 Quick Start

```bash
# Install napari and the plugin
mamba create -y -n napari-tmidas -c conda-forge python=3.11
mamba activate napari-tmidas
pip install "napari[all]"
pip install napari-tmidas

# Launch napari
napari
```

Then find napari-tmidas in the **Plugins** menu. [Watch video tutorials →](https://www.youtube.com/@macromeer/videos)

> **💡 Tip**: AI methods auto-install their dependencies on first use - no manual setup required!

## 📖 Documentation

### AI-Powered Methods

| Method | Description | Documentation |
|--------|-------------|---------------|
| 🎨 **VisCy** | Virtual staining from phase/DIC | [Guide](docs/viscy_virtual_staining.md) |
| 🔧 **CAREamics** | Noise2Void/CARE denoising | [Guide](docs/careamics_denoising.md) |
| 🎯 **Spotiflow** | Spot/puncta detection | [Guide](docs/spotiflow_detection.md) |
| 🔬 **Cellpose** | Cell/nucleus segmentation | [Guide](docs/cellpose_segmentation.md) |
| 🎨 **Convpaint** | Custom semantic/instance segmentation | [Guide](docs/convpaint_prediction.md) |
| 📈 **Trackastra** | Cell tracking over time | [Guide](docs/trackastra_tracking.md) |

### Core Workflows

- **[File Conversion](docs/file_conversion.md)** - Multi-format microscopy file conversion (LIF, ND2, CZI, NDPI, Acquifer)
- **[Batch Processing](docs/basic_processing.md)** - Label operations, filters, channel splitting
- **[Frame Removal](docs/frame_removal.md)** - Interactive human-in-the-loop frame removal from time series
- **[Label-Based Cropping](docs/label_based_cropping.md)** - Interactive ROI extraction with label expansion
- **[Quality Control](docs/grid_view_overlay.md)** - Visual QC with grid overlay
- **[Quantification](docs/regionprops_analysis.md)** - Extract measurements from labels
- **[Colocalization](docs/advanced_processing.md#colocalization-analysis)** - Multi-channel ROI analysis

### Advanced Features

- [Batch Crop Anything](docs/crop_anything.md) - Interactive object cropping with SAM2
- [Batch Label Inspection](docs/batch_label_inspection.md) - Manual label verification and editing
- [SciPy Filters](docs/advanced_processing.md#scipy-filters) - Gaussian, median, morphological operations
- [Scikit-Image Filters](docs/advanced_processing.md#scikit-image-filters) - CLAHE, thresholding, edge detection

## 💻 Installation

### Step 1: Install napari

```bash
mamba create -y -n napari-tmidas -c conda-forge python=3.11
mamba activate napari-tmidas
python -m pip install "napari[all]"
```

### Step 2: Install napari-tmidas

| Your Needs | Command |
|----------|---------|
| **Just process & convert images** | `pip install napari-tmidas` |
| **Need AI features** (SAM2, Cellpose, Spotiflow, etc.) | `pip install 'napari-tmidas[deep-learning]'` |
| **Want the latest dev features** | `pip install git+https://github.com/MercaderLabAnatomy/napari-tmidas.git` |

**Recommended for most users:** `pip install 'napari-tmidas[deep-learning]'`

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

## 🤝 Contributing

Contributions are welcome! Please ensure tests pass before submitting PRs:

```bash
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
- [Cellpose](https://github.com/MouseLand/cellpose) • [Convpaint](https://github.com/guiwitz/napari-convpaint) • [VisCy](https://github.com/mehta-lab/VisCy) • [CAREamics](https://github.com/CAREamics/careamics) • [Spotiflow](https://github.com/weigertlab/spotiflow) • [Trackastra](https://github.com/weigertlab/trackastra) • [SAM2](https://github.com/facebookresearch/segment-anything-2)

**Core Scientific Stack:**
- [NumPy](https://numpy.org/) • [scikit-image](https://scikit-image.org/) • [PyTorch](https://pytorch.org/)

**File Format Support:**
- [OME-Zarr](https://github.com/ome/ome-zarr-py) • [tifffile](https://github.com/cgohlke/tifffile) • [nd2](https://github.com/tlambert03/nd2) • [pylibCZIrw](https://github.com/ZEISS/pylibczi) • [readlif](https://github.com/nimne/readlif)

---

[PyPI]: https://pypi.org/project/napari-tmidas
[pip]: https://pypi.org/project/pip/
[tox]: https://tox.readthedocs.io/en/latest/
