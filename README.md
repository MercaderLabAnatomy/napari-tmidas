# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/macromeer/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tmidas.svg?color=green)](https://python.org)
[![Downloads](https://static.pepy.tech/badge/napari-tmidas)](https://pepy.tech/project/napari-tmidas)
[![DOI](https://zenodo.org/badge/943353883.svg)](https://doi.org/10.5281/zenodo.17988815)
[![tests](https://github.com/macromeer/napari-tmidas/workflows/tests/badge.svg)](https://github.com/macromeer/napari-tmidas/actions)

This napari plugin consists of a growing collection of pipelines for fast batch processing of confocal and whole slide microscopy images of biological tissues. This is a WIP and based on the [T-MIDAS terminal](https://github.com/MercaderLabAnatomy/T-MIDAS).

## Features
Currently, **napari-tmidas** provides pipelines as widgets for batch image conversion and processing, object cropping, label image inspection and ROI colocalization (cf. [usage](#usage) below). You can request new batch image processing features in [issues](https://github.com/MercaderLabAnatomy/napari-tmidas/issues).

## Installation

(Video installation guides: https://www.youtube.com/@macromeer/videos)

First, install Napari in a virtual environment:

    mamba create -y -n napari-tmidas -c conda-forge python=3.11
    mamba activate napari-tmidas
    python -m pip install "napari[all]"

Now you can install `napari-tmidas` via [pip]:

    pip install napari-tmidas

**For deep learning features** (Batch Crop Anything with SAM2, VisCy, Spotiflow, CAREamics, Trackastra, Cellpose), also install:

    pip install 'napari-tmidas[deep-learning]'

Or install everything at once:

    pip install 'napari-tmidas[all]'

> **Note**: AI methods (VisCy, CAREamics, Spotiflow, Cellpose, Trackastra) run in isolated environments that are automatically created on first use. This prevents dependency conflicts and requires no manual setup.

It is recommended though to install the **latest development version**. Please also execute this command from time to time in the activated environment to benefit from newly added features:

    pip install git+https://github.com/MercaderLabAnatomy/napari-tmidas.git

### Additional Setup for Batch Crop Anything

To use the Batch Crop Anything pipeline with SAM2, you only need to install ffmpeg manually. SAM2 and its model checkpoints will be automatically installed when you first run the pipeline:

    mamba install -c conda-forge ffmpeg

If you want to batch compress image data using [Zstandard](https://github.com/facebook/zstd), use the package manager of your operating system to install it:

   ~~sudo apt-get install zstd~~    # Pre-installed on Linux :man_shrugging:

    brew install zstd            # for macOS (requires Homebrew)
    pip install zstandard        # Windows with Python >= 3.7

And you are done!

## Usage

To use the plugin, start napari in the activated virtual environment with this terminal command:

    mamba run -n napari-tmidas napari

You can then find the installed plugin in the Plugins tab.

### Microscopy Image Conversion

Converts `.lif, .nd2, .czi, .ndpi` and Acquifer data to TIF or OME-Zarr formats. Scan a folder, select files, and export with preserved spatial metadata.

**Supported Formats:**
- **TIF** - Standard format for compatibility
- **OME-Zarr** - Recommended for large datasets, [spec v0.5](https://ngff.openmicroscopy.org/latest/) compliant with automatic physical metadata extraction (voxel sizes, spacing)

<img src="https://github.com/user-attachments/assets/e377ca71-2f30-447d-825e-d2feebf7061b" alt="Microscopy Image Conversion Widget" style="width:75%; height:auto;">


### Image Processing

1. You start with entering the path to the folder containing the images to be processed (supports TIFF and OME-Zarr) and optionally a filter for filename suffix

![image](https://github.com/user-attachments/assets/41ecb689-9abe-4371-83b5-9c5eb37069f9)

2. After indexing the files, a table appears with the found images. You can click on them to inspect them in the viewer.

![image](https://github.com/user-attachments/assets/8360942a-be8f-49ec-bc25-385ee43bd601)

3. Next, select a processing function, set parameters if applicable and `Start Batch Processing`.

![image](https://github.com/user-attachments/assets/05929660-6672-4f76-89da-4f17749ccfad)

4. You can click on the images in the table to show them in the viewer. For example first click on one of the `Original Files`, and then the corresponding `Processed File` to see an overlay.

<img src="https://github.com/user-attachments/assets/cfe84828-c1cc-4196-9a53-5dfb82d5bfce" alt="Image Processing Widget" style="width:75%; height:auto;">


Note that whenever you click on an `Original File` or `Processed File` in the table, it will replace the one that is currently shown in the viewer. So naturally, you'd first select the original image, and then the processed image to correctly see the image pair that you want to inspect.


#### AI-Powered Processing Functions 🤖

napari-tmidas integrates state-of-the-art deep learning methods for microscopy image analysis. Each method runs in a dedicated environment to prevent conflicts and is automatically installed on first use.

**🔬 Image Enhancement & Restoration**
- **[VisCy Virtual Staining](docs/viscy_virtual_staining.md)** - Transform phase/DIC microscopy into virtual fluorescence (nuclei + membrane channels)
  - Powered by [VisCy](https://github.com/mehta-lab/VisCy)
  - VSCyto3D model for 3D/4D data
  - No labels needed - learn from transmitted light alone
  
- **[CAREamics Denoising](docs/careamics_denoising.md)** - Content-aware image restoration using Noise2Void/CARE
  - Powered by [CAREamics](https://github.com/CAREamics/careamics)
  - Train on noisy images (no ground truth needed)
  - Preserves fine structures while removing noise

**🎯 Detection & Segmentation**
- **[Spotiflow Spot Detection](docs/spotiflow_detection.md)** - Accurate detection of spots/puncta in fluorescence images
  - Powered by [Spotiflow](https://github.com/weigertlab/spotiflow)
  - Multiple pretrained models (2D/3D)
  - Subpixel localization for precise coordinates
  
- **[Cellpose Segmentation](docs/cellpose_segmentation.md)** - Universal cell/nucleus segmentation
  - Powered by [Cellpose 4](https://github.com/MouseLand/cellpose)
  - Cellpose-SAM for improved generalization
  - Works on 2D, 3D, and time-lapse data

**🔍 Tracking & Analysis**
- **[Trackastra Tracking](docs/trackastra_tracking.md)** - Cell tracking across time-lapse data
  - Powered by [Trackastra](https://github.com/weigertlab/trackastra)
  - Graph-based tracking with deep learning
  - Handles divisions and complex cell behavior

#### Additional Processing Functions

**Core Operations**
- [Basic Processing Functions](docs/basic_processing.md) - Label and intensity operations, channel splitting/merging, time series

**Analysis & Quality Control**
- [Grid View: Intensity + Labels Overlay](docs/grid_view_overlay.md) - Visual QC for segmentation results
- [Intensity-Based Label Filtering](docs/intensity_label_filter.md) - Filter labels by signal intensity
- [Regionprops Analysis](docs/regionprops_analysis.md) - Extract quantitative properties from labels

**Advanced Methods**
- [Advanced Processing Functions](docs/advanced_processing.md) - SciPy/scikit-image filters, compression, colocalization

### Batch Label Inspection
If you have already segmented a folder full of images and now you want to maybe inspect and edit each label image, you can use the `Plugins > T-MIDAS > Batch Label Inspection`, which automatically saves your changes to the existing label image once you click the `Save Changes and Continue` button (bottom right).

<img src="https://github.com/user-attachments/assets/0bf8c6ae-4212-449d-8183-e91b23ba740e" alt="Batch Label Inspection Widget" style="width:75%; height:auto;">

### Crop Anything

This pipeline combines the Segment Anything Model (SAM2; supports YX, ZYX and TYX data) for automatic object detection with an interactive interface for selecting and cropping multiple objects from images. To launch the widget, open `Plugins > T-MIDAS > Batch Crop Anything`. Cropping works like this: Enter 2D view and go to the first z slice where the object to be cropped is appearing. Activate/select the points layer and click on the object. Terminal shows progress. You can then proceed to select another object (always do this in 2D mode)

<img src="https://github.com/user-attachments/assets/6d72c2a2-1064-4a27-b398-a9b86fcbc443" alt="Crop Anything Widget" style="width:75%; height:auto;">




### ROI Colocalization

This pipeline quantifies colocalization between labeled regions of interest (ROIs) across multiple image channels. It determines the extent of overlap between ROIs in a reference channel and those in one or two other channels. The output is a table of colocalization counts. Optionally, the size of reference channel ROIs, as well as the total or median size of colocalizing ROIs in the other channels, can be included. Colocalization is determined using Boolean masking. The number of colocalizing instances is determined by counting unique label IDs within the overlapping regions. Typically, the reference channel contains larger structures, while other channels contain smaller, potentially nested, structures. For example, the reference channel might contain cell bodies, with the second and third channels containing nuclei and sub-nuclear objects, respectively.

<img src="https://github.com/user-attachments/assets/2f9022a0-7b88-4588-a448-250f07a634d7" alt="ROI Colocalization Widget" style="width:75%; height:auto;">

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-tmidas" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/macromeer/napari-tmidas/issues

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
