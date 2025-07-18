# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/macromeer/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tmidas.svg?color=green)](https://python.org)
[![tests](https://github.com/macromeer/napari-tmidas/workflows/tests/badge.svg)](https://github.com/macromeer/napari-tmidas/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-tmidas)](https://napari-hub.org/plugins/napari-tmidas)
<!-- [![codecov](https://codecov.io/gh/macromeer/napari-tmidas/branch/main/graph/badge.svg)](https://codecov.io/gh/macromeer/napari-tmidas) -->
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

It is recommended though to install the **latest development version**. Please also execute this command from time to time in the activated environment to benefit from newly added features:

    pip install git+https://github.com/MercaderLabAnatomy/napari-tmidas.git

To use the Batch Crop Anything pipeline, we need to install **Segment Anything 2** (2D/3D):

    cd /opt # if the folder does not exist: mkdir /opt && cd /opt
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    pip install -e .
    curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -o checkpoints/sam2.1_hiera_large.pt
    mamba install -c conda-forge ffmpeg # we also need ffmpeg

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

Currently, this pipeline supports the conversion of `.nd2, .lif, .ndpi, .czi` and acquifer data. After scanning a folder of your choice for microscopy image data, select a file in the first column of the table and preview and export any image data it contains.


<img src="https://github.com/user-attachments/assets/e377ca71-2f30-447d-825e-d2feebf7061b" alt="Microscopy Image Conversion Widget" style="width:75%; height:auto;">


### Image Processing

1. You start with entering the path to the folder containing the images to be processed (currently supports TIF, later also ZARR) and optionally a filter for filename suffix

![image](https://github.com/user-attachments/assets/41ecb689-9abe-4371-83b5-9c5eb37069f9)

2. After indexing the files, a table appears with the found images. You can click on them to inspect them in the viewer.

![image](https://github.com/user-attachments/assets/8360942a-be8f-49ec-bc25-385ee43bd601)

3. Next, select a processing function, set parameters if applicable and `Start Batch Processing`.

![image](https://github.com/user-attachments/assets/05929660-6672-4f76-89da-4f17749ccfad)

4. You can click on the images in the table to show them in the viewer. For example first click on one of the `Original Files`, and then the corresponding `Processed File` to see an overlay.

<img src="https://github.com/user-attachments/assets/cfe84828-c1cc-4196-9a53-5dfb82d5bfce" alt="Image Processing Widget" style="width:75%; height:auto;">


Note that whenever you click on an `Original File` or `Processed File` in the table, it will replace the one that is currently shown in the viewer. So naturally, you'd first select the original image, and then the processed image to correctly see the image pair that you want to inspect.


#### Processing Function Credits

The image processing capabilities are powered by several excellent open-source tools:
- [Cellpose 4](https://github.com/MouseLand/cellpose): Advanced cell segmentation
- [Trackastra](https://github.com/weigertlab/trackastra): Cell tracking and analysis
- [CAREamics](https://github.com/CAREamics/careamics): Content-aware image restoration and enhancement

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
