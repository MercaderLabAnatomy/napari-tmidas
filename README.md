# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/macromeer/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tmidas.svg?color=green)](https://python.org)
[![tests](https://github.com/macromeer/napari-tmidas/workflows/tests/badge.svg)](https://github.com/macromeer/napari-tmidas/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-tmidas)](https://napari-hub.org/plugins/napari-tmidas)
<!-- [![codecov](https://codecov.io/gh/macromeer/napari-tmidas/branch/main/graph/badge.svg)](https://codecov.io/gh/macromeer/napari-tmidas) -->
The `napari-tmidas` plugin consists of a growing collection of pipelines for fast batch processing of microscopy images.

## Feature Overview
* Currently, napari-tmidas includes four pipelines: Batch Image Processing, Batch Label Inspection, Batch Microscopy Image Conversion, Batch Crop Anything.
* The Batch Image Processing Pipeline allows to inspect a folder of images, apply a function and inspect the results. Currently available functions: Gamma correction (to enhance or suppress dark regions), maximum z projection to convert 3D image stacks to 2D images with brightest pixels, split color channels, Gaussian blur, median filter, Otsu thresholding (semantic or instance), manual thresholding and remove small labels from label images.
* The Batch Label Inspection pipeline allows you to sequentially go through a folder of label images, any edits will be automatically saved.
* The Batch Microscopy Image COnversion pipeline allows you to convert .nd2, .lif, .ndpi, .czi and acquifer data to either multidimensional .tif or .zarr formats while keeping essential metadata such as resolution.
* The Batch Crop Anything pipeline allows you to sequerntially go through a folder of images where you want to select regions of interest to keep while deleting other regions by clicking on them.
* More features coming up in April 2025.

## Installation

First install Napari in a virtual environment:

    mamba create -y -n napari-tmidas -c conda-forge python=3.11 tqdm
    mamba activate napari-tmidas
    python -m pip install "napari[all]"

Now you can install `napari-tmidas` via [pip]:

    pip install napari-tmidas

To install the latest development version:

    pip install git+https://github.com/macromeer/napari-tmidas.git

### Dependencies
To use the Batch Microscopy Image Conversion pipeline, we need some libraries to read microscopy formats and to write ome-zarr:

    pip install nd2 readlif tiffslide pylibCZIrw acquifer-napari ome-zarr

For the Batch Crop Anything pipeline, we need to install MobileSAM and its dependencies:

    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    pip install torch torchvision timm opencv-python

## Usage

To use the plugin, start napari in the activated virtual environment with this terminal command:

    mamba run -n napari-tmidas napari

You can find the installed plugin here:

![image](https://github.com/user-attachments/assets/504db09a-d66e-49eb-90cd-3237024d9d7a)


### Batch Microscopy Image Conversion

You can start this pipeline via `Plugins > T-MIDAS > Batch Microscopy Image Conversion`. Currently, this pipeline supports the conversion of `.nd2, .lif, .ndpi, .czi` and acquifer data. After scanning a folder of your choice for microscopy image data, select a file in the first column of the table and preview and export any image data it contains.

![image](https://github.com/user-attachments/assets/e377ca71-2f30-447d-825e-d2feebf7061b)

### Batch File Processing

1. After opening `Plugins > T-MIDAS > Batch Image Processing`, enter the path to the folder containing the images to be processed (currently supports TIF, later also ZARR). You can also filter for filename suffix.

![image](https://github.com/user-attachments/assets/41ecb689-9abe-4371-83b5-9c5eb37069f9)

2. As a result, a table appears with the found images.

![image](https://github.com/user-attachments/assets/8360942a-be8f-49ec-bc25-385ee43bd601)

3. Next, select a processing function, set parameters if applicable and `Start Batch Processing`.

![image](https://github.com/user-attachments/assets/05929660-6672-4f76-89da-4f17749ccfad)

4. You can click on the images in the table to show them in the viewer. For example first click on one of the `Original Files`, and then the corresponding `Processed File` to see an overlay.

![image](https://github.com/user-attachments/assets/cfe84828-c1cc-4196-9a53-5dfb82d5bfce)

Note that whenever you click on an `Original File` or `Processed File` in the table, it will replace the one that is currently shown in the viewer. So naturally, you'd first select the original image, and then the processed image to correctly see the image pair that you want to inspect.

### Batch Label Inspection
If you have already segmented a folder full of images and now you want to maybe inspect and edit each label image, you can use the `Plugins > T-MIDAS > Batch Label Inspection`, which automatically saves your changes to the existing label image once you click the `Save Changes and Continue` button (bottom right).

![image](https://github.com/user-attachments/assets/0bf8c6ae-4212-449d-8183-e91b23ba740e)

### Batch Crop Anything
This pipeline combines the Segment Anything Model (SAM) for automatic object detection with an interactive interface for selecting and cropping multiple objects from images. To launch the widget, open `Plugins > T-MIDAS > Batch Crop Anything`

![image](https://github.com/user-attachments/assets/6d72c2a2-1064-4a27-b398-a9b86fcbc443)


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
