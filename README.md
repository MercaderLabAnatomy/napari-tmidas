# napari-tmidas

[![License BSD-3](https://img.shields.io/pypi/l/napari-tmidas.svg?color=green)](https://github.com/macromeer/napari-tmidas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tmidas.svg?color=green)](https://pypi.org/project/napari-tmidas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tmidas.svg?color=green)](https://python.org)
[![tests](https://github.com/macromeer/napari-tmidas/workflows/tests/badge.svg)](https://github.com/macromeer/napari-tmidas/actions)
<!-- [![codecov](https://codecov.io/gh/macromeer/napari-tmidas/branch/main/graph/badge.svg)](https://codecov.io/gh/macromeer/napari-tmidas) -->
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-tmidas)](https://napari-hub.org/plugins/napari-tmidas)

The Tissue Microscopy Image Data Analysis Suite (short: T-MIDAS), is a collection of pipelines for image preprocessing, segmentation, regions-of-interest (ROI) analysis and other useful batch image processing features. This is a work in progress (WIP) and an evolutionary step away from the [terminal / command-line version of T-MIDAS](https://github.com/MercaderLabAnatomy/T-MIDAS).

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

First install napari in a virtual environment following latest [Napari installation instructions](https://github.com/Napari/napari?tab=readme-ov-file#installation).


You can install `napari-tmidas` via [pip]:

    pip install napari-tmidas



To install latest development version :

    pip install git+https://github.com/macromeer/napari-tmidas.git

## Usage
1. You can find the installed plugin here:
   
![image](https://github.com/user-attachments/assets/41f83fbd-5cc2-4b26-89f7-b4224016a405)

3. After opening the plugin, select the folder with the images to be processed (currently supports TIF, later also ZARR). You can also filter for filename suffix.
   
![image](https://github.com/user-attachments/assets/41ecb689-9abe-4371-83b5-9c5eb37069f9)

5. As a result, a table appears with the found images.
   
![image](https://github.com/user-attachments/assets/8360942a-be8f-49ec-bc25-385ee43bd601)

7. Next, select a processing function, set parameters if applicable and start batch processing.
   
![image](https://github.com/user-attachments/assets/05929660-6672-4f76-89da-4f17749ccfad)

9. You can click on the images to show them in the viewer. For example first click on the original, then the processed image to see an overlay.
    
![image](https://github.com/user-attachments/assets/cfe84828-c1cc-4196-9a53-5dfb82d5bfce)







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

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
