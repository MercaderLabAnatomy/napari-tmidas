"""
Common UI utilities for napari widgets.
"""

import os

# Lazy imports for optional heavy dependencies
try:
    from qtpy.QtWidgets import QFileDialog, QPushButton

    _HAS_QTPY = True
except ImportError:
    QFileDialog = QPushButton = None
    _HAS_QTPY = False


def add_browse_button_to_folder_field(widget, folder_field_name: str):
    """
    Add a browse button next to a folder path field in a magicgui widget.

    Parameters
    ----------
    widget : magicgui widget
        The widget containing the folder field
    folder_field_name : str
        The name of the folder field attribute

    Returns
    -------
    QWidget
        The modified widget with browse button
    """
    folder_field = getattr(widget, folder_field_name)

    # Create browse button
    browse_button = QPushButton("Browse...")

    def on_browse_clicked():
        current_value = folder_field.value
        start_dir = current_value if current_value else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            folder_field.value = folder

    browse_button.clicked.connect(on_browse_clicked)

    # Insert the browse button next to the folder_path field
    field_layout = folder_field.native.parent().layout()
    if field_layout:
        field_layout.addWidget(browse_button)

    return widget
