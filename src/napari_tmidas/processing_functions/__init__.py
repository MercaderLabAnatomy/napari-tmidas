# processing_functions/__init__.py
"""
Package for processing functions that can be registered with the batch processing system.
"""
import importlib
import os
import pkgutil
import sys
from typing import Dict, List

# Keep the registry global
from napari_tmidas._registry import BatchProcessingRegistry


def discover_and_load_processing_functions(reload: bool = False) -> List[str]:
    """
    Discover and load all processing functions from the processing_functions package.

    Args:
        reload: If True, force reload of modules even if already imported

    Returns:
        List of registered function names
    """
    # Get the current package
    package = __name__

    # Find all modules in the package
    for _, module_name, is_pkg in pkgutil.iter_modules(
        [os.path.dirname(__file__)]
    ):
        if not is_pkg:  # Only load non-package modules
            try:
                module_fullname = f"{package}.{module_name}"
                
                # If reload requested and module already loaded, reload it
                if reload and module_fullname in sys.modules:
                    importlib.reload(sys.modules[module_fullname])
                else:
                    # Import the module
                    importlib.import_module(module_fullname)
                
                print(f"Loaded processing function module: {module_name}")
            except (ImportError, ValueError) as e:
                # Log the error but continue with other modules
                # ValueError catches NumPy binary incompatibility issues
                error_msg = str(e)
                if "numpy.dtype size changed" in error_msg:
                    print(
                        f"Failed to import {module_name}: NumPy binary incompatibility. "
                        "Try: pip install --force-reinstall --no-cache-dir scikit-learn-extra"
                    )
                else:
                    print(f"Failed to import {module_name}: {e}")

    # Return the list of registered functions
    return BatchProcessingRegistry.list_functions()


def get_processing_function_info() -> Dict[str, Dict]:
    """
    Get information about all registered processing functions.

    Returns:
        Dictionary of function information
    """
    return {
        name: {
            "description": BatchProcessingRegistry.get_function_info(name).get(
                "description", ""
            ),
            "suffix": BatchProcessingRegistry.get_function_info(name).get(
                "suffix", ""
            ),
            "parameters": BatchProcessingRegistry.get_function_info(name).get(
                "parameters", {}
            ),
        }
        for name in BatchProcessingRegistry.list_functions()
    }
