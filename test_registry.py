#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from napari_tmidas.processing_functions import (
    discover_and_load_processing_functions,
)

funcs = discover_and_load_processing_functions()
if "Spotiflow Spot Detection" in funcs:
    print("✓ Spotiflow function found and registered!")
else:
    print("✗ Spotiflow function missing from registry!")

print(f"Total registered functions: {len(funcs)}")
