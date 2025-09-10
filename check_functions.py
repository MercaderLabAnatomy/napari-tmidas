#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from napari_tmidas.processing_functions import (
    discover_and_load_processing_functions,
)

print("Loading processing functions...")
funcs = discover_and_load_processing_functions()
print(f"\nFound {len(funcs)} registered functions:")
for func in sorted(funcs):
    print(f"  - {func}")

# Check if spotiflow is in the list
spotiflow_funcs = [
    f for f in funcs if "spotiflow" in f.lower() or "spot" in f.lower()
]
if spotiflow_funcs:
    print("\nSpotiflow-related functions found:")
    for func in spotiflow_funcs:
        print(f"  ✓ {func}")
else:
    print("\n⚠ No Spotiflow functions found in registry")
