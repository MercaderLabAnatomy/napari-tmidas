# Ultrack Environment Patches

This document describes the compatibility patches applied to the ultrack environment in napari-tmidas.

## Overview

The ultrack conda environment requires several patches to ensure compatibility with:
- Modern GPU architectures (NVIDIA Blackwell sm_120)
- Latest dependency versions (scikit-image 0.26+, scipy)
- Robust CPU/GPU fallback behavior

## Patches Applied

### 1. Blackwell GPU Support (sm_120)

**Problem**: 
- CuPy (used by ultrack for GPU acceleration) does not support NVIDIA Blackwell architecture
- Blackwell GPUs (RTX 50 series, compute capability 12.0) fail with CuPy operations
- Affects: `labels_to_contours()` function used for boundary detection

**Solution**:
- Custom PyTorch implementation of `labels_to_contours()`
- PyTorch 2.5+ nightly builds include sm_120 support
- Implementation produces identical output to CuPy version
- Included inline in generated tracking scripts

**Files**:
- Implementation: `src/napari_tmidas/processing_functions/torch_labels_to_contours.py`
- Integration: `src/napari_tmidas/processing_functions/ultrack_tracking.py` (line ~80-200)

**Why PyTorch?**:
- NVIDIA officially supports Blackwell in PyTorch 2.5+
- CuPy Blackwell support timeline unclear
- PyTorch provides identical GPU acceleration
- No dependency on CuPy for GPU operations

---

### 2. scikit-image 0.26+ Compatibility

**Problem**:
- scikit-image 0.26.0 deprecated `min_size` parameter in `morphology.remove_small_objects()`
- Parameter replaced with `max_size`
- ultrack's `hierarchy.py` uses old `min_size` parameter
- Causes deprecation warnings and potential future breaks

**Solution**:
- Runtime patch to ultrack's `hierarchy.py`
- Changes `min_size=` to `max_size=` in `remove_small_objects()` call

**Patch Script**: `/tmp/fix_ultrack_hierarchy.py`

```python
#!/usr/bin/env python3
"""Fix deprecated min_size parameter in ultrack hierarchy.py"""
import sys
from pathlib import Path

hierarchy_file = Path(sys.argv[1])
content = hierarchy_file.read_text()

# Fix the deprecated min_size parameter
old_code = """        morphology.remove_small_objects(
            labels,
            min_size=int(kwargs["min_area"] / kwargs.get("min_area_factor", 4.0)),
            out=labels,
        )"""

new_code = """        morphology.remove_small_objects(
            labels,
            max_size=int(kwargs["min_area"] / kwargs.get("min_area_factor", 4.0)),
            out=labels,
        )"""

content = content.replace(old_code, new_code)
hierarchy_file.write_text(content)
print(f"✓ Fixed deprecated min_size parameter")
```

**Apply Manually**:
```bash
# Find hierarchy.py in ultrack environment
HIERARCHY_PATH=$(conda run -n ultrack python -c "import ultrack; from pathlib import Path; print(Path(ultrack.__file__).parent / 'core' / 'segmentation' / 'hierarchy.py')")

# Apply patch
python /tmp/fix_ultrack_hierarchy.py "$HIERARCHY_PATH"
```

**Status**: ⚠️ Currently requires manual application after environment creation

---

### 3. NumPy Array Module Assignment Fix

**Problem**:
- When CuPy is installed but CUDA is unavailable, ultrack's `cuda.py` doesn't set `xp = np`
- Causes `NameError: name 'xp' is not defined` when GPU is disabled
- Affects CPU-only execution paths

**Solution**:
- Auto-applied patch during environment creation
- Adds `xp = np` assignment when CUDA unavailable

**Patch Code**:
```python
# OLD (buggy):
if not cp.cuda.is_available():
    cp = None
    LOG.info("cupy found but cuda is not available.")
else:
    xp = cp

# NEW (fixed):
if not cp.cuda.is_available():
    cp = None
    xp = np  # ← Added this line
    LOG.info("cupy found but cuda is not available.")
else:
    xp = cp
```

**Files Patched**:
- `<ultrack_env>/lib/python3.11/site-packages/ultrack/utils/cuda.py`

**Status**: ✅ Automatically applied by `ultrack_env_manager._patch_ultrack_xp()`

**Implementation**: `src/napari_tmidas/processing_functions/ultrack_env_manager.py` (line ~203-292)

---

### 4. Read-Only Zarr Array Handling (Bonus)

**Problem**:
- ultrack's solver fails with read-only zarr arrays during fancy indexing
- Causes "assignment destination is read-only" errors

**Solution**:
- Auto-applied patch to make arrays writable before indexing

**Status**: ✅ Automatically applied by `ultrack_env_manager._patch_ultrack_readonly_arrays()`

**Implementation**: `src/napari_tmidas/processing_functions/ultrack_env_manager.py` (line ~295-380)

---

## Patch Application Timeline

| Patch | When Applied | How | Status |
|-------|-------------|-----|--------|
| Blackwell GPU support | Script generation | Inline PyTorch code | ✅ Automatic |
| scikit-image 0.26+ fix | After env creation | Manual script | ⚠️ Manual |
| NumPy xp assignment | During env creation | Auto-patch | ✅ Automatic |
| Read-only zarr arrays | During env creation | Auto-patch | ✅ Automatic |

## To-Do: Automate scikit-image Patch

The scikit-image 0.26+ fix should be integrated into `ultrack_env_manager.py` as an auto-applied patch similar to the xp assignment fix.

**Implementation plan**:
1. Add `_patch_ultrack_hierarchy()` function to `ultrack_env_manager.py`
2. Call during environment creation after package installation
3. Check if already patched before applying (idempotent)
4. Remove need for manual `/tmp/fix_ultrack_hierarchy.py` script

## Testing GPU Support

To verify Blackwell GPU support:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: sm_{props.major}{props.minor}")
    
    # Test tensor operation
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print("✓ GPU operations working!")
```

Expected output for Blackwell:
```
PyTorch: 2.5.0+cu130
CUDA available: True
GPU: NVIDIA RTX PRO 4000 Blackwell
Compute capability: sm_120
✓ GPU operations working!
```

## Patch Maintenance

These patches are temporary workarounds until upstream fixes are available:

**Blackwell support**:
- Upstream: CuPy issue [#XXXX](https://github.com/cupy/cupy/issues)
- Remove when: CuPy supports sm_120 natively
- Timeline: Unknown

**scikit-image 0.26+**:
- Upstream: ultrack issue/PR needed
- Remove when: ultrack updates to `max_size` parameter
- Timeline: Pending ultrack maintainer response

**xp assignment**:
- Upstream: ultrack issue/PR needed  
- Remove when: ultrack fixes CPU fallback
- Timeline: Could submit PR

## References

- [PyTorch Blackwell support](https://github.com/pytorch/pytorch/releases)
- [scikit-image 0.26.0 deprecations](https://scikit-image.org/docs/stable/release_notes/v0.26.html)
- [ultrack repository](https://github.com/royerlab/ultrack)
- [CuPy GPU support](https://docs.cupy.dev/en/stable/install.html)
