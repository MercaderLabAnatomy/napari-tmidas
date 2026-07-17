# Batch Label Inspection

The Batch Label Inspection widget enables interactive verification, correction, and refinement of segmentation labels across entire image datasets. Inspect and manually edit label images while automatically saving changes back to disk.

## Overview

Streamline the quality control workflow for segmentation results:

- **Side-by-side viewing**: Original image + label mask for easy comparison
- **Interactive editing**: Use napari's paint, eraser, and selection tools
- **Automatic saving**: Changes saved to disk as you proceed through pairs
- **Progress tracking**: Navigate through entire dataset with visual progress indicator
- **Batch workflow**: Process hundreds of images without manual file management
- **Format validation**: Automatic detection and validation of label image formats

## Quick Start

1. Open napari and navigate to **Plugins → napari-tmidas → Batch Label Inspection**
2. Select folder containing your image-label pairs
3. Specify label suffix (e.g., `_labels.tif`, `_segmentation.tif`)
4. Click **Load** to index image-label pairs
5. Edit labels in the viewer using napari's drawing tools
6. Click **Save and Continue** to save changes and move to next pair
7. Click **Previous** to revisit earlier pairs if needed

## Workflow

### Step 1: Prepare Your Files

Organize files with a consistent naming pattern:

```
segmentation_results/
├── sample1.tif              (original image)
├── sample1_labels.tif       (segmentation labels)
├── sample2.tif              (original image)
├── sample2_labels.tif       (segmentation labels)
└── ...
```

**File Requirements**:
- Label images must be integer type (8-bit, 16-bit, or 32-bit)
- Image and label must have matching spatial dimensions
- Any image format supported by scikit-image (TIF, PNG, etc.)

### Step 2: Configure Inspection

**Folder Path**: Select directory containing image-label pairs

**Label Suffix**: Specify the suffix that identifies label files
- Examples: `_labels`, `_segmentation`, `_mask`, `_labels_filtered`
- The suffix is used to match labels with images
- File before suffix is treated as the base image name

**Example matching**:
```
sample1.tif + sample1_labels.tif          ✓ Match
sample1_labels.tif + sample1.tif          ✓ Match (order doesn't matter)
sample1_seg.tif + sample1_seg_labels.tif  ✗ No match (_labels not in sample1_seg.tif)
sample1_labels_filtered.tif               ✓ Match (if suffix is "_labels")
```

**Raw channel axis**: Controls how the raw image's channel dimension is
aligned with the (channel-less) label. Leave on **Auto** for well-formed
OME-TIFF/Zarr; override only when Auto misaligns the overlay. See
[Multi-Channel Raw Image Alignment](#multi-channel-raw-image-alignment).

### Step 3: Load Pairs

Click **Load** to scan the folder and create image-label pairs.

**Status Report**:
- Number of valid pairs found
- Any skipped files and reasons
- Format validation issues (if any)

### Step 4: Edit and Review

For each pair displayed:

**Viewing**:
- Left panel: Original image
- Right panel: Label layer (editable)
- Status bar: Current pair number and filename

**Editing Tools** (napari built-in):
- **Paint**: Add new labels
  - Select label ID from right panel
  - Click and drag to paint
- **Eraser**: Remove labels
  - Set label ID to 0 to erase
- **Selection tools**: Select and modify regions
- **Undo/Redo**: Ctrl+Z / Ctrl+Y

**Click Modes** (napari-tmidas, see [One-Click Label Editing](#one-click-label-editing-all-timepoints)):
- **Click-to-delete**: Left-click a label to remove it from every timepoint
- **Click-to-relabel**: Ctrl+click to pipette an ID, then left-click labels to reassign them to it on every timepoint
- **Click-to-split** (see [Splitting Merged Labels](#splitting-merged-labels)): click one point per cell inside an under-segmented label, then **Apply split** to divide it at the clicked timepoint

**Track-level tools** (napari-tmidas, for tracked time series):
- **Whole-track 3D views** (see [Whole-Track 3D Inspection](#whole-track-3d-inspection)): view the entire movie as one 3-D volume so each track is a single clickable object
- **Delete low-intensity tracks** (see [Delete Low-Intensity Tracks](#delete-low-intensity-tracks)): remove every track dimmer than a threshold in one step, with a live preview

**Viewing Tips**:
- Adjust label opacity (right panel) to see image beneath
- Use different colormaps for better visibility
- Toggle layers on/off to compare

### Step 5: Save Progress

**Save and Continue**:
- Saves current label edits to disk
- Moves to next image-label pair
- Shows confirmation status

**Previous**:
- Saves current edits
- Returns to previous pair (useful for refinement)

**Stop**:
- Saves final edits and closes widget

## Features

### One-Click Label Editing (all timepoints)

Two toggleable click modes, each in its own dock widget, edit a label across
**every timepoint** of a time series (e.g. tracked TZYX labels) with a single
click. Both use a lazy remapping path, so they are instant even for stacks far
larger than RAM — no data is rewritten until you save.

**Click-to-Delete** — enable *"Click a label to delete it from all timepoints"*:

- **Left-click** any label in the viewer to remove it from every timepoint
  (e.g. delete a whole mistracked cell in one click)
- Replaces the slow bucket-fill workflow, which loads the entire array

**Click-to-Relabel** — enable *"Click a label to relabel it (Ctrl+click picks up an ID)"*:

- **Ctrl+left-click** a label to *pipette* its ID (this sets napari's
  selected label)
- **Left-click** other labels to reassign them to the pipetted ID on every
  timepoint — clicked objects are merged into the target label
  (e.g. fix a track that switched IDs halfway through)
- Alternatively, pick the target ID with napari's own tools: select the
  **pipette (color picker)** in the layer controls and click a label,
  switch back to the **pan/zoom tool (camera symbol)**, then click the
  labels to relabel — or simply type the ID into the label spinbox
- Relabeling to ID 0 is equivalent to deletion

**Behavior common to both modes**:

- **Ctrl+Z** undoes the last delete/relabel (paint edits made in between are
  preserved)
- Click-dragging (pan/zoom) and clicks on background do nothing
- Edits are staged in memory; press **Save and Continue** to write them to
  the file — saved operations can no longer be undone
- The two modes are mutually exclusive: enabling one switches the other off
- The active mode persists as you move through image-label pairs

### Splitting Merged Labels

Docked as **Split label**, this tool divides an under-segmented label — two
or more touching cells that a segmenter gave a single ID — into separate
labels. It is the inverse of Click-to-Relabel's merge, and unlike the other
click modes it edits **only the clicked timepoint** (each frame's geometry is
different, so there is no all-timepoints shortcut).

Enable *"Click one point per cell to split a merged label"*, then:

- **Left-click** one point inside each cell of the merged label — one seed per
  cell. The status bar shows the running seed count.
- **Ctrl+left-click** removes the most recently placed seed.
- Press **Apply split**. A seeded watershed on the label's mask cuts it at the
  constrictions between the seeds. The first seed's region keeps the original
  ID; every other region gets a new, globally-unique ID.

Notes:

- Two or more seeds are required, all on the **same label and timepoint** — a
  click on a different label or timepoint starts a fresh seed set. Clustered
  cells split in one pass: place a seed in each, then Apply once.
- **Ctrl+Z** merges the whole split back in one step.
- Splits are staged in memory; press **Save and Continue** to write them —
  saved splits can no longer be undone.
- Works in the **normal frame view only** (2D or 3D display) — turn *Track
  view* off first, since the projected track views cannot resolve a precise
  source voxel to seed from.
- Mutually exclusive with Click-to-Delete and Click-to-Relabel; the mode
  persists as you move through image-label pairs.

### Whole-Track 3D Inspection

Docked as **Track inspection**, the *Track view* dropdown shows the whole
movie as a **single 3-D volume** so each track (label ID) appears as one
connected, clickable object. Switch napari to its 3D display to see entire
tracks at once, and use the click modes to delete or relabel a whole track
with a single click. This turns per-timepoint scrubbing into a single
overview of every track's lifetime.

Three modes:

- **Off** *(default)* — the normal side-by-side layers.
- **Stack T along Z** — concatenates the timepoints into one lazy `(T*Z, Y, X)`
  volume (plane `i` = timepoint `i//Z`, slice `i%Z`). **Fully editable**:
  paint and fill map back to the correct `(t, z)`, so all normal editing works
  (unless the view had to be downsampled to fit GPU memory — see below).
- **Max-project Z per T** — shows one Z-projected plane per timepoint, a
  `(T, Y, X)` volume in which tracks read as clean tubes. **Painting is
  disabled** here (a projected pixel has no unique Z origin), and where labels
  overlap in Z the higher ID wins. The ID-based click tools still work.

**Behavior**:

- Both views read through the same lazy (dask) wrapper as the normal view —
  no data is copied while scrubbing in 2D. napari's 3D display loads the whole
  volume into RAM **and uploads it to the GPU as one 3-D texture**.
- Movies whose full view volume would exceed the GPU budget (4 GiB by
  default, override with the `NAPARI_TMIDAS_TRACK_VIEW_GB` environment
  variable) are **automatically YX-downsampled** by the smallest integer
  step that fits — e.g. a `33×75×2720×2720` uint32 movie (a ~68 GiB stacked
  volume) is shown at step 5, ~2.8 GiB. Label IDs, the click tools, Ctrl+Z
  and saving are unaffected (the layer scale compensates, and the file stays
  full resolution); only painting is disabled in a downsampled view, since a
  strided pixel cannot be written back losslessly. The status bar reports
  when a view is downsampled.
- Click-to-delete / click-to-relabel, **Ctrl+Z** undo, and **Save Changes and
  Continue** all work exactly as in the normal view; label files stay TZYX.
- Delete/relabel edits remap the cached 3-D volume in place, so 3-D picking and
  refreshes cost no extra I/O.
- The chosen view persists as you move through image-label pairs and is rebuilt
  over each new pair. Requires a 3-D (TYX) or 4-D (TZYX) label source.

### Delete Low-Intensity Tracks

Docked as **Delete low-intensity tracks**, this tool removes every track
(label ID) whose raw-image brightness falls below a threshold — across **all
timepoints** — in one step. It is aimed at tracked data where dim, spurious
tracks should be culled in bulk rather than clicked away one by one.

**Controls**:

- **Intensity threshold (0–1)** — a track's brightness is the median of its
  raw voxel intensities, normalized to `0–1` using the raw image's own global
  min/max. Because normalization is relative to the image, the **same threshold
  works for 8-bit and 16-bit** images (and for data that only occupies part of
  its dtype range, e.g. a 12-bit camera). `0` shows all tracks.
- **Measure channel** — for a multi-channel raw, which channel supplies the
  intensity: **Mean** averages all channels, or pick a channel index (`0`–`4`,
  0-based along the raw's channel axis) to score on a single marker. Ignored for
  single-channel raws.

**Live preview and workflow**:

1. Set the threshold and press **Apply** to preview the deletion.
2. Re-applying is safe — the previous preview is **restored first**, so each
   Apply reflects only the current settings rather than compounding. Slide the
   threshold and Apply again to refine.
3. The status bar reports how many of the measured tracks were removed (and the
   deleted IDs, when few).
4. Deletions are staged in memory and **undoable with Ctrl+Z** (while a click
   mode is active); press **Save Changes and Continue** to write them to disk.

> **Note**: This in-inspector tool is distinct from the k-medoids
> [Intensity-Based Label Filtering](intensity_label_filter.md) batch-processing
> functions — here you set an explicit threshold with a live preview inside the
> inspector, rather than clustering labels automatically.

### Automatic Pair Matching

The widget intelligently matches images with their labels:

```
Input: label suffix "_labels"

✓ Correct matches:
  image.tif ↔ image_labels.tif
  sample_001.tif ↔ sample_001_labels.tif
  data_ch1.tif ↔ data_ch1_labels.tif

✗ No match:
  image1.tif + image2_labels.tif (different base names)
  file_labels.tif (no matching image found)
```

### Multi-Channel Raw Image Alignment

Raw images often carry a **channel** dimension that the label lacks — for
example a `TZCYX` raw (time, Z, 2 channels, Y, X) paired with a `TZYX`
tracked label. napari aligns axes from the last dimension backwards, so
if the channel axis is not identified, the label's timepoints get matched
against the raw's Z (and Z against C), and the overlay is misaligned.

The widget resolves the raw's channel axis so it can (a) split the raw into
one layer per channel and (b) exclude that axis when scaling the label to the
raw's spatial extent. Resolution follows a layered strategy, most-trusted
first:

1. **Manual override** (*Raw channel axis* dropdown) — wins whenever set:
   - **Auto** *(default)* — detect automatically (steps 2–3 below)
   - **None** — the raw has no channel axis
   - **0**–**4** — force that axis index as the channel dimension
2. **Metadata** — read from the file's axes (OME-TIFF `DimensionOrder`,
   ImageJ hyperstack order, or Zarr `.zattrs`). This is source-robust: the
   channel axis is read from the file rather than assumed at a fixed slot, so
   images written by ImageJ/Java, OME, or plain Python all resolve correctly
   even when a singleton dimension is squeezed away.
3. **Shape heuristic** — when no axes metadata exists, the channel axis is
   guessed as the small dimension (size 2–16) with larger Y/X. This handles
   most layouts but is ambiguous when a real Z or T axis is also small — in
   that case, set the *Raw channel axis* dropdown manually.

A resolved index is always range-checked against the loaded array, so a stale
or incorrect value degrades to "no channel axis" rather than corrupting the
overlay. The status bar reports the channel axis and the label scale in use.

### Format Validation

Automatic checks ensure label integrity:

- **Integer type validation**: Labels must be integer (not float/RGB)
- **File format support**: TIF, PNG, etc. (any scikit-image format)
- **Dimension matching**: Labels must match image spatial dimensions
- **Error reporting**: Detailed messages for any validation issues

### Progress Tracking

**Status Bar Display**:
```
Viewing pair 5 of 47: sample_005.tif
```

Shows:
- Current pair number
- Total number of pairs
- Current filename

Navigate using **Previous** / **Save and Continue** buttons

### Automatic Saving

**When saving**:
- Current label layer written to disk
- Original filename preserved
- Data type preserved (8/16/32-bit as original)
- File overwritten (use backup if needed)
- Status confirmed in notification

## Use Cases

### Quality Control of Automated Segmentation

After running Cellpose or another segmenter:
1. Load output label images
2. Visually compare with original images
3. Fix errors (merge split objects, remove false positives)
4. Auto-saves corrections

### Merging Split Objects

When segmentation over-splits cells:
1. Enable **Click-to-Relabel**
2. Ctrl+click the object to keep (pipettes its ID)
3. Click the split-off fragments to merge them into it (all timepoints at once)
4. Or paint manually with the same label for partial merges
5. Save changes

### Splitting Merged Objects

When segmentation under-splits — several touching cells share one ID:
1. Enable **Click-to-split**
2. Navigate to the timepoint where the objects are merged
3. Click one point inside each cell (Ctrl+click removes the last seed)
4. Press **Apply split** — the label divides into one region per seed, each
   after the first getting a new ID
5. Repeat at other timepoints as needed, then save changes

### Removing False Positives

When segmentation detects spurious objects:
1. Enable **Click-to-Delete** and click each spurious object —
   removed from every timepoint instantly
2. Or use eraser (label = 0) for partial removal
3. Save corrected labels

### Culling Dim Tracks in Bulk

When tracking produces many faint, spurious tracks:
1. Use **Delete low-intensity tracks**
2. (Multi-channel raw) pick the **Measure channel** for the relevant marker
3. Set the **Intensity threshold** and press **Apply** to preview
4. Adjust the threshold and Apply again until only real tracks remain
5. Save changes

### Inspecting Whole Tracks in 3D

To review a track's entire lifetime at once:
1. Set **Track view** to *Stack T along Z* (editable) or *Max-project Z per T*
   (tubes)
2. Switch napari to its 3D display
3. Rotate to see each track as one connected object
4. Click-to-delete or click-to-relabel whole tracks, then save

### Fixing Tracking ID Switches

When a tracked object changes ID partway through a time series:
1. Enable **Click-to-Relabel**
2. Ctrl+click the object at a timepoint where it has the correct ID
3. Navigate to a timepoint after the switch and click the object —
   the wrong ID is reassigned to the correct one everywhere
4. Save changes

### Refining Boundaries

For inaccurate object boundaries:
1. Paint with same object ID to expand
2. Use eraser to shrink
3. Fine-tune label borders
4. Save refined masks

## Tips & Best Practices

### Organization
- Keep consistent naming scheme across dataset
- Use descriptive suffix names (`_labels_v2`, not just `_v2`)
- Backup original labels before mass editing

### Editing Efficiency
- Edit in 2D view for precise control
- Use opacity adjustment to see image beneath labels
- Zoom in for fine boundary adjustments
- Use selection tools for large regions

### Data Management
- Check "Save and Continue" status confirms write
- Verify edits saved by reloading file
- Use version suffixes for multiple iterations (`_labels_v1`, `_labels_v2`)
- Keep audit trail of manual corrections

### Performance
- For >100 pairs, consider processing in batches
- Verify label format before batch processing
- Use SSD storage for faster loading

## Troubleshooting

### "No Label Files Found"

**Cause**: Suffix doesn't match any files

**Solutions**:
- Check actual label filenames in folder
- Verify suffix spelling and case sensitivity
- Try shorter suffix (e.g., `_labels` instead of `_labels_filtered`)

### "No Valid Image-Label Pairs"

**Cause**: Labels don't match images or format issues

**Solutions**:
- Verify image and label basenames match
- Check label images are integer type (not float/RGB)
- Ensure dimensions match between image and label

### "Format Issues" Warning

**Cause**: Some label files not in expected format

**Possible Issues**:
- Label image is RGB/float instead of integer
- Label file corrupted or incompatible
- Dimension mismatch with image

**Solutions**:
- Convert labels to integer format if needed
- Regenerate problematic label files
- Verify with external tools (ImageJ, etc.)

### Image and Label Overlay Misaligned

**Cause**: The raw image has a channel dimension the label lacks (e.g. a
`TZCYX` raw with a `TZYX` label) and its channel axis could not be identified
automatically — usually a TIFF written without clean axes metadata.

**Solutions**:
- Set the **Raw channel axis** dropdown to the channel dimension's index
  (0-based) before loading — for a `TZCYX` raw that is `2`
- Choose **None** if the raw genuinely has no channel axis
- Confirm the channel axis and label scale reported in the status bar

### Edits Not Saving

**Cause**: Wrong layer selected or permission issue

**Solutions**:
- Ensure "Labels" layer (right panel) is selected
- Check folder write permissions
- Verify label filename in confirmation message

### Changes Lost After Clicking Previous

**Note**: Previous saves current edits first

If edits appear lost:
- Check file modification time
- Reload file to verify save
- Check for backup/version files

## File Format Support

| Format | Input | Output | Status |
|--------|-------|--------|--------|
| TIF/TIFF | ✓ | ✓ | Full support |
| PNG | ✓ | ✓ | Full support |
| JPEG | ✓ (8-bit only) | ✗ | Read-only |
| Zarr | ✓ | Limited | Supported |
| HDF5 | ✗ | ✗ | Not supported |

## Data Types Supported

| Type | Support |
|------|---------|
| uint8 | ✓ Full |
| uint16 | ✓ Full |
| uint32 | ✓ Full |
| int8, int16, int32 | ✓ Supported |
| float, RGB | ✗ Not supported (validation error) |

## Related Features

- **[Cellpose Segmentation](cellpose_segmentation.md)** - Generate labels to inspect
- **[Batch Processing](all_processing_functions.md)** - Post-process labels
- **[Label Operations](all_processing_functions.md#label-image-operations)** - Filter/transform labels
- **[RegionProps Analysis](regionprops_analysis.md)** - Analyze edited labels

## Technical Details

### Workflow Architecture

```
1. User selects folder + suffix
         ↓
2. Widget scans folder
         ↓
3. Matches image-label pairs
         ↓
4. Validates formats
         ↓
5. Loads first pair into napari
         ↓
6. User edits labels
         ↓
7. Click "Save and Continue"
         ↓
8. Write label file to disk
         ↓
9. Load next pair (repeat from step 5)
```

### File Matching Logic

```
Label suffix: "_labels"
Label file: sample1_labels.tif

1. Extract base: "sample1"
2. Find files starting with "sample1"
3. Find files NOT equal to label file
4. Find files with SAME extension (.tif)
5. Match first found = Image file
```

### Format Validation

```
For each label file:
  1. Read file (scikit-image imread)
  2. Check: Is dtype integer?
  3. Check: Does it load without error?
  4. Add to pairs list or report issue
```

## Citation

If you use Batch Label Inspection in your research, please cite:

```bibtex
@software{napari_tmidas_2024,
  title = {napari-tmidas: Batch Image Processing for Microscopy},
  author = {Mercader Lab},
  year = {2024},
  url = {https://github.com/MercaderLabAnatomy/napari-tmidas}
}
```
