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
- **Undo/Redo**: Ctrl+Z / Ctrl+Shift+Z (Ctrl+Y is napari's 2D/3D display
  toggle, not redo)

**Click Modes** (napari-tmidas, docked as **Label manipulations**): one
**Click mode** dropdown picks what a left-click does, and only that mode's
settings are shown beneath it. Set it to *Off* to click without editing.
- **Delete label** (see [One-Click Label Editing](#one-click-label-editing-all-timepoints)): left-click a label to remove it from every timepoint
- **Relabel label**: Ctrl+click to pipette an ID, then left-click labels to reassign them to it on every timepoint
- **Split merged label** (see [Splitting Merged Labels](#splitting-merged-labels)): click one point per cell inside an under-segmented label, then **Apply split** to divide it at the clicked timepoint
- **Merge touching neighbors** (see [Merging Touching Neighbors](#merging-touching-neighbors)): click a label to merge every label touching it into it, at the clicked timepoint or across the whole movie
- **Grow label to cell (SAM2)** (see [Growing Incomplete Labels](#growing-incomplete-labels)): click a label that covers only part of its cell and SAM2 extends it to the whole object, at the clicked timepoint
- **Add missed cell (SAM2)** (see [Segmenting Missed Cells](#segmenting-missed-cells)): click a cell that has no label at all and SAM2 segments it into a new one, at the clicked timepoint

**Track-level tools** (napari-tmidas, for tracked time series):
- **Whole-track 3D views** (docked as **Track inspection**, see [Whole-Track 3D Inspection](#whole-track-3d-inspection)): view the entire movie as one 3-D volume so each track is a single clickable object
- **Delete low-intensity tracks** (docked as **Track manipulations**, see [Delete Low-Intensity Tracks](#delete-low-intensity-tracks)): remove every track dimmer than a threshold in one step, with a live preview

**Viewing Tips**:
- Adjust label opacity (right panel) to see image beneath
- Use different colormaps for better visibility
- Toggle layers on/off to compare

### Step 5: Save Progress

Docked together as **Save / Skip**:

**Save Changes and Continue**:
- Saves current label edits to disk
- Moves to next image-label pair
- Shows confirmation status

**Skip (Discard Changes)**:
- Discards any in-memory (unsaved) edits to the current pair
- Moves to next image-label pair without writing to disk
- Useful for pairs that need no correction, or to abandon a mistaken edit

## Features

### One-Click Label Editing (all timepoints)

Two of the **Click mode** dropdown's entries, under **Label manipulations**,
edit a label across **every timepoint** of a time series (e.g. tracked TZYX
labels) with a single click. Both use a lazy remapping path, so they are
instant even for stacks far larger than RAM — no data is rewritten until you
save.

**Delete label** — set *Click mode* to it, leaving *Apply to* on
*All timepoints*:

- **Left-click** any label in the viewer to remove it from every timepoint
  (e.g. delete a whole mistracked cell in one click)
- Replaces the slow bucket-fill workflow, which loads the entire array

**Relabel label** — set *Click mode* to it:

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
- Only one click mode is active at a time — the dropdown makes that automatic
- The active mode persists as you move through image-label pairs

### Splitting Merged Labels

Docked (with the other click modes) as part of **Label manipulations**, this tool divides an under-segmented label — two
or more touching cells that a segmenter gave a single ID — into separate
labels. It is the inverse of the *Relabel label* mode's merge, and unlike the other
click modes it edits **only the clicked timepoint** (each frame's geometry is
different, so there is no all-timepoints shortcut).

Set **Click mode** to *Split merged label*, then:

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
- Mutually exclusive with the other click modes; the mode persists as you
  move through image-label pairs.

### Merging Touching Neighbors

Docked (with the other click modes) as part of **Label manipulations**, this tool is the fix for
*over*-segmentation — one cell broken into several touching IDs — and the
counterpart to [Splitting Merged Labels](#splitting-merged-labels).

Set **Click mode** to *Merge touching neighbors*, then
**left-click** any fragment of the cell. Every label that shares a border with
the clicked one — a shared face, edge or corner all count as touching — is
relabeled to the clicked ID.

**Apply to** decides how far that reaches:

- **Clicked timepoint only** *(default)* — repairs just the frame you clicked,
  like the split tool.
- **All timepoints** — merges those same neighbor IDs across the whole movie.
  This is the right unit for **tracked** data, where a label ID is one object's
  trajectory: an over-segmented cell keeps the same fragment IDs frame to
  frame, so a single click repairs the entire track rather than one frame of
  it. It runs through the value-remap LUT, so it costs no extra I/O however
  long the movie is.

  Which labels count as touching is **always** decided on the clicked frame
  alone — never re-detected per frame. An all-timepoint merge therefore cannot
  quietly swallow a different cell that only brushes past the label at some
  other timepoint; if a fragment only appears later in the movie, scrub to a
  frame where it touches and click again.

Notes:

- Only **direct** neighbors merge, not a neighbor's neighbors. Re-click the
  now-larger label to grow the region another ring outward, so you stay in
  control of how far a merge reaches.
- **Ctrl+Z** reverts the whole merge in one step.
- Merges are staged in memory; press **Save and Continue** to write them —
  saved merges can no longer be undone.
- Works in the **track views** as well: the view only supplies the clicked ID
  and timepoint, and which labels touch is then measured on the source's
  full-resolution spatial slice — so a subsampled or Z-projected view never
  changes the result.
- Mutually exclusive with the other click modes; the mode persists as you
  move through image-label pairs.

### Growing Incomplete Labels

Docked (with the other click modes) as part of **Label manipulations**, this
tool is the fix for a label that covers only *part* of its cell — under-
segmentation of a different kind from
[Splitting Merged Labels](#splitting-merged-labels), where the problem is one
label spanning several cells. Here one cell has one label; the label is simply
too small.

Set **Click mode** to *Grow label to cell (SAM2)* and **left-click** the
label. The cell is segmented with
[SAM2](https://github.com/facebookresearch/sam2) and the label is extended to
the full object, at the clicked timepoint only.

The existing label is what prompts the model: its most interior pixels become
positive points ("this object"), and every neighboring label contributes a
negative point ("not this object"), which is what stops SAM2 returning a whole
clump of touching cells as one. No bounding box is sent — a box drawn around an
incomplete label would tell the model the object ends at the label's current
edge, which is exactly the error being corrected.

Settings:

- **Max growth (px)** — does double duty: how far the label may travel from its
  current boundary, and how much context around it the model is shown. Too
  small clips a genuinely larger cell; too large makes the crop mostly
  background, leaving less resolution on the cell itself (the crop is rescaled
  to 1024x1024 whatever its size). Roughly one cell radius is a good start.
- **Smoothing (px)** — closes small bays and removes thin spurs in the returned
  contour. SAM2 masks are already fairly smooth, so 0-2 is usually right; 0
  keeps the model's boundary exactly.
- **Signal channel** — which channel of a multi-channel raw defines the
  boundary. Use this to grow to a membrane or cytoplasm marker rather than a
  nuclear one.

Notes:

- The label only ever **grows**, and only into **background** — a neighboring
  label is never eaten, whatever the model returns. Re-click after raising
  *Max growth* if it stopped short.
- Runs on **every Z-plane** the label appears on, each segmented from its own
  in-plane image, which suits anisotropic stacks. The label keeps one ID, so
  the planes need no stitching.
- **Ctrl+Z** removes exactly the pixels the last grow added.
- Grows are staged in memory; press **Save and Continue** to write them.
- Works in the **track views** as well: the view supplies only the clicked ID
  and timepoint, and the growth is computed on the source's full-resolution
  slice and raw image.
- Mutually exclusive with the other click modes; the mode persists as you move
  through image-label pairs.

**Requirements.** SAM2 runs in its own environment
(`~/.napari-tmidas/envs/sam2-env`), because the napari environment carries no
PyTorch. Opening the **Batch Crop Anything** widget once creates it and
downloads the checkpoint (~850 MB); until then, clicking to grow reports what
is missing instead of editing. The model is loaded once per session by a
resident worker process, so the **first click pauses for a few seconds** and
every later one costs about **0.2 s per Z-plane**. A GPU is used when
available, otherwise it falls back to CPU (considerably slower).

### Segmenting Missed Cells

Docked (with the other click modes) as part of **Label manipulations**, this
tool is the fix for a **false negative** — a cell the segmentation left out
entirely. There is no label to correct, so none of the other tools apply:
[Growing Incomplete Labels](#growing-incomplete-labels) needs a label to
extend, and painting one by hand is the slow alternative this replaces.

Set **Click mode** to *Add missed cell (SAM2)* and **left-click inside the
cell**. The clicked pixel is the whole prompt:
[SAM2](https://github.com/facebookresearch/sam2) segments the object around it
and the result is written as a **new label with a fresh, globally-unique ID**,
at the clicked timepoint only. Click **background** — a click on an existing
label is refused with a pointer to click-to-grow.

Every label near the click contributes a negative point ("not this object"),
nearest first, and the returned mask is intersected with **background only**.
That combination is what makes the common case safe: a cell often gets missed
*because* it touches a labeled one, and neither the model nor a stray brush
stroke can take a pixel from the neighbor.

Settings:

- **Max radius (px)** — does double duty: how far the new label may reach from
  the pixel you clicked, and how much context around it the model is shown.
  Too small clips a large cell (or one clicked off-center); too large makes the
  crop mostly background, leaving less resolution on the cell itself (the crop
  is rescaled to 1024x1024 whatever its size). Roughly one cell *diameter* is a
  good start, since a click rarely lands dead center.
- **Smoothing (px)** — as for growing: closes small bays and removes thin
  spurs; 0-2 is usually right.
- **Signal channel** — which channel of a multi-channel raw outlines the cell,
  e.g. a membrane marker rather than a nuclear one.
- **Continue through Z** *(on by default)* — for Z-stacks, continue the same
  object onto the planes above and below the clicked one.

Notes:

- A click resolves **one plane**, so on a stack the object is traced through Z
  from there: each plane is prompted with the previous plane's outline and
  segmented from its own in-plane image (which suits anisotropic stacks). The
  walk stops once a plane holds only the cell's **out-of-focus halo** — judged
  by how far the object stands out from the background *on that plane*, since
  the halo is real signal that segments as readily as the cell — and also
  where the cell tapers out, where another label owns the footprint, or where
  the mask suddenly leaks. Every plane keeps the one new ID, so nothing needs
  stitching. Turn *Continue through Z* off to label the clicked plane alone.
- Nothing is written if the model finds no object at the click — the status bar
  says so and suggests clicking nearer the middle of the cell or raising *Max
  radius*.
- **Ctrl+Z** removes the label just added.
- New labels are staged in memory; press **Save and Continue** to write them.
- Works in **2D and in 3D display**. In 3D there is no pixel to pick — napari
  reports only where the view ray hits a *label*, and a missed cell has none —
  so the ray is marched and its **brightest sample** becomes the seed, which is
  exactly the voxel napari's maximum-intensity rendering drew under the cursor.
  Any camera angle works, and a click in 3D lands on the same seed (and
  produces the same label) as the equivalent click in 2D. If that brightest
  sample already belongs to a label, the status bar says which — that cell is
  what you pointed at, and it needs *Grow label to cell* instead.
- Needs the **normal frame view**: turn *Track view* off, since those views
  restack or project Z. On a movie with no Z axis, napari's 3D display stacks
  *time*, so switch back to 2D there.
- Mutually exclusive with the other click modes; the mode persists as you move
  through image-label pairs.

**Requirements.** The same SAM2 environment as
[Growing Incomplete Labels](#growing-incomplete-labels), and the same costs: a
few seconds for the first click of a session, then ~0.2 s per plane.

On synthetic cells with blurred, noisy edges, a click in a missed cell wedged
between two labeled ones recovered it at IoU 0.94 against ground truth (0.87
for a deliberately off-center click), with zero pixels taken from either
neighbor. On a Z-stack the propagation covered all 9 planes of a sphere at IoU
0.93; on a sphere sitting in a taller stack it labeled exactly the 9 planes the
cell occupies and none of the 8 that hold only its halo (IoU 0.66 there — the
edge planes, being mostly halo in-plane, come out wider than the cell).

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
- The *Delete label* / *Relabel label* / *Merge touching neighbors* modes, **Ctrl+Z**
  undo, and **Save Changes and Continue** all work exactly as in the normal
  view; label files stay TZYX. (*Split merged label* is one exception — it needs
  precise source voxels as seeds, so turn *Track view* off for it.)
- Editing stays interactive no matter how long the movie is. The 3-D volume is
  built **once** and then kept up to date in place: a delete or relabel touches
  only the affected labels' bounding boxes, and only that box is re-uploaded to
  the GPU, instead of re-reading the movie and re-uploading the whole texture
  for every click. The data itself is never rewritten until you save — edits
  accumulate as a value map on the lazy array.
- The chosen view persists as you move through image-label pairs and is rebuilt
  over each new pair. Requires a 3-D (TYX) or 4-D (TZYX) label source.

### Delete Low-Intensity Tracks

Docked as part of **Track manipulations**, this tool removes every track
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
1. Set **Click mode** to *Relabel label*
2. Ctrl+click the object to keep (pipettes its ID)
3. Click the split-off fragments to merge them into it (all timepoints at once)
4. Or paint manually with the same label for partial merges
5. Save changes

For a single over-segmented cell, set **Click mode** to *Merge touching
neighbors* instead and click one fragment: every label touching it fuses into the clicked ID, no
pipetting needed (re-click to reach further out). Set its **Apply to** to *All
timepoints* to repair the whole track from that one click, or leave it on
*Clicked timepoint only* when the fragmentation differs frame to frame.

### Splitting Merged Objects

When segmentation under-splits — several touching cells share one ID:
1. Set **Click mode** to *Split merged label*
2. Navigate to the timepoint where the objects are merged
3. Click one point inside each cell (Ctrl+click removes the last seed)
4. Press **Apply split** — the label divides into one region per seed, each
   after the first getting a new ID
5. Repeat at other timepoints as needed, then save changes

### Removing False Positives

When segmentation detects spurious objects:
1. Set **Click mode** to *Delete label* and click each spurious object —
   removed from every timepoint instantly
2. Or use eraser (label = 0) for partial removal
3. Save corrected labels

### Recovering False Negatives

When segmentation misses cells entirely (dim ones, or ones packed against a
detected neighbor):
1. Set **Click mode** to *Add missed cell (SAM2)* and set *Max radius* to
   about one cell diameter
2. Click inside each missed cell; SAM2 segments it into a new label with a
   fresh ID, without touching the labels around it
3. On a Z-stack leave *Continue through Z* on to get the whole object from one
   click
4. Check the result and **Ctrl+Z** any label you do not want
5. **Save and Continue**

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
4. Delete or relabel whole tracks with one click, then save

### Fixing Tracking ID Switches

When a tracked object changes ID partway through a time series:
1. Set **Click mode** to *Relabel label*
2. Ctrl+click the object at a timepoint where it has the correct ID
3. Navigate to a timepoint after the switch and click the object —
   the wrong ID is reassigned to the correct one everywhere
4. Save changes

### Completing Under-Sized Labels

When a segmentation systematically under-covers its objects (thresholding that
clipped dim edges, a model trained on tighter masks):
1. Set **Click mode** to *Grow label to cell (SAM2)* and set *Max growth* to
   about one cell radius
2. Click each under-sized label; SAM2 extends it to the cell boundary
3. Check the result and **Ctrl+Z** any grow you do not want
4. **Save and Continue**

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

### Grow / Add Reports SAM2 Is Missing

**Cause**: The SAM2 environment or its model checkpoint has not been created yet

**Solutions**:
- Open the **Batch Crop Anything** widget once and let it create
  `~/.napari-tmidas/envs/sam2-env` and download the checkpoint (~850 MB)
- Check the console for the worker's error output if it starts and then exits
- The status bar names exactly what is missing (environment, checkpoint, or
  plugin file)

### Grow / Add Is Slow

**Cause**: The model loads on the first click, and cost scales with Z-depth

**Solutions**:
- Expect a few seconds on the first click of a session; later clicks are about
  0.2 s per Z-plane and the model stays resident
- Without a GPU, SAM2 falls back to CPU and is considerably slower
- A label spanning many Z-planes costs proportionally more, since each plane is
  segmented separately

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
