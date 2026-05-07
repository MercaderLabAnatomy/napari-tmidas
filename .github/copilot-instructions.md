
# GitHub Copilot Instructions for napari-tmidas

## Mandatory Environment Activation

Before running any repository analysis, tests, scripts, or Python commands, always activate the `napari-tmidas` mamba environment first.

Use this exact command:
`source ~/miniforge3/etc/profile.d/conda.sh && mamba activate napari-tmidas`

After activation, verify that the active environment is `napari-tmidas`.
If activation fails, stop immediately and report the failure instead of continuing in a different environment.

## Zarr processing functions

When writing or modifying any function that reads a zarr and writes a processed zarr:

- **Preserve the source pyramid structure.** The output zarr must have the same number of pyramid levels as the source. Do not add levels that do not exist in the source; do not remove levels that do.
- If the source has only one level (no pyramid), write the output with `max_layer=0` (i.e. a single level).
- The general rule: **copy everything from the source except the data dimensions/values that the function explicitly modifies.** This includes number of pyramid levels, axes metadata, coordinate transforms, chunk layout, omero channel metadata, and data type.
- When using `ome_zarr.writer.write_image`, always derive `max_layer` from the source: `max_layer = len(source_levels) - 1`.
- Never hard-code `max_layer=4` or any fixed pyramid depth.
