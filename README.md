# BrainSegmentationTools

`brain-segmentation-tools` is a PyTorch-based CLI and Python package for running brain segmentation and skull stripping on volumetric scans.
It provides an alternative way to run [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) models without FreeSurfer dependencies (Synthseg is implemented in PyTorch), and focuses on performance (especially on GPU), batch processing, and to be called as a library from other Python applications.
When using this library, please give appropriate credit to the original SynthSeg and SynthStrip papers (see linked websites).
This library does NOT guarantee exact matching outputs to the original FreeSurfer-based implementations, caused by different library versions, floating point precision, and other implementation details. 
Tests show a dice overlap to the original implementation of 0.985-0.995 (see `test/test_optional_synthseg_integration_matrix.py`), but your mileage may vary based on the input data and hardware.

The repository provides:

- `synthseg`: whole-brain segmentation, optional cortical parcellation, optional QC scores, and optional brain-mask generation
- `synthseg-models`: model inspection and conversion utilities
- automatic model caching with hash validation
- a test and development workflow built around `uv` and `just`

The runtime defaults to CUDA and the package depends on CUDA-oriented libraries, but the inference code can also run on CPU by setting `--device=cpu`.

## What It Does

The main CLI wraps two model families:

- **SynthSeg** for segmentation
- **SynthStrip** for brain-mask generation

Common options supported by the `Application` CLI include:

- `--parcellation=True` to add cortical parcellation
- `--robust=True` to use the robust SynthSeg v2.0 model
- `--qc=path/to/qc.csv` to write per-subject QC scores
- `--brain_mask_out=...` to save a SynthStrip mask
- `--ct=True` to apply CT intensity clipping before inference
- `--device=cpu` or `--device=cuda`
- `--skip_existing=True` to avoid recomputing outputs that already exist

## Installation

This repository uses `uv`.

```bash
uv sync
mkdir -p build/model_cache
export BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache
```

The `BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR` environment variable is recommended in this repo so downloaded and converted model artifacts stay inside the workspace.

## Quick Start

Run SynthSeg on a single NIfTI volume:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg run path/to/scan.nii.gz \
  --segmentation_out=build/out/scan_seg.nii.gz
```

Run segmentation and save a brain mask:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg run path/to/scan.nii.gz \
  --segmentation_out=build/out/scan_seg.nii.gz \
  --brain_mask_out=build/out/scan_mask.nii.gz
```

Enable parcellation, robust mode, and QC output:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg --parcellation=True --robust=True \
  --qc=build/out/qc.csv \
  run path/to/scan.nii.gz \
  --segmentation_out=build/out/scan_seg.nii.gz
```

Run on CPU explicitly:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg --device=cpu run path/to/scan.nii.gz \
  --segmentation_out=build/out/scan_seg.nii.gz
```

## Batch And Directory Inputs

`Application.run(...)` accepts:

- a single file path
- a directory path
- a `.txt` file containing one input path per line

When the input is a directory, the CLI recursively finds `*.nii` and `*.nii.gz` files and mirrors the relative directory structure into the output directory.

Example:

```bash
mkdir -p build/seg build/mask

BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg run data/study_a \
  --segmentation_out=build/seg \
  --brain_mask_out=build/mask
```

If you pass a `.txt` list and want output paths to be derived relative to a dataset root, use `--data_root`:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg run inputs.txt \
  --data_root=data \
  --segmentation_out=build/seg
```

All saved images are written as `.nii.gz`.

## Model Caching And Conversion

Models are resolved through `ModelManager`.

- In a development checkout, local model sources under `dev/freesurfer` are used when available.
- Otherwise the package downloads converted `.pt` checkpoints into the model cache directory.
- Cached files are hash-checked and automatically re-downloaded if corruption is detected.

List configured model identifiers:

```bash
uv run synthseg-models list_models
```

Export all configured models into `build/converted_models`:

```bash
mkdir -p build/model_cache build/converted_models
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache \
uv run synthseg-models save_all_converted --output_dir=build/converted_models
```

Convert one model explicitly:

```bash
uv run synthseg-models convert_h5_to_pt \
  --model_name=synthseg \
  --model_type=segmentation \
  --version=2.0 \
  --output_path=build/converted_models/synthseg_segmentation_2.0.pt
```

## Python API

```python
from brain_segmentation_tools.app import Application

app = Application(
    version="v2.0",
    parcellation=True,
    device="cuda",
    no_compile=True,
)

app.run(
    input_paths="path/to/scan.nii.gz",
    segmentation_out="build/out/scan_seg.nii.gz",
    brain_mask_out="build/out/scan_mask.nii.gz",
)
```

## Development

Common tasks are defined in [`justfile`](/home/jz1079/projects/BrainSegmentationTools/justfile):

```bash
just init
just test
just test-slow
just test-all
just convert-models
```

The default test command already uses:

```bash
BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache uv run pytest
```

## Notes

- The default application version is `v2.0`.
- `robust=True` is only supported for `v2.0`.
- QC output requires `segmentation_out`.
- Directory discovery currently looks for `.nii` and `.nii.gz` inputs.
