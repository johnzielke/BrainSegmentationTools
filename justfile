#Just (similar to Makefile), install from https://github.com/casey/just#installation

# Run the full test suite.
test:
    mkdir -p build/model_cache
    BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache uv run pytest

# Generate converted model artifacts under build/converted_models.
convert-models:
    mkdir -p build/model_cache build/converted_models
    BRAIN_SEGMENTATION_TOOLS_MODEL_CACHE_DIR=build/model_cache uv run synthseg-models save_all_converted --output_dir build/converted_models


[parallel]
init: init-freesurfer init-precommit init-python-project


init-freesurfer:
    cd dev && ./init.bash

init-python-project:
    uv sync

@pre-commit-available:
    which pre-commit > /dev/null || { echo "pre-commit is not available on the path. You can install it using 'uv tool install pre-commit'."; exit 1; }

init-precommit: pre-commit-available
    pre-commit install --install-hooks

pre-commit: pre-commit-available
    pre-commit run

# Run pre-commit across the entire repository.
pre-commit-all: pre-commit-available
    pre-commit run -a

qc: pre-commit-all test
