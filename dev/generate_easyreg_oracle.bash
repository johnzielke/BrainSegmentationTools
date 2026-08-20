#!/usr/bin/env bash
set -euo pipefail

# Generate EasyReg deformable-only oracle outputs using the vendored model stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${REPO_ROOT}/test/res/spgr_unstrip.nii.gz"
OUT_DIR="${REPO_ROOT}/test/res/oracle"
OUT_PREFIX="easyreg_oracle"
MOVING_PATH="${OUT_DIR}/easyreg_oracle_moving_image.nii.gz"
REFERENCE_PATH_FOR_RUN="${OUT_DIR}/easyreg_oracle_reference_image.nii.gz"
ORACLE_ENV="${REPO_ROOT}/build/.venv-oracle-py310-easyreg"
ORACLE_FS_HOME="${REPO_ROOT}/build/fs_oracle_home"
ORACLE_PYTHON="3.10"
THREADS="1"
SKIP_INSTALL="0"

usage() {
  cat <<'EOF'
Usage: dev/generate_easyreg_oracle.bash [options]

Options:
  --input PATH           Reference input scan (default: test/res/spgr_unstrip.nii.gz)
  --moving PATH          Moving scan to register; if omitted, generate a deterministic deformed copy
  --out-dir PATH         Output directory for generated files (default: test/res/oracle)
  --out-prefix NAME      Filename prefix (default: easyreg_oracle)
  --oracle-env PATH      UV oracle venv path (default: build/.venv-oracle-py310-easyreg)
  --oracle-fs-home PATH  FREESURFER_HOME for oracle runtime (default: build/fs_oracle_home)
  --python VERSION       Python version for oracle env (default: 3.10)
  --threads N            CPU threads for oracle generation (default: 1)
  --skip-install         Do not install oracle dependencies
  -h, --help             Show this help

Outputs:
  <out-prefix>_deformation_field.nii.gz
  <out-prefix>_warped_image.nii.gz

Notes:
  - Skips affine registration and compares only the deformable field stage.
  - Generates a small elastic moving image from the reference image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_PATH="$2"
      shift 2
      ;;
    --moving)
      MOVING_PATH="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --out-prefix)
      OUT_PREFIX="$2"
      shift 2
      ;;
    --oracle-env)
      ORACLE_ENV="$2"
      shift 2
      ;;
    --oracle-fs-home)
      ORACLE_FS_HOME="$2"
      shift 2
      ;;
    --python)
      ORACLE_PYTHON="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Input not found: ${INPUT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [[ ! -x "${ORACLE_ENV}/bin/python" ]]; then
  echo "Creating oracle env at ${ORACLE_ENV} (python ${ORACLE_PYTHON})"
  uv venv "${ORACLE_ENV}" --python "${ORACLE_PYTHON}"
fi

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  "${ORACLE_ENV}/bin/python" - <<'PY' >/dev/null 2>&1 || uv pip install --python "${ORACLE_ENV}/bin/python" 'tensorflow-cpu==2.15.*' 'keras==2.15.*' 'voxelmorph==0.2' 'neurite==0.2' 'surfa==0.6.3' 'torch==2.1.2' 'scipy<1.13' 'nibabel>=5,<6'
import keras  # noqa: F401
import tensorflow  # noqa: F401
import torch  # noqa: F401
import voxelmorph  # noqa: F401
PY
fi

mkdir -p "${ORACLE_FS_HOME}/models"
required_files=(
  easyreg_v10_230103.h5
)

for name in "${required_files[@]}"; do
  src="${REPO_ROOT}/dev/freesurfer/mri_easyreg/${name}"
  if [[ ! -e "${src}" ]]; then
    src="${REPO_ROOT}/dev/freesurfer/mri_synthseg/${name}"
  fi
  if [[ ! -e "${src}" ]]; then
    echo "Missing FreeSurfer model/resource: ${name}" >&2
    exit 1
  fi
  ln -sf "${src}" "${ORACLE_FS_HOME}/models/${name}"
done

REFERENCE_PATH_FOR_RUN="${OUT_DIR}/${OUT_PREFIX}_reference_image.nii.gz"
MOVING_PATH="${OUT_DIR}/${OUT_PREFIX}_moving_image.nii.gz"

out_field="${OUT_DIR}/${OUT_PREFIX}_deformation_field.nii.gz"
out_warped="${OUT_DIR}/${OUT_PREFIX}_warped_image.nii.gz"

cmd=(
  "${ORACLE_ENV}/bin/python"
  "${REPO_ROOT}/dev/generate_easyreg_deformable_oracle.py"
  --reference "${INPUT_PATH}"
  --moving "${MOVING_PATH}"
  --field-out "${out_field}"
  --warped-out "${out_warped}"
  --model-path "${REPO_ROOT}/dev/freesurfer/mri_easyreg/easyreg_v10_230103.h5"
  --threads "${THREADS}"
)

echo "Running EasyReg deformable-only oracle generation..."
FREESURFER_HOME="${ORACLE_FS_HOME}" "${cmd[@]}"
echo "Wrote deformation field: ${out_field}"
echo "Wrote warped image:      ${out_warped}"