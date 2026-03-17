#!/usr/bin/env bash
set -euo pipefail

# Generate SynthStrip oracle brain masks for normal/no-CSF variants.
# Defaults target the repository test resources.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${REPO_ROOT}/test/res/spgr_unstrip.nii.gz"
OUT_DIR="${REPO_ROOT}/test/res/oracle"
OUT_PREFIX="synthstrip_oracle"
ORACLE_ENV="${REPO_ROOT}/build/.venv-oracle-py311"
ORACLE_FS_HOME="${REPO_ROOT}/build/fs_oracle_home"
ORACLE_PYTHON="3.11"
THREADS="1"
BORDER="1"
VARIANTS="all"
SKIP_INSTALL="0"

usage() {
  cat <<'EOF'
Usage: dev/generate_synthstrip_oracle.bash [options]

Options:
  --input PATH           Input scan (default: test/res/spgr_unstrip.nii.gz)
  --out-dir PATH         Output directory for generated mask files (default: test/res/oracle)
  --out-prefix NAME      Filename prefix (default: synthstrip_oracle)
  --oracle-env PATH      UV oracle venv path (default: build/.venv-oracle-py311)
  --oracle-fs-home PATH  FREESURFER_HOME for oracle runtime (default: build/fs_oracle_home)
  --python VERSION       Python version for oracle env (default: 3.11)
  --threads N            PyTorch CPU threads (default: 1)
  --border MM            SynthStrip border threshold in mm (default: 1)
  --variants MODE        Which variants to generate: all|normal|nocsf (default: all)
  --skip-install         Do not install oracle dependencies
  -h, --help             Show this help

Notes:
  - Variant mapping:
      normal -> synthstrip.<version>.pt
      nocsf  -> synthstrip.nocsf.<version>.pt
  - Filenames:
      <prefix>_nocsf_0.nii.gz
      <prefix>_nocsf_1.nii.gz
  - If dependencies are missing, they are installed via uv pip into --oracle-env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_PATH="$2"
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
    --border)
      BORDER="$2"
      shift 2
      ;;
    --variants)
      VARIANTS="$2"
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

case "${VARIANTS}" in
  all)
    variant_specs=("normal:0" "nocsf:1")
    ;;
  normal)
    variant_specs=("normal:0")
    ;;
  nocsf)
    variant_specs=("nocsf:1")
    ;;
  *)
    echo "Unknown --variants value: ${VARIANTS}" >&2
    echo "Expected one of: all, normal, nocsf" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

if [[ ! -x "${ORACLE_ENV}/bin/python" ]]; then
  echo "Creating oracle env at ${ORACLE_ENV} (python ${ORACLE_PYTHON})"
  uv venv "${ORACLE_ENV}" --python "${ORACLE_PYTHON}"
fi

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  "${ORACLE_ENV}/bin/python" - <<'PY' >/dev/null 2>&1 || uv pip install --python "${ORACLE_ENV}/bin/python" 'torch==2.1.2' 'surfa==0.6.3'
import surfa  # noqa: F401
import torch  # noqa: F401
PY
fi

mkdir -p "${ORACLE_FS_HOME}/models"

required_files=(
  synthstrip.1.pt
  synthstrip.nocsf.1.pt
)

for name in "${required_files[@]}"; do
  src="${REPO_ROOT}/dev/freesurfer/mri_synthstrip/${name}"
  if [[ ! -e "${src}" ]]; then
    echo "Missing FreeSurfer model/resource: ${src}" >&2
    exit 1
  fi
  ln -sf "${src}" "${ORACLE_FS_HOME}/models/${name}"
done

for variant_spec in "${variant_specs[@]}"; do
  IFS=":" read -r variant_name no_csf_flag <<<"${variant_spec}"
  out_mask="${OUT_DIR}/${OUT_PREFIX}_nocsf_${no_csf_flag}.nii.gz"

  cmd=(
    "${ORACLE_ENV}/bin/python"
    "${REPO_ROOT}/dev/freesurfer/mri_synthstrip/mri_synthstrip"
    --image "${INPUT_PATH}"
    --mask "${out_mask}"
    --threads "${THREADS}"
    --border "${BORDER}"
  )

  if [[ "${variant_name}" == "nocsf" ]]; then
    cmd+=(--no-csf)
  fi

  echo "Running oracle for ${variant_name} (nocsf=${no_csf_flag})..."
  FREESURFER_HOME="${ORACLE_FS_HOME}" "${cmd[@]}"
  echo "Wrote brain mask: ${out_mask}"
done
