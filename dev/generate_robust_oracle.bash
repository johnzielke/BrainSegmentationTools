#!/usr/bin/env bash
set -euo pipefail

# Generate SynthSeg oracle outputs for all robust/parc combinations.
# Defaults target the repository test resources.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${REPO_ROOT}/test/res/spgr_unstrip.nii.gz"
OUT_DIR="${REPO_ROOT}/test/res"
OUT_PREFIX="synthseg_oracle"
ORACLE_ENV="${REPO_ROOT}/build/.venv-oracle-py311"
ORACLE_FS_HOME="${REPO_ROOT}/build/fs_oracle_home"
ORACLE_PYTHON="3.11"
THREADS="1"
SKIP_INSTALL="0"

usage() {
  cat <<'EOF'
Usage: dev/generate_robust_oracle.bash [options]

Options:
  --input PATH           Input scan (default: test/res/spgr_unstrip.nii.gz)
  --out-dir PATH         Output directory for all generated files (default: test/res)
  --out-prefix NAME      Filename prefix (default: synthseg_oracle)
  --oracle-env PATH      UV oracle venv path (default: build/.venv-oracle-py311)
  --oracle-fs-home PATH  FREESURFER_HOME for oracle runtime (default: build/fs_oracle_home)
  --python VERSION       Python version for oracle env (default: 3.11)
  --threads N            CPU threads for FreeSurfer script (default: 1)
  --skip-install         Do not install oracle dependencies
  -h, --help             Show this help

Notes:
  - Runs all 4 combinations:
      robust=0/parc=0
      robust=0/parc=1
      robust=1/parc=0
      robust=1/parc=1
  - Each run emits a segmentation NIfTI and QC CSV.
  - Filenames:
      <prefix>_robust_<0|1>_parc_<0|1>.nii.gz
      <prefix>_robust_<0|1>_parc_<0|1>_qc.csv
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
  "${ORACLE_ENV}/bin/python" - <<'PY' >/dev/null 2>&1 || uv pip install --python "${ORACLE_ENV}/bin/python" 'tensorflow-cpu==2.15.*' 'surfa==0.6.3'
import tensorflow  # noqa: F401
import surfa  # noqa: F401
PY
fi

mkdir -p "${ORACLE_FS_HOME}/models"
ln -sf "${REPO_ROOT}/dev/freesurfer/distribution/FreeSurferColorLUT.txt" "${ORACLE_FS_HOME}/FreeSurferColorLUT.txt"

required_files=(
  synthseg_2.0.h5
  synthseg_robust_2.0.h5
  synthseg_parc_2.0.h5
  synthseg_qc_2.0.h5
  synthseg_segmentation_labels_2.0.npy
  synthseg_denoiser_labels_2.0.npy
  synthseg_parcellation_labels.npy
  synthseg_qc_labels_2.0.npy
  synthseg_segmentation_names_2.0.npy
  synthseg_parcellation_names.npy
  synthseg_qc_names_2.0.npy
  synthseg_topological_classes_2.0.npy
)

for name in "${required_files[@]}"; do
  src="${REPO_ROOT}/dev/freesurfer/mri_synthseg/${name}"
  if [[ ! -e "${src}" ]]; then
    echo "Missing FreeSurfer model/resource: ${src}" >&2
    exit 1
  fi
  ln -sf "${src}" "${ORACLE_FS_HOME}/models/${name}"
done

for robust in 0 1; do
  for parc in 0 1; do
    combo_tag="robust_${robust}_parc_${parc}"
    out_seg="${OUT_DIR}/${OUT_PREFIX}_${combo_tag}.nii.gz"
    out_qc="${OUT_DIR}/${OUT_PREFIX}_${combo_tag}_qc.csv"

    cmd=(
      "${ORACLE_ENV}/bin/python"
      "${REPO_ROOT}/dev/freesurfer/mri_synthseg/mri_synthseg"
      --i "${INPUT_PATH}"
      --o "${out_seg}"
      --qc "${out_qc}"
      --cpu
      --threads "${THREADS}"
      --noaddctab
    )

    if [[ "${robust}" == "1" ]]; then
      cmd+=(--robust)
    fi
    if [[ "${parc}" == "1" ]]; then
      cmd+=(--parc)
    fi

    echo "Running oracle for ${combo_tag}..."
    FREESURFER_HOME="${ORACLE_FS_HOME}" "${cmd[@]}"
    echo "Wrote segmentation: ${out_seg}"
    echo "Wrote QC CSV:      ${out_qc}"
  done
done
