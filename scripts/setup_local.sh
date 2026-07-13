#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

usage() {
  cat <<USAGE
Usage: scripts/setup_local.sh [--recreate] [--skip-torch]

Environment overrides:
  PYTHON_BIN        Python executable to use. Default: python3
  VENV_DIR          Virtualenv path. Default: .venv
  TORCH_VARIANT     cpu, default, or skip. Default: cpu
  TORCH_INDEX_URL   PyTorch package index for cpu mode.
USAGE
}

RECREATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-torch)
      TORCH_VARIANT="skip"
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

cd "${REPO_ROOT}"

if [[ "${RECREATE}" == "1" && -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${TORCH_VARIANT}" == "cpu" ]]; then
  python -m pip install --index-url "${TORCH_INDEX_URL}" torch
elif [[ "${TORCH_VARIANT}" == "default" ]]; then
  python -m pip install torch
elif [[ "${TORCH_VARIANT}" == "skip" ]]; then
  echo "Skipping torch installation."
else
  echo "Invalid TORCH_VARIANT: ${TORCH_VARIANT}. Use cpu, default, or skip." >&2
  exit 2
fi

python -m pip install \
  lightgbm \
  matplotlib \
  numpy \
  pandas \
  pymavlink \
  pyulog \
  pywavelets \
  pyyaml \
  rdp \
  scikit-learn \
  scipy \
  seaborn \
  tqdm \
  tsfresh

ACTIVATE_HOOK="${VENV_DIR}/bin/activate"
PYTHONPATH_LINE="export PYTHONPATH=\"${REPO_ROOT}:${REPO_ROOT}/drone_classifier:${REPO_ROOT}/drone_classifier_svm:${REPO_ROOT}/trajectory_processing:${REPO_ROOT}/drone_classifier_modules/src:\${PYTHONPATH:-}\""

if ! grep -Fq "FIRE_moonshot_classifier PYTHONPATH" "${ACTIVATE_HOOK}"; then
  {
    echo ""
    echo "# FIRE_moonshot_classifier PYTHONPATH"
    echo "${PYTHONPATH_LINE}"
  } >> "${ACTIVATE_HOOK}"
fi

echo ""
echo "Local environment is ready."
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
