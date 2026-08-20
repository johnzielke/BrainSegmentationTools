#!/usr/bin/env bash
set -euo pipefail

echo "==> Preparing local caches..."
mkdir -p build/model_cache .uv

echo "==> Installing pre-commit via uv..."
uv tool install pre-commit
export PATH="${HOME}/.local/bin:${PATH}"

echo "==> Running repo initialization via just..."
if [ -f README.md ]; then
  just init-python-project init-precommit
else
  echo "==> README.md is missing; skipping 'just init-python-project'"
  just init-precommit
fi

echo "==> Initializing FreeSurfer assets via just (best effort)..."
if ! just init-freesurfer; then
  echo "==> FreeSurfer init failed; you can retry later with 'just init-freesurfer'"
fi

echo "==> Devcontainer bootstrap complete"
