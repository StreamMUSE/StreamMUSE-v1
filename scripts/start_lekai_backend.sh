#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/src"
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m streammuse.infrastructure.inference.server_lekai
