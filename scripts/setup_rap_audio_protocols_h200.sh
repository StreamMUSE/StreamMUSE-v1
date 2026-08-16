#!/usr/bin/env bash
set -euo pipefail

ROOT_PREFIX="${ROOT_PREFIX:-/data/home/Andrew.Yang/StreamMUSE}"
ENV_ROOT="${ENV_ROOT:-${ROOT_PREFIX}/envs/rap-audio-protocols}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-${ROOT_PREFIX}/checkouts/rap-audio-protocols}"
HF_HOME="${HF_HOME:-${ROOT_PREFIX}/hf-cache}"
MANIFEST_PATH="${MANIFEST_PATH:-${ENV_ROOT}/environment_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MOSS_REPO_URL="${MOSS_REPO_URL:-https://github.com/OpenMOSS/MOSS-TTS.git}"
TED_REPO_URL="${TED_REPO_URL:-https://github.com/Simon-leong/TED-TTS.git}"
NEMO_REPO_URL="${NEMO_REPO_URL:-https://github.com/NVIDIA/NeMo.git}"
MFA_REPO_URL="${MFA_REPO_URL:-https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git}"

MOSS_REF="${MOSS_REF:-main}"
TED_REF="${TED_REF:-main}"
NEMO_REF="${NEMO_REF:-main}"
MFA_REF="${MFA_REF:-main}"

mkdir -p "${ENV_ROOT}" "${CHECKOUT_ROOT}" "${HF_HOME}"

clone_or_update() {
  local url="$1"
  local ref="$2"
  local target="$3"
  if [ ! -d "${target}/.git" ]; then
    git clone --filter=blob:none "${url}" "${target}"
  fi
  git -C "${target}" fetch --tags origin
  git -C "${target}" checkout "${ref}"
}

create_env() {
  local name="$1"
  local requirements="$2"
  local env_dir="${ENV_ROOT}/${name}"
  "${PYTHON_BIN}" -m venv "${env_dir}"
  # shellcheck disable=SC1091
  source "${env_dir}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  if [ -n "${requirements}" ]; then
    python -m pip install ${requirements}
  fi
  deactivate
}

clone_or_update "${MOSS_REPO_URL}" "${MOSS_REF}" "${CHECKOUT_ROOT}/moss"
clone_or_update "${TED_REPO_URL}" "${TED_REF}" "${CHECKOUT_ROOT}/ted"
clone_or_update "${NEMO_REPO_URL}" "${NEMO_REF}" "${CHECKOUT_ROOT}/nemo"
clone_or_update "${MFA_REPO_URL}" "${MFA_REF}" "${CHECKOUT_ROOT}/align"

create_env moss "'numpy scipy huggingface_hub'"
create_env ted "'numpy scipy soundfile'"
create_env nemo "'numpy scipy librosa'"
create_env align "'montreal-forced-aligner rubberband-cli'"

export ENV_ROOT CHECKOUT_ROOT HF_HOME MANIFEST_PATH
python <<'PY'
import json
import os
import subprocess
from pathlib import Path

env_root = Path(os.environ["ENV_ROOT"])
checkout_root = Path(os.environ["CHECKOUT_ROOT"])
manifest_path = Path(os.environ["MANIFEST_PATH"])

def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

payload = {
    "schema_version": "streammuse.rap_audio_protocols.environment_manifest.v1",
    "root_prefix": str(env_root.parent.parent),
    "hf_home": os.environ["HF_HOME"],
    "environments": {
        name: {
            "path": str((env_root / name).resolve()),
            "python": str((env_root / name / "bin" / "python").resolve()),
        }
        for name in ("moss", "ted", "nemo", "align")
    },
    "repositories": {
        "moss": {
            "path": str((checkout_root / "moss").resolve()),
            "url": os.environ.get("MOSS_REPO_URL", ""),
            "resolved_commit": git_head(checkout_root / "moss"),
        },
        "ted": {
            "path": str((checkout_root / "ted").resolve()),
            "url": os.environ.get("TED_REPO_URL", ""),
            "resolved_commit": git_head(checkout_root / "ted"),
        },
        "nemo": {
            "path": str((checkout_root / "nemo").resolve()),
            "url": os.environ.get("NEMO_REPO_URL", ""),
            "resolved_commit": git_head(checkout_root / "nemo"),
        },
        "align": {
            "path": str((checkout_root / "align").resolve()),
            "url": os.environ.get("MFA_REPO_URL", ""),
            "resolved_commit": git_head(checkout_root / "align"),
        },
    },
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "Wrote ${MANIFEST_PATH}"
