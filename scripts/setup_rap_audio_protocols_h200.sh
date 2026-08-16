#!/usr/bin/env bash
set -euo pipefail

ROOT_PREFIX="${ROOT_PREFIX:-/data/home/Andrew.Yang/StreamMUSE}"
ENV_ROOT="${ENV_ROOT:-${ROOT_PREFIX}/envs/rap-audio-protocols}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-${ROOT_PREFIX}/checkouts/rap-audio-protocols}"
ASSET_ROOT="${ASSET_ROOT:-${ROOT_PREFIX}/assets/rap-audio-protocols}"
HF_HOME="${HF_HOME:-/data/home/Andrew.Yang/.cache/huggingface}"
NEMO_CACHE_DIR="${NEMO_CACHE_DIR:-${HF_HOME}/nemo}"
MFA_ROOT_DIR="${MFA_ROOT_DIR:-${ASSET_ROOT}/mfa}"
MANIFEST_PATH="${MANIFEST_PATH:-${ENV_ROOT}/environment_manifest.json}"

UV_BIN="${UV_BIN:-/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
MOSS_PYTHON_BIN="${MOSS_PYTHON_BIN:-/usr/bin/python3.12}"
PYTHON310_BIN="${PYTHON310_BIN:-/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/python}"
if [ -z "${CONDA_BIN:-}" ]; then
  if command -v micromamba >/dev/null 2>&1; then
    CONDA_BIN="$(command -v micromamba)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    echo "error: conda or micromamba is required for the align environment" >&2
    exit 2
  fi
fi

MOSS_REPO_URL="${MOSS_REPO_URL:-https://github.com/OpenMOSS/MOSS-TTS.git}"
TED_REPO_URL="${TED_REPO_URL:-https://github.com/Simon-leong/TED-TTS.git}"
MOSS_REF="${MOSS_REF:-58b20a0d5fcc6766658d50967a90a9d890009a46}"
TED_REF="${TED_REF:-36ffc3e2de346156baa7d60a1749ca4a9365625b}"

MOSS_MODEL_ID="${MOSS_MODEL_ID:-OpenMOSS-Team/MOSS-TTS-v1.5}"
MOSS_MODEL_REVISION="${MOSS_MODEL_REVISION:-cdd3b911b1585e3f2dbc7775ef10f9926f58850a}"
INDEXTTS_MODEL_ID="${INDEXTTS_MODEL_ID:-IndexTeam/IndexTTS-2}"
INDEXTTS_MODEL_REVISION="${INDEXTTS_MODEL_REVISION:-740dcaff396282ffb241903d150ac011cd4b1ede}"
NEMO_SOURCE_TAG="${NEMO_SOURCE_TAG:-v2.7.3}"
NEMO_VERSION="${NEMO_VERSION:-2.7.3}"
MFA_SOURCE_TAG="${MFA_SOURCE_TAG:-v3.4.1}"
MFA_VERSION="${MFA_VERSION:-3.4.1}"
FFMPEG7_VERSION="${FFMPEG7_VERSION:-7.1.1}"
RUBBERBAND_APT_VERSION="${RUBBERBAND_APT_VERSION:-3.3.0+dfsg-2build1}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1+cu128}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.1+cu128}"
FASTER_WHISPER_VERSION="${FASTER_WHISPER_VERSION:-1.2.1}"

MOSS_ENV="${ENV_ROOT}/moss"
TED_ENV="${ENV_ROOT}/ted"
NEMO_ENV="${ENV_ROOT}/nemo"
ALIGN_ENV="${ENV_ROOT}/align"
FFMPEG7_ENV="${ENV_ROOT}/ffmpeg7"
MOSS_CHECKOUT="${CHECKOUT_ROOT}/moss"
TED_CHECKOUT="${CHECKOUT_ROOT}/ted"
RUBBERBAND_ROOT="${ASSET_ROOT}/rubberband"
RUBBERBAND_DOWNLOAD_DIR="${RUBBERBAND_ROOT}/packages"
RUBBERBAND_EXTRACT_ROOT="${RUBBERBAND_ROOT}/rootfs"
RUBBERBAND_BIN="${RUBBERBAND_EXTRACT_ROOT}/usr/bin/rubberband"
RUBBERBAND_LIB_DIR="${RUBBERBAND_EXTRACT_ROOT}/usr/lib/x86_64-linux-gnu"
TED_REFERENCE_RELATIVE_PATH="datasets/Ref/0011_000001.wav"
TED_REFERENCE_SOURCE="${TED_CHECKOUT}/${TED_REFERENCE_RELATIVE_PATH}"
TED_REFERENCE_COPY="${ASSET_ROOT}/ted/0011_000001.wav"

require_executable() {
  local executable="$1"
  local label="$2"
  if [ ! -x "${executable}" ]; then
    echo "error: ${label} is not executable: ${executable}" >&2
    exit 2
  fi
}

clone_at_ref() {
  local url="$1"
  local ref="$2"
  local target="$3"
  if [ ! -d "${target}/.git" ]; then
    git clone --filter=blob:none "${url}" "${target}"
  fi
  git -C "${target}" fetch --tags --prune origin
  git -C "${target}" checkout --detach "${ref}"
  local resolved
  resolved="$(git -C "${target}" rev-parse HEAD)"
  if [ "${resolved}" != "${ref}" ]; then
    echo "error: ${target} resolved to ${resolved}, expected ${ref}" >&2
    exit 2
  fi
}

download_hf_snapshot() {
  local hf_bin="$1"
  local repo_id="$2"
  local revision="$3"
  local output
  local snapshot_path
  output="$(HF_HOME="${HF_HOME}" "${hf_bin}" download "${repo_id}" --revision "${revision}")"
  snapshot_path="${output##*$'\n'}"
  if [ ! -d "${snapshot_path}" ]; then
    echo "error: hf download did not produce a snapshot directory for ${repo_id}: ${snapshot_path}" >&2
    exit 2
  fi
  case "${snapshot_path}" in
    */snapshots/"${revision}") ;;
    *)
      echo "error: ${repo_id} resolved to an unexpected snapshot: ${snapshot_path}" >&2
      exit 2
      ;;
  esac
  printf '%s\n' "${snapshot_path}"
}

require_executable "${UV_BIN}" "uv"
require_executable "${PYTHON_BIN}" "manifest Python"
require_executable "${CONDA_BIN}" "conda/micromamba"
mkdir -p "${ENV_ROOT}" "${CHECKOUT_ROOT}" "${ASSET_ROOT}" "${HF_HOME}" "${NEMO_CACHE_DIR}" "${MFA_ROOT_DIR}"

clone_at_ref "${MOSS_REPO_URL}" "${MOSS_REF}" "${MOSS_CHECKOUT}"
clone_at_ref "${TED_REPO_URL}" "${TED_REF}" "${TED_CHECKOUT}"
git -C "${TED_CHECKOUT}" lfs install --local
git -C "${TED_CHECKOUT}" lfs pull

"${UV_BIN}" venv --python "${MOSS_PYTHON_BIN}" "${MOSS_ENV}"
"${UV_BIN}" pip install \
  --python "${MOSS_ENV}/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -e "${MOSS_CHECKOUT}[torch-runtime]"
"${UV_BIN}" pip install --python "${MOSS_ENV}/bin/python" "faster-whisper==${FASTER_WHISPER_VERSION}"

"${UV_BIN}" venv --python "${PYTHON310_BIN}" "${TED_ENV}"
(
  cd "${TED_CHECKOUT}"
  UV_PROJECT_ENVIRONMENT="${TED_ENV}" "${UV_BIN}" sync --all-extras
)

"${UV_BIN}" venv --python "${PYTHON310_BIN}" "${NEMO_ENV}"
"${UV_BIN}" pip install \
  --python "${NEMO_ENV}/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "torch==${TORCH_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "nemo_toolkit[tts]==${NEMO_VERSION}"

if [ -d "${ALIGN_ENV}/conda-meta" ]; then
  conda_action=install
else
  conda_action=create
fi
"${CONDA_BIN}" "${conda_action}" --yes --prefix "${ALIGN_ENV}" --channel conda-forge \
  "python=3.10" \
  "montreal-forced-aligner=${MFA_VERSION}"

if [ -d "${FFMPEG7_ENV}/conda-meta" ]; then
  conda_action=install
else
  conda_action=create
fi
"${CONDA_BIN}" "${conda_action}" --yes --prefix "${FFMPEG7_ENV}" --channel conda-forge \
  "ffmpeg=${FFMPEG7_VERSION}" \
  fftw \
  libsamplerate \
  libsndfile

if [ ! -x "${RUBBERBAND_BIN}" ]; then
  if ! command -v apt >/dev/null 2>&1; then
    echo "error: apt is required to download rubberband-cli without admin access" >&2
    exit 2
  fi
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "error: dpkg-deb is required to extract rubberband-cli without admin access" >&2
    exit 2
  fi
  mkdir -p "${RUBBERBAND_DOWNLOAD_DIR}" "${RUBBERBAND_EXTRACT_ROOT}"
  (
    cd "${RUBBERBAND_DOWNLOAD_DIR}"
    apt download \
      "rubberband-cli=${RUBBERBAND_APT_VERSION}" \
      "librubberband2=${RUBBERBAND_APT_VERSION}"
  )
  rubberband_deb_count=0
  for deb_path in "${RUBBERBAND_DOWNLOAD_DIR}"/*.deb; do
    if [ ! -f "${deb_path}" ]; then
      continue
    fi
    dpkg-deb -x "${deb_path}" "${RUBBERBAND_EXTRACT_ROOT}"
    rubberband_deb_count=$((rubberband_deb_count + 1))
  done
  if [ "${rubberband_deb_count}" -lt 2 ]; then
    echo "error: expected rubberband-cli and librubberband2 debs in ${RUBBERBAND_DOWNLOAD_DIR}" >&2
    exit 2
  fi
fi

require_executable "${MOSS_ENV}/bin/hf" "MOSS hf CLI"
require_executable "${TED_ENV}/bin/hf" "TED hf CLI"
require_executable "${NEMO_ENV}/bin/python" "NeMo Python"
require_executable "${ALIGN_ENV}/bin/mfa" "MFA CLI"
require_executable "${FFMPEG7_ENV}/bin/ffmpeg" "FFmpeg 7 CLI"
require_executable "${RUBBERBAND_BIN}" "Rubber Band CLI"

FFMPEG7_RESOLVED_VERSION="$(
  LD_LIBRARY_PATH="${FFMPEG7_ENV}/lib:${LD_LIBRARY_PATH:-}" \
    "${FFMPEG7_ENV}/bin/ffmpeg" -version 2>&1
)"
FFMPEG7_RESOLVED_VERSION="${FFMPEG7_RESOLVED_VERSION%%$'\n'*}"
case "${FFMPEG7_RESOLVED_VERSION}" in
  *"${FFMPEG7_VERSION}"*) ;;
  *)
    echo "error: expected FFmpeg ${FFMPEG7_VERSION}, got ${FFMPEG7_RESOLVED_VERSION}" >&2
    exit 2
    ;;
esac

RUBBERBAND_RESOLVED_VERSION="$(
  PATH="${FFMPEG7_ENV}/bin:${PATH}" \
  LD_LIBRARY_PATH="${RUBBERBAND_LIB_DIR}:${FFMPEG7_ENV}/lib:${LD_LIBRARY_PATH:-}" \
    "${RUBBERBAND_BIN}" --version
)"
RUBBERBAND_RESOLVED_VERSION="${RUBBERBAND_RESOLVED_VERSION%%$'\n'*}"
case "${RUBBERBAND_RESOLVED_VERSION}" in
  *"3.3.0"*) ;;
  *)
    echo "error: expected Rubber Band 3.3.0, got ${RUBBERBAND_RESOLVED_VERSION}" >&2
    exit 2
    ;;
esac

MOSS_SNAPSHOT_PATH="$(download_hf_snapshot "${MOSS_ENV}/bin/hf" "${MOSS_MODEL_ID}" "${MOSS_MODEL_REVISION}")"
INDEXTTS_SNAPSHOT_PATH="$(download_hf_snapshot "${TED_ENV}/bin/hf" "${INDEXTTS_MODEL_ID}" "${INDEXTTS_MODEL_REVISION}")"

NEMO_RUNTIME_JSON="$(HF_HOME="${HF_HOME}" NEMO_CACHE_DIR="${NEMO_CACHE_DIR}" "${NEMO_ENV}/bin/python" - <<'PY'
import json
from importlib.metadata import version

import torch
import torchaudio
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

nemo_version = version("nemo_toolkit")
if nemo_version != "2.7.3":
    raise RuntimeError(f"expected nemo_toolkit 2.7.3, got {nemo_version}")
if "+cu128" not in torch.__version__ or "+cu128" not in torchaudio.__version__:
    raise RuntimeError(
        f"expected cu128 torch wheels, got torch={torch.__version__} torchaudio={torchaudio.__version__}"
    )
if torch.version.cuda != "12.8":
    raise RuntimeError(f"expected CUDA 12.8 runtime, got {torch.version.cuda}")

model_specs = (
    (FastPitchModel, "tts_en_fastpitch"),
    (HifiGanModel, "tts_en_hifigan"),
)
downloaded_models = []
for model_class, model_name in model_specs:
    model = model_class.from_pretrained(model_name=model_name, map_location="cpu")
    downloaded_models.append(model_name)
    del model

print(
    json.dumps(
        {
            "cuda_version": torch.version.cuda,
            "models": downloaded_models,
            "nemo_toolkit_version": nemo_version,
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
        },
        sort_keys=True,
    )
)
PY
)"

MFA_ROOT_DIR="${MFA_ROOT_DIR}" "${ALIGN_ENV}/bin/mfa" model download acoustic english_us_arpa
MFA_ROOT_DIR="${MFA_ROOT_DIR}" "${ALIGN_ENV}/bin/mfa" model download dictionary english_us_arpa
MFA_RESOLVED_VERSION="$("${ALIGN_ENV}/bin/mfa" version)"
case "${MFA_RESOLVED_VERSION}" in
  *"${MFA_VERSION}"*) ;;
  *)
    echo "error: expected MFA ${MFA_VERSION}, got ${MFA_RESOLVED_VERSION}" >&2
    exit 2
    ;;
esac

if [ ! -f "${TED_REFERENCE_SOURCE}" ]; then
  echo "error: missing TED reference: ${TED_REFERENCE_SOURCE}" >&2
  exit 2
fi
mkdir -p "$(dirname "${TED_REFERENCE_COPY}")"
if [ ! -f "${TED_REFERENCE_COPY}" ] || ! cmp -s "${TED_REFERENCE_SOURCE}" "${TED_REFERENCE_COPY}"; then
  cp "${TED_REFERENCE_SOURCE}" "${TED_REFERENCE_COPY}"
fi
if ! cmp -s "${TED_REFERENCE_SOURCE}" "${TED_REFERENCE_COPY}"; then
  echo "error: copied TED reference does not match its source" >&2
  exit 2
fi

UV_VERSION="$("${UV_BIN}" --version)"
CONDA_VERSION="$("${CONDA_BIN}" --version)"
export ROOT_PREFIX ENV_ROOT CHECKOUT_ROOT ASSET_ROOT HF_HOME NEMO_CACHE_DIR MFA_ROOT_DIR MANIFEST_PATH
export UV_BIN UV_VERSION PYTHON_BIN MOSS_PYTHON_BIN PYTHON310_BIN CONDA_BIN CONDA_VERSION
export MOSS_REPO_URL TED_REPO_URL MOSS_REF TED_REF MOSS_CHECKOUT TED_CHECKOUT
export MOSS_MODEL_ID MOSS_MODEL_REVISION MOSS_SNAPSHOT_PATH
export INDEXTTS_MODEL_ID INDEXTTS_MODEL_REVISION INDEXTTS_SNAPSHOT_PATH
export NEMO_SOURCE_TAG NEMO_VERSION NEMO_RUNTIME_JSON MFA_SOURCE_TAG MFA_VERSION MFA_RESOLVED_VERSION
export FFMPEG7_VERSION FFMPEG7_ENV FFMPEG7_RESOLVED_VERSION
export RUBBERBAND_APT_VERSION RUBBERBAND_BIN RUBBERBAND_LIB_DIR RUBBERBAND_RESOLVED_VERSION
export FASTER_WHISPER_VERSION
export TED_REFERENCE_RELATIVE_PATH TED_REFERENCE_SOURCE TED_REFERENCE_COPY

"${PYTHON_BIN}" <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


env_root = Path(os.environ["ENV_ROOT"])
moss_checkout = Path(os.environ["MOSS_CHECKOUT"])
ted_checkout = Path(os.environ["TED_CHECKOUT"])
reference_source = Path(os.environ["TED_REFERENCE_SOURCE"])
reference_copy = Path(os.environ["TED_REFERENCE_COPY"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
nemo_runtime = json.loads(os.environ["NEMO_RUNTIME_JSON"])

payload = {
    "schema_version": "streammuse.rap_audio_protocols.environment_manifest.v1",
    "root_prefix": os.environ["ROOT_PREFIX"],
    "hf_home": os.environ["HF_HOME"],
    "nemo_cache_dir": os.environ["NEMO_CACHE_DIR"],
    "mfa_root_dir": os.environ["MFA_ROOT_DIR"],
    "tools": {
        "uv": {"path": os.environ["UV_BIN"], "version": os.environ["UV_VERSION"]},
        "manifest_python": os.environ["PYTHON_BIN"],
        "moss_python": os.environ["MOSS_PYTHON_BIN"],
        "python_3_10": os.environ["PYTHON310_BIN"],
        "conda": {"path": os.environ["CONDA_BIN"], "version": os.environ["CONDA_VERSION"]},
    },
    "environments": {
        **{
            name: {
                "path": str((env_root / name).resolve()),
                "python": str((env_root / name / "bin" / "python").resolve()),
            }
            for name in ("moss", "ted", "nemo", "align")
        },
        "ffmpeg7": {
            "path": str((env_root / "ffmpeg7").resolve()),
            "ffmpeg": str((env_root / "ffmpeg7" / "bin" / "ffmpeg").resolve()),
        },
    },
    "repositories": {
        "moss": {
            "path": str(moss_checkout.resolve()),
            "url": os.environ["MOSS_REPO_URL"],
            "requested_ref": os.environ["MOSS_REF"],
            "resolved_commit": git_head(moss_checkout),
        },
        "ted": {
            "path": str(ted_checkout.resolve()),
            "url": os.environ["TED_REPO_URL"],
            "requested_ref": os.environ["TED_REF"],
            "resolved_commit": git_head(ted_checkout),
        },
    },
    "packages": {
        "nemo_toolkit": {
            "source_tag": os.environ["NEMO_SOURCE_TAG"],
            "package_version": os.environ["NEMO_VERSION"],
        },
        "montreal_forced_aligner": {
            "source_tag": os.environ["MFA_SOURCE_TAG"],
            "package_version": os.environ["MFA_VERSION"],
        },
        "faster_whisper": {
            "package_version": os.environ["FASTER_WHISPER_VERSION"],
        },
        "ffmpeg": {
            "package_version": os.environ["FFMPEG7_VERSION"],
            "resolved_version": os.environ["FFMPEG7_RESOLVED_VERSION"],
        },
        "rubberband": {
            "apt_package_version": os.environ["RUBBERBAND_APT_VERSION"],
            "binary_path": str(Path(os.environ["RUBBERBAND_BIN"]).resolve()),
            "resolved_version": os.environ["RUBBERBAND_RESOLVED_VERSION"],
        },
    },
    "models": {
        "moss_tts": {
            "repo_id": os.environ["MOSS_MODEL_ID"],
            "revision": os.environ["MOSS_MODEL_REVISION"],
            "snapshot_path": os.environ["MOSS_SNAPSHOT_PATH"],
        },
        "index_tts_2": {
            "repo_id": os.environ["INDEXTTS_MODEL_ID"],
            "revision": os.environ["INDEXTTS_MODEL_REVISION"],
            "snapshot_path": os.environ["INDEXTTS_SNAPSHOT_PATH"],
        },
        "nemo": nemo_runtime["models"],
        "nemo_runtime": nemo_runtime,
        "mfa": {
            "acoustic": "english_us_arpa",
            "dictionary": "english_us_arpa",
            "resolved_version": os.environ["MFA_RESOLVED_VERSION"],
        },
    },
    "ted_reference": {
        "relative_source_path": os.environ["TED_REFERENCE_RELATIVE_PATH"],
        "source_path": str(reference_source.resolve()),
        "source_sha256": file_sha256(reference_source),
        "copied_path": str(reference_copy.resolve()),
        "copied_sha256": file_sha256(reference_copy),
    },
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Wrote ${MANIFEST_PATH}"
