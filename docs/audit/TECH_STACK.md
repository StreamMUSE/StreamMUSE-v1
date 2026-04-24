# TECH_STACK.md — Verified Library Versions

**Audited commit:** `05fc2fa` on branch `new_system_stanley`
**Audit date:** 2026-04-24
**Source of truth:** `pyproject.toml` (declared minima) and `uv.lock` (pinned).

---

## Runtime

| Item | Declared | Pinned (uv.lock) | Notes |
|---|---|---|---|
| Python | `>=3.10` | — | Dev machine observed: 3.12.2; `.python-version` pins 3.10 |
| Build backend | `setuptools>=61.0` | — | |
| Package manager | `uv` | — | `uv sync` is the install path |
| Entry point | `streammuse-cli` | — | → `streammuse.presentation.cli.cli:main` |

---

## Core Framework (HTTP / ASGI)

| Package | Declared | Pinned | Used For |
|---|---|---|---|
| fastapi | `>=0.116.1` | 0.116.1 (or later per resolver) | Inference server + fake server |
| uvicorn | `>=0.35.0` | 0.35.0 | ASGI runtime |
| requests | (shadow-imported) | 2.32.x | `HttpInferenceClient` POSTs to inference server |
| pydantic | `>=2.11.7` | 2.11.7 | Request/response validation, config dataclasses |

> **Note:** `requests` is imported by `infrastructure/inference/http_client.py` but not explicitly declared in `pyproject.toml`. It's pulled transitively (e.g. via `wandb`, `transformers`). This is a **shadow dependency**; listed in `progress.txt` KNOWN BUGS.

---

## Machine Learning

| Package | Declared | Pinned | Used For |
|---|---|---|---|
| torch | `>=2.5.0` | 2.7.1 (macOS) / 2.9.1+cu128 (linux/win) | Model inference |
| torchvision | `>=0.22.0` | 0.22.1 / 0.24.1 | Transitive |
| transformers | (local editable) | `transformers/` path install | Custom RoFormer positional encoding |
| pytorch-lightning | `>=2.5.1.post0` | 2.5.2 | Checkpoint loading for Stanley |
| lightning | `>=2.5.2` | 2.5.2 | Umbrella package |
| deepspeed | `>=0.17.1` | — | Training dep; not used at inference time |
| tokenizers | `>=0.21.1` | — | HuggingFace tokenizers |
| tensorflow | `>=2.19.0` | 2.19.0 | **Declared but not imported in `src/streammuse/`** — candidate for removal |

---

## Music / MIDI / Audio

| Package | Declared | Pinned | Used For |
|---|---|---|---|
| mido | `>=1.3.3` | 1.3.3 | MIDI device I/O, audio output port |
| python-rtmidi | `>=1.5.8` | 1.5.8 | mido's backend on macOS |
| pretty-midi | `>=0.2.10` | 0.2.10 | MIDI file recording |
| miditok | `>=3.0.5.post1` | 3.0.5.post1 | Tokenization utilities |
| music21 | `>=9.7.1` | 9.7.1 | Music analysis helpers |
| mir-eval | `>=0.8.2` | — | Evaluation metrics (analysis scripts) |

---

## I/O & Input

| Package | Declared | Pinned | Used For |
|---|---|---|---|
| pynput | `>=1.8.1` | 1.8.1 | Keyboard input source |

---

## Logging / Observability / Training Infrastructure

> These dependencies are declared and present in `uv.lock` but are **not used by the core inference/service code**. They are remnants of the training workflow (which lives elsewhere) or analysis tooling.

| Package | Pinned | Status |
|---|---|---|
| tensorboard | 2.19.0 | Not used at runtime |
| tensorboardx | 2.6.4 | Not used at runtime |
| torch-tb-profiler | 0.4.3 | Not used at runtime |
| wandb | 0.20.1 | Not used at runtime |
| joblib | 1.5.1 | Not used at runtime |
| nvitop | (as declared) | Not used at runtime |
| matplotlib | 3.10.3 | Used by analysis scripts only |
| ipykernel | (as declared) | Dev-only (Jupyter) |

Candidates for a future dependency diet; deliberately left in place for now to avoid breaking analysis workflows. See `progress.txt` KNOWN BUGS.

---

## Testing

| Package | Declared | Pinned | Used For |
|---|---|---|---|
| pytest | `>=8.4.1` | — | Test runner (unit + integration) |

---

## Local Editable Install

**`transformers/`** — Modified HuggingFace Transformers, installed via uv source:

```toml
transformers = { path = "transformers", editable = true }
```

Contains a custom RoFormer positional-encoding implementation required by `stanley_stack/m2a_transformer.py`. Replacing with upstream `transformers` will break the Stanley pathway.

---

## PyTorch Index Override

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Linux/Windows pull CUDA 12.8 wheels; macOS uses the default PyPI CPU/MPS wheels.

---

## External Assets (Not in pyproject)

| Asset | Size | Tracked | Notes |
|---|---|---|---|
| `FluidR3Mono_GM.sf3` | 23.7 MB | No (untracked) | General MIDI SoundFont for audio playback. Local-only. |
| `prompts/*/*.mid` | small | Yes | Injection melody/accompaniment prompts, organized by key |
| `transformers/` | bulk | Yes | Local editable install (above) |

---

## Platform Support (Observed)

| Platform | Status |
|---|---|
| macOS (Apple Silicon) | Supported; `requirements.txt` comments out CUDA-specific packages for Mac |
| Linux (CUDA 12.8) | Primary production target |
| Windows | Should work via the pytorch-cu128 index, untested in audit |
