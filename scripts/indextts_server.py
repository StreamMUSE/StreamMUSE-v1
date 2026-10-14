"""Minimal /v1/audio/speech server around IndexTTS-2.5 (model-upgrade plan, B1).

IndexTTS is not served by SGLang-Omni and needs its own Python 3.11
environment, so this wraps its Python API with the same request shape the other
candidates use. Run it inside the index-tts checkout's uv environment:

    cd <index-tts> && PYTHONPATH=. uv run python <repo>/scripts/indextts_server.py \\
        --model-dir <IndexTTS-2.5 snapshot> --port 8501

POST /v1/audio/speech {"input": text, "ref_audio": path, "seed": int, "duration_factor": float}
returns a WAV; GET /health returns 200 once the model is loaded.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import tempfile
import threading


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    import numpy as np
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(cfg_path=str(args.model_dir / "config.yaml"), model_dir=str(args.model_dir), use_bf16=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *values):  # noqa: A002 - keep the server quiet
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/audio/speech":
                self.send_response(404)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seed = int(body.get("seed", 0))
            with lock, tempfile.TemporaryDirectory() as folder:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                out = Path(folder) / "out.wav"
                tts.infer(spk_audio_prompt=body["ref_audio"], text=body["input"], output_path=str(out), lang="EN",
                          duration_factor=float(body.get("duration_factor", 1.0)), verbose=False)
                data = out.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"IndexTTS-2.5 ready on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
