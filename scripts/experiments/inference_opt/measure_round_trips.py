"""Time round trips to the render server, to record the link's state next to a measurement.

    uv run python scripts/experiments/inference_opt/measure_round_trips.py --url http://127.0.0.1:18620

Reports the median of ``-n`` tries (default 5) for: a GET on a new connection,
the same GET on a reused connection, and a WebSocket handshake. The handshake
opens a socket and closes it without sending ``start``, so no session is created.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

import httpx


def measure(url: str, count: int) -> dict:
    from websockets.sync.client import connect

    base = url.rstrip("/")
    path = f"{base}/v1/rap/capabilities"
    new, reused, handshake = [], [], []
    for _ in range(count):
        started = time.perf_counter()
        httpx.get(path, timeout=10).raise_for_status()
        new.append((time.perf_counter() - started) * 1000)
    with httpx.Client(timeout=10) as client:
        client.get(path).raise_for_status()
        for _ in range(count):
            started = time.perf_counter()
            client.get(path).raise_for_status()
            reused.append((time.perf_counter() - started) * 1000)
    socket_url = ("wss://" + base[len("https://"):]) if base.startswith("https://") else "ws://" + base[len("http://"):]
    for _ in range(count):
        started = time.perf_counter()
        socket = connect(f"{socket_url}/v1/rap/sessions/probe-{uuid.uuid4().hex}", open_timeout=10)
        handshake.append((time.perf_counter() - started) * 1000)
        socket.close()
    summary = lambda values: {"median_ms": round(statistics.median(values), 1),
                              "min_ms": round(min(values), 1), "max_ms": round(max(values), 1)}
    return {"url": base, "tries": count, "measured_at_unix": time.time(),
            "get_new_connection": summary(new), "get_reused_connection": summary(reused),
            "websocket_handshake": summary(handshake)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18620")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(measure(args.url, args.n), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
