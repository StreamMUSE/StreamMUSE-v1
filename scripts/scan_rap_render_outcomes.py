"""Map every render-server request id to its server-side outcome (run on the H200).

Prints JSON ``{request_id: outcome}`` where the outcome is ``completed`` (the
server wrote a response package), the server's failure code from
``failure.json`` (e.g. ``no_valid_candidates``, ``budget_exhausted``), or
``incomplete``. Works for both artifact layouts: ``<root>/<request_id>/``
(v0.0) and ``<root>/<producer>/<request_id>/`` (v0.1 and later).

    python3 scripts/scan_rap_render_outcomes.py <render_artifacts> [more...]
"""

import json
from pathlib import Path
import sys

outcomes = {}
for root in map(Path, sys.argv[1:]):
    for request_json in root.rglob("request.json"):
        folder = request_json.parent
        if (folder / "failure.json").exists():
            outcome = json.loads((folder / "failure.json").read_text()).get("code", "failed")
        elif any(folder.glob("response*.zip")) or (folder / "complete.v1.json").exists():
            outcome = "completed"
        else:
            outcome = "incomplete"
        outcomes[folder.name] = outcome
print(json.dumps(outcomes))
