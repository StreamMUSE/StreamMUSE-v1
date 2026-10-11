"""Run the normal HTTP backend with an explicit experiment-only repair hook."""
import json
import os
from pathlib import Path
import time


def main():
    from streammuse.infrastructure.inference import server_lekai as server
    from streammuse.infrastructure.inference.history_repair_controller import FixedBarHistoryRepair
    from experiments.bar_quality_history.process_worker import ProcessRepairWorker
    import torch
    import transformers

    ce = server.prompt_continuation_backend._engine._continuation_engine
    worker = None
    if os.environ["BAR_HISTORY_MODE"] == "repair":
        config = json.loads(Path(os.environ["BAR_HISTORY_CONFIG"]).read_text())
        worker = ProcessRepairWorker(config=config, gpu=os.environ["CUDA_VISIBLE_DEVICES"], seed=101,
                                     audit_dir=os.environ["BAR_HISTORY_WORKER_AUDIT"], allow_gpu=True,
                                     boundary_policy=os.environ.get('BAR_HISTORY_REPAIR_POLICY', 'strict'))
        deadline = time.monotonic() + 180
        while not worker.ready():
            if time.monotonic() >= deadline:
                worker.close()
                raise TimeoutError("repair worker preload timeout")
            time.sleep(0.1)
        ce.configure_history_repair_factory(lambda: FixedBarHistoryRepair(
            start_tick=64, end_tick=80, worker=worker))

    @server.app.get("/debug/history_repair")
    async def repair_audit():
        controller = ce.history_repair_audit()
        saved = [a for r in controller for a in r.get("application_audit", [])]
        return {"controller": controller,
                "applied": saved or ce._backend.history_repair_audit_snapshot(),
                "guard": ce._backend._empty_token_guard.snapshot()}

    snapshot = {"standard": server.backend.runtime_info(),
                "prompt_continuation": server.prompt_continuation_backend.runtime_info(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "transformers_file": transformers.__file__, "gpu": torch.cuda.get_device_name(0)}
    Path(os.environ["STREAMMUSE_EVAL_RUNTIME_SNAPSHOT"]).write_text(json.dumps(snapshot, indent=2))
    try:
        server.main()
    finally:
        ce.configure_history_repair_factory(None)
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
