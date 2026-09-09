"""Start the unchanged server and record native runtime settings once."""
import json
import os
from pathlib import Path

from streammuse.infrastructure.inference import server_lekai


if __name__ == '__main__':
    import torch
    import transformers
    target = Path(os.environ['STREAMMUSE_EVAL_RUNTIME_SNAPSHOT'])
    snapshot = {'standard': server_lekai.backend.runtime_info(),
                'prompt_continuation': server_lekai.prompt_continuation_backend.runtime_info(),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'transformers_file': transformers.__file__,
                'gpu': torch.cuda.get_device_name(0),
                'scope': 'One-time startup snapshot, no generation or sampling observer'}
    target.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    server_lekai.main()
