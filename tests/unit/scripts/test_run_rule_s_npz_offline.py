from dataclasses import dataclass

import torch

from scripts.run_rule_s_npz_offline import inject_prompt_prefix


@dataclass(frozen=True)
class Step:
    action: str
    data: torch.Tensor | None


def test_inject_prompt_prefix_replaces_only_initial_accompaniment_actions():
    schedule = [
        Step("inject", torch.tensor([1])),
        Step("generate", None),
        Step("inject", torch.tensor([2])),
        Step("inject_gt", torch.tensor([3])),
        Step("generate", None),
    ]
    prefix = [torch.tensor([10]), torch.tensor([11])]

    result = inject_prompt_prefix(schedule, prefix)

    assert [step.action for step in result] == [
        "inject",
        "inject_gt",
        "inject",
        "inject_gt",
        "generate",
    ]
    assert result[1].data.tolist() == [10]
    assert result[3].data.tolist() == [11]
    assert result[4].data is None
