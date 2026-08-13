import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedModel
from typing import Optional


class PianoLLaMA(PreTrainedModel):
    """基于LLaMA架构的Piano生成模型（纯模型层，不含 I/O）"""
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LlamaForCausalLM(self.config)
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False if labels is not None else True,
        )

    # ==================== 自回归生成 ====================

    @torch.no_grad()
    def generate_music(
        self,
        initial_tokens: torch.Tensor,
        device: str = 'cuda',
        max_length: int = 8192,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        input_ids = initial_tokens.unsqueeze(0).to(device)

        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        return output.cpu()

    @torch.no_grad()
    def generate_music_batch(
        self,
        initial_tokens: torch.Tensor,
        batch_size: int,
        device: str = 'cuda',
        max_length: int = 8192,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate independent samples from one repeated Melody condition."""

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.eval()
        input_ids = (
            initial_tokens.unsqueeze(0)
            .repeat(int(batch_size), 1)
            .contiguous()
            .to(device)
        )
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        return output.cpu()
