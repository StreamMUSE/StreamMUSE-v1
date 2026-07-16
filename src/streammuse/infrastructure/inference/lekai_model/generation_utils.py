import torch

def sample_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
):
    """
    Apply sampling strategies to logits and return the next token.
    
    Args:
        logits: (batch_size, vocab_size)
        generated_tokens: (batch_size, seq_len) - required for repetition penalty
        temperature: float
        top_k: int
        top_p: float
        repetition_penalty: float
        generator: Explicit RNG owned by the run/session.  Passing it avoids consuming
            PyTorch's process-global RNG and makes sampling reproducible under a fixed
            call order.  Its device must match ``logits.device``.
    """
    # Handle temperature=0 (greedy decoding)
    if temperature == 0.0:
        # Greedy: return the token with highest logit
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token
    
    # Temperature scaling
    logits = logits / temperature
    
    # Repetition penalty
    # Note: This implementation follows the original code in model.py.
    # Ideally, it should handle negative logits differently.
    if repetition_penalty != 1.0 and generated_tokens is not None:
        batch_size = logits.shape[0]
        for i in range(batch_size):
            # Convert to list for set operation
            # generated_tokens[i] might be on GPU, move to cpu/list
            current_tokens = generated_tokens[i].tolist()
            for token_id in set(current_tokens):
                # Original implementation: simple division
                logits[i, token_id] /= repetition_penalty
    
    # Top-k sampling
    if top_k > 0:
        # Filter top_k
        effective_top_k = min(top_k, logits.shape[-1])
        top_k_values = torch.topk(logits, effective_top_k)[0]
        # Use the smallest of the top_k as the threshold
        min_top_k = top_k_values[..., -1, None]
        indices_to_remove = logits < min_top_k
        logits[indices_to_remove] = -float('Inf')
    
    # Top-p sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter the sorted removal mask back to the vocabulary order.  This works for
        # every batch row; the historical implementation accidentally applied all rows'
        # indices to row zero.
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits.masked_fill_(indices_to_remove, -float('Inf'))
    
    # Sampling
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token
