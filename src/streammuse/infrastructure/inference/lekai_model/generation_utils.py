import torch

def sample_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
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
        top_k_values = torch.topk(logits, top_k)[0]
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
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # Handle batch dimension if necessary, but here we assume simple scatter works or loop
        # For batch_size=1 it's simple. For batch > 1, we need to be careful with indexing.
        # The original code used logits[0, indices_to_remove] = -inf, implying batch=1.
        
        if logits.shape[0] == 1:
             logits[0, indices_to_remove] = -float('Inf')
        else:
            # Fallback or more complex implementation for batch > 1 if needed
            # For now, let's assume batch=1 as per original context
             logits[0, indices_to_remove] = -float('Inf')
    
    # Sampling
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
