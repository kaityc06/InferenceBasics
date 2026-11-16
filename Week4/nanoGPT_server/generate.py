import torch
from torch.nn import functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, attention_mask=None, token_callback=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.

    For a batch of prompts with different lengths, LEFT-pad idx and pass attention_mask
    (shape (b,t), 1 = real token, 0 = pad). Left padding keeps every row's last real token in
    the final column, so all rows sample their next token from the same position. The returned
    idx still contains the pads; strip them with the mask.

    token_callback: optional fn(step, idx_next) called right after each token is sampled, with
    idx_next of shape (b, 1). Lets a caller stream tokens out before generation finishes.
    """
    block_size = model.config.block_size
    B, T = idx.size()
    assert max_new_tokens <= block_size, f"max_new_tokens ({max_new_tokens}) must be <= block_size ({block_size})"
    model.reset_cache() # clear K/V from any previous generate() call

    # crop the prompt so prompt + generated tokens fed back in fit in block_size.
    # the last sampled token is never fed back, so we need (T - start_index) + (max_new_tokens - 1) <= block_size
    start_index = max(0, T + max_new_tokens - 1 - block_size)

    # prefill: run the whole prompt through in one forward pass, filling the KV cache
    prompt = idx[:, start_index:]
    if attention_mask is not None:
        assert attention_mask.shape == idx.shape, "attention_mask must have the same shape as idx"
        attention_mask = attention_mask[:, start_index:].to(device=idx.device, dtype=torch.bool)
        assert attention_mask[:, -1].all(), "prompts must be LEFT-padded (last column must be real tokens)"
    logits, _ = model(prompt, use_cache = True, pos_value = 0, attention_mask = attention_mask)
    pos_value = prompt.size(1)

    # decode
    for i in range(max_new_tokens):
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        if token_callback is not None:
            token_callback(i, idx_next)
        if i < max_new_tokens - 1:
            # forward only the new token; K/V for earlier tokens come from the cache
            if attention_mask is not None:
                # the new token is real for every row
                attention_mask = torch.cat((attention_mask, attention_mask.new_ones(B, 1)), dim=1)
            logits, _ = model(idx_next, use_cache = True, pos_value = pos_value, attention_mask = attention_mask)
            pos_value += 1

    return idx
