from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch.nn import functional as F
from transformers import GPT2TokenizerFast

from model import GPT


class StreamDecoder(Protocol):
    def push(self, token_id: int) -> str:
        """Feed one generated id; return any text that is now complete (possibly "")."""
        ...


@dataclass
class RequestState:
    prompt_ids: list[int]
    max_new_tokens: int
    decoder: StreamDecoder
    output_ids: list[int] = field(default_factory=list) # generated tokens sent to the client (end-of-sequence excluded)
    finished: bool = False # True once the request hit end-of-sequence or max_new_tokens


# one generated token for one request
@dataclass
class TokenOut:
    token_id: int
    text: str # decoded text that is now complete (possibly "" while a multi-token character is pending)


class ModelRunner(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def new_state(self, prompt_ids: list[int], max_new_tokens: int) -> RequestState:
        """A fresh state for one request."""
        ...

    def prefill(self, batch: list[RequestState]) -> list[TokenOut | None]:
        """Run each request's prompt through the model and sample its first token.
        Returns one entry per state, in order: the new token, or None if the request sampled
        end-of-sequence (nothing to send). Afterwards state.finished says whether it is done."""
        ...

    def step(self, batch: list[RequestState]) -> list[TokenOut | None]:
        """Feed each request's last token and sample its next one. Every state must have been
        prefilled and not be finished, but the batch doesn't have to match the one passed to
        prefill: the cache lives in each state. Returns the same way as prefill."""
        ...


# a UTF-8 character is at most 4 bytes = 4 tokens for GPT2
MAX_PENDING = 4

# one UTF-8 character can span multiple tokens (i.e. an emoji), so this holds back tokens until they decode to complete text
class GPT2StreamDecoder:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pending = [] # tokens whose text hasn't been emitted yet

    def push(self, token_id):
        self.pending.append(token_id)
        text = self.tok.decode(self.pending)
        if text.endswith("�") and len(self.pending) < MAX_PENDING: # incomplete character, so wait for the next token
            return ""
        self.pending = []
        return text


# logits (B, vocab) -> sampled ids (B,)
def sample(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)[:, 0]


@dataclass
class GPT2State(RequestState):
    # this request's own KV cache: one (k, v) per layer, each (1, T, C) with only real tokens (no padding)
    kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    last_token: int | None = None # the most recent sampled token, fed to the model on the next step


class GPT2Runner:
    def __init__(self, model_name, device, temperature=1.0, top_k=None):
        self.name = model_name
        self.device = device
        self.temperature, self.top_k = temperature, top_k
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT.from_pretrained(model_name).to(device).eval()

    def encode(self, text):
        return self.tok.encode(text)

    def new_state(self, prompt_ids, max_new_tokens):
        block_size = self.model.config.block_size
        assert 1 <= max_new_tokens <= block_size, f"max_new_tokens ({max_new_tokens}) must be in [1, block_size ({block_size})]"
        # crop the prompt so prompt + generated tokens fed back in fit in block_size, the last sampled token is never fed back, so we need len(prompt) + (max_new_tokens - 1) <= block_size
        prompt_ids = prompt_ids[-(block_size - max_new_tokens + 1):]
        return GPT2State(prompt_ids, max_new_tokens, GPT2StreamDecoder(self.tok))

    # left-pads so every row's last real token is in the final column; mask marks real tokens
    def _left_pad(self, prompts):
        T = max(len(p) for p in prompts)
        pad_id = self.tok.eos_token_id
        idx = torch.full((len(prompts), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), T), dtype=torch.long)
        for row, p in enumerate(prompts):
            n = len(p)
            idx[row, T - n:] = torch.tensor(p, dtype=torch.long)
            mask[row, T - n:] = 1
        return idx.to(self.device), mask.to(self.device)

    # stacks every state's own cache into one left-padded batch cache.
    # returns (past_kv, mask) where mask covers the cache plus one new token, or None if no padding was needed
    def _stack_kv(self, batch):
        lengths = [s.kv[0][0].size(1) for s in batch]
        T = max(lengths)
        if all(n == T for n in lengths): # same length, so no padding and no mask needed
            past_kv = [(torch.cat([s.kv[layer][0] for s in batch]), torch.cat([s.kv[layer][1] for s in batch]))
                       for layer in range(len(batch[0].kv))]
            return past_kv, None
        past_kv = []
        for layer in range(len(batch[0].kv)):
            k0 = batch[0].kv[layer][0]
            k = k0.new_zeros(len(batch), T, k0.size(-1))
            v = k0.new_zeros(len(batch), T, k0.size(-1))
            for row, (s, n) in enumerate(zip(batch, lengths)):
                k[row, T - n:] = s.kv[layer][0][0]
                v[row, T - n:] = s.kv[layer][1][0]
            past_kv.append((k, v))
        mask = torch.zeros(len(batch), T + 1, dtype=torch.long, device=self.device)
        for row, n in enumerate(lengths):
            mask[row, T - n:] = 1 # real cached tokens and the new token
        return past_kv, mask

    # splits a batch cache (left-padded to width T) back into each state's own cache
    @staticmethod
    def _unstack_kv(batch, present_kv, lengths):
        T = present_kv[0][0].size(1)
        for row, (s, n) in enumerate(zip(batch, lengths)):
            s.kv = [(k[row:row + 1, T - n:], v[row:row + 1, T - n:]) for k, v in present_kv]

    # records each state's newly sampled token: updates the state and returns what to send to the client
    def _record(self, batch, next_ids):
        outs = []
        for s, tid in zip(batch, next_ids.tolist()): # next_ids is stored on the device, so copy to CPU
            if tid == self.tok.eos_token_id:
                s.finished = True
                outs.append(None)
            else:
                s.output_ids.append(tid)
                s.last_token = tid
                s.finished = len(s.output_ids) >= s.max_new_tokens
                outs.append(TokenOut(tid, s.decoder.push(tid)))
            if s.finished:
                s.kv = None # free this request's cache
        return outs

    @torch.no_grad()
    def prefill(self, batch):
        idx, mask = self._left_pad([s.prompt_ids for s in batch])
        logits, _, present_kv = self.model(idx, attention_mask=mask)
        self._unstack_kv(batch, present_kv, [len(s.prompt_ids) for s in batch])
        return self._record(batch, sample(logits[:, -1, :], self.temperature, self.top_k))

    @torch.no_grad()
    def step(self, batch):
        assert all(s.kv is not None and not s.finished for s in batch), "step() needs prefilled, unfinished states"
        past_kv, mask = self._stack_kv(batch)
        idx = torch.tensor([[s.last_token] for s in batch], dtype=torch.long, device=self.device) # (B, 1)
        logits, _, present_kv = self.model(idx, past_kv=past_kv, attention_mask=mask)
        self._unstack_kv(batch, present_kv, [s.kv[0][0].size(1) + 1 for s in batch])
        return self._record(batch, sample(logits[:, -1, :], self.temperature, self.top_k))


def make_runner(model_type, model_name, device, **gen_kwargs) -> ModelRunner:
    if model_type == "gpt2":
        return GPT2Runner(model_name, device, **gen_kwargs)
    raise ValueError(f"unknown MODEL_TYPE {model_type!r}")
