import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from aiohttp import web
from transformers import GPT2TokenizerFast

import config
from generate import generate
from model import GPT

_SENTINEL = object() # pushed onto the request queue to stop the worker


@dataclass
class GenRequest:
    prompt_ids: list[int]
    loop: asyncio.AbstractEventLoop # the original event loop that submitted the request
    out: asyncio.Queue = field(default_factory=asyncio.Queue) # holds the generated tokens for a given client request, only the original event loop should be able to touch it

    # called by worker thread, instructs the original event loop to add item to out (since out is task-safe but not thread-safe)
    def emit(self, item):
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

# one UTF-8 character can span multiple tokens (i.e. an emoji), so this holds back tokens until they decode to complete text
class IncrementalDecoder:
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


class BatchWorker(threading.Thread):
    def __init__(self, model, tokenizer, device):
        super().__init__(daemon=True, name="batch-worker")
        self.model, self.tok, self.device = model, tokenizer, device
        self.requests = queue.Queue()

    def submit(self, req):
        self.requests.put(req)

    def stop(self):
        self.requests.put(_SENTINEL)

    # blocks until BATCH_SIZE requests are waiting or the oldest request has waited MAX_WAIT seconds, returns None on shutdown
    def _collect_batch(self):
        first = self.requests.get() # blocking call
        if first is _SENTINEL:
            return None
        batch = [first]
        deadline = None if config.MAX_WAIT is None else time.monotonic() + config.MAX_WAIT
        while len(batch) < config.BATCH_SIZE:
            try:
                if deadline is None:
                    req = self.requests.get()
                else:
                    req = self.requests.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty: # only triggered by the above timeout
                break
            if req is _SENTINEL:
                self.requests.put(_SENTINEL) # finish this batch, then stop on the next collect
                break
            batch.append(req)
        return batch

    def _left_pad(self, batch):
        T = max(len(r.prompt_ids) for r in batch)
        pad_id = self.tok.eos_token_id
        idx = torch.full((len(batch), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for row, r in enumerate(batch):
            n = len(r.prompt_ids)
            idx[row, T - n:] = torch.tensor(r.prompt_ids, dtype=torch.long)
            mask[row, T - n:] = 1
        return idx.to(self.device), mask.to(self.device)

    def _run_batch(self, batch):
        idx, mask = self._left_pad(batch)
        steps = config.MAX_NEW_TOKENS
        decoders = [IncrementalDecoder(self.tok) for _ in batch] # each request gets a separate IncrementalDecoder

        # model calls this for every generated batch of tokens
        def on_token(step, idx_next):
            ids = idx_next[:, 0].tolist() # idx_next is tensor stored on GPU, so make copy on CPU
            for r, dec, tid in zip(batch, decoders, ids):
                r.emit({"token_id": tid, "text": dec.push(tid)})
                if step == steps - 1:
                    r.emit({"done": True, "num_tokens": steps})

        generate(self.model, idx, steps, temperature=config.TEMPERATURE, top_k=config.TOP_K,
                 attention_mask=mask, token_callback=on_token)
        print(f"[worker] batch={len(batch)} padded_prompt_len={idx.size(1)} new_tokens={steps}", flush=True)

    def run(self):
        while True:
            batch = self._collect_batch()
            if batch is None:
                return
            try:
                self._run_batch(batch)
            except Exception as e:
                print(f"[worker] batch failed: {e!r}", flush=True)
                for r in batch:
                    r.emit({"error": f"generation failed: {e}"})


async def handle_generate(request: web.Request):
    app = request.app
    try:
        body = await request.json()
        prompt = body["prompt"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return web.json_response({"error": 'expected JSON {"prompt": str}'}, status=400)

    prompt_ids = app["tokenizer"].encode(prompt) if isinstance(prompt, str) else []
    if not prompt_ids:
        return web.json_response({"error": "prompt must be a non-empty string"}, status=400)

    req = GenRequest(prompt_ids, asyncio.get_running_loop())
    app["worker"].submit(req)

    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request) # starts the streaming response, sends headers and allows resp.write()
    try:
        while True:
            item = await req.out.get()
            await resp.write((json.dumps(item) + "\n").encode())
            if "done" in item or "error" in item:
                break
    except ConnectionResetError:
        pass # if client disconnects, the worker finishes the batch and its tokens for this request are dropped
    await resp.write_eof()
    return resp


def main():
    device = config.DEVICE or ("cuda" if torch.cuda.is_available()
                               else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(config.MODEL)
    model = GPT.from_pretrained(config.MODEL).to(device).eval()
    print(f"[server] model={config.MODEL} device={device} batch_size={config.BATCH_SIZE} max_wait={config.MAX_WAIT}")

    worker = BatchWorker(model, tokenizer, device)

    app = web.Application()
    app.update(model=model, tokenizer=tokenizer, worker=worker)
    app.router.add_post("/generate", handle_generate)

    async def start_worker(app):
        worker.start()

    async def stop_worker(app):
        worker.stop()
        await asyncio.to_thread(worker.join, 5) # wait up to 5 seconds for server to finish its current batch

    app.on_startup.append(start_worker)
    app.on_cleanup.append(stop_worker)
    web.run_app(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
