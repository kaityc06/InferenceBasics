import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from aiohttp import web

import config
from runners import make_runner

_SENTINEL = object() # pushed onto the request queue to stop the worker


@dataclass
class GenRequest:
    prompt_ids: list[int]
    loop: asyncio.AbstractEventLoop # the original event loop that submitted the request
    out: asyncio.Queue = field(default_factory=asyncio.Queue) # holds the generated tokens for a given client request, only the original event loop should be able to touch it

    # called by worker thread, instructs the original event loop to add item to out (since out is task-safe but not thread-safe)
    def emit(self, item):
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

class BatchWorker(threading.Thread):
    def __init__(self, runner):
        super().__init__(daemon=True, name="batch-worker")
        self.runner = runner
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

    def _run_batch(self, batch):
        states = [self.runner.new_state(r.prompt_ids, config.MAX_NEW_TOKENS) for r in batch] # one state per request
        active = list(zip(batch, states))
        n_steps = 0

        outs = self.runner.prefill(states) # runs the prompts and samples each request's first token
        while True:
            for (r, s), out in zip(active, outs):
                if out is not None: # None means the request sampled the EOS token, which isn't sent
                    r.emit({"token_id": out.token_id, "text": out.text})
                if s.finished: # reached EOS or MAX_NEW_TOKENS
                    r.emit({"done": True, "num_tokens": len(s.output_ids)})
            active = [(r, s) for r, s in active if not s.finished] # finished requests leave the batch
            if not active:
                break
            outs = self.runner.step([s for _, s in active])
            n_steps += 1

        print(f"[worker] batch={len(batch)} max_prompt_len={max(len(s.prompt_ids) for s in states)} "
              f"steps={n_steps} num_tokens={[len(s.output_ids) for s in states]}", flush=True)

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

    prompt_ids = app["runner"].encode(prompt) if isinstance(prompt, str) else []
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
    runner = make_runner(config.MODEL_TYPE, config.MODEL, device,
                         temperature=config.TEMPERATURE, top_k=config.TOP_K)
    print(f"[server] model_type={config.MODEL_TYPE} model={config.MODEL} device={device} batch_size={config.BATCH_SIZE} max_wait={config.MAX_WAIT}")

    worker = BatchWorker(runner)

    app = web.Application()
    app.update(runner=runner, worker=worker)
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
