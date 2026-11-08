import asyncio
from aiohttp import web
import time

async def register_job(request):
    job_json = await request.json()
    job_id = job_json["id"]
    future = asyncio.get_running_loop().create_future()
    await request.app["queue"].put(job_id, future)
    result = await future
    return web.json_response(result)

async def worker_thread():
    while True:
        job_id, future = await app["queue"].get()
        print(f"Job ID {job_id} submitted")
        result = await asyncio.to_thread(process, job_id)
        future.set_result(result)

def process(job_id):    
    time.sleep(0.2)
    return {"id": job_id, "status": "done"}

app = web.Application()
app["queue"] = asyncio.Queue()
app.add_routes([web.post("/jobs", register_job)])

async def background_ctx(app):
    task = asyncio.create_task(worker_thread())
    yield
    task.cancel()
app.cleanup_ctx.append(background_ctx)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=8080)


