import asyncio
from aiohttp import web
import time

async def register_job(request):
    job_json = await request.json()
    job_id = job_json["id"]
    await request.app["queue"].put(job_id)

async def process_jobs():
    while True:
        job_id = await app["queue"].get()
        print(f"Job ID {job_id} submitted")
        time.sleep(2)

app = web.Application()
app["queue"] = asyncio.Queue()
app.add_routes([web.post("/jobs", register_job)])

async def background_ctx(app):
    task = asyncio.create_task(process_jobs())
    yield
    task.cancel()
app.cleanup_ctx.append(background_ctx)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=8080)


