import asyncio
from client import send_request
import aiohttp

async def main():
    async with aiohttp.ClientSession(base_url="http://127.0.0.1:8080"):
        job_ids = [i for i in range(0,100)]
        asyncio.gather(*(send_request(job_id) for job_id in job_ids))
        