"""
Async client for server.py: fires several prompts concurrently and prints each stream as it arrives.

  python3 client.py                                   # 4 default prompts
  python3 client.py --num-requests 8
  python3 client.py --prompt "The meaning of life is" --prompt "Once upon a time"
"""

import argparse
import asyncio
import json

import aiohttp

DEFAULT_PROMPTS = [
    "The meaning of life is",
    "Once upon a time, in a small village by the sea,",
    "In machine learning, a transformer is",
    "My favorite food is",
    "The quick brown fox",
    "Breaking news: scientists have discovered",
    "def fibonacci(n):",
    "The best way to learn a new language is",
]


async def send_prompt(session, url, i, prompt, verbose):
    pieces, n = [], 0
    async with session.post(url, json={"prompt": prompt}) as resp:
        if resp.status != 200:
            return dict(i=i, prompt=prompt, error=f"HTTP {resp.status}: {await resp.text()}")
        async for line in resp.content: # runs until resp.write_eof() is called
            item = json.loads(line)
            if "error" in item:
                return dict(i=i, prompt=prompt, error=item["error"])
            if item.get("done"): # sent when MAX_NEW_TOKENS have been genereated or when model generates EOS token
                break
            n += 1
            pieces.append(item["text"])
            if verbose:
                print(f"[req {i}] {item['text']!r}", flush=True)
    return dict(i=i, prompt=prompt, text="".join(pieces), n=n)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/generate")
    p.add_argument("--prompt", action="append", help="repeatable; overrides the default prompts")
    p.add_argument("--num-requests", type=int, default=4)
    p.add_argument("--quiet", action="store_true", help="don't print tokens as they stream in")
    args = p.parse_args()

    prompts = args.prompt or [DEFAULT_PROMPTS[k % len(DEFAULT_PROMPTS)] for k in range(args.num_requests)]
    timeout = aiohttp.ClientTimeout(total=None) # the server may hold a request until its batch fills
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*(
            send_prompt(session, args.url, i, pr, not args.quiet) for i, pr in enumerate(prompts)
        ))

    print("\n" + "=" * 80)
    for r in results:
        if "error" in r:
            print(f"[req {r['i']}] ERROR: {r['error']}")
            continue
        print(f"[req {r['i']}] tokens={r['n']}")
        print(f"  {r['prompt']}\033[1m{r['text']}\033[0m\n")


if __name__ == "__main__":
    asyncio.run(main())
