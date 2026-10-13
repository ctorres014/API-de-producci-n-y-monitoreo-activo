"""
Concurrent load test — sends 5 tasks simultaneously and polls until done.

Usage:
    python load_test.py [--url http://localhost:8000]

Results (job_id, status, latency) are printed to stdout and saved to
load_test_results.json so you can cross-reference with the LangSmith /
Phoenix dashboard.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime

import httpx


BASE_URL = "http://localhost:8000"

QUERIES = [
    "What is machine learning and how does it work?",
    "Research the history of artificial intelligence",
    "Analyze the current state of renewable energy adoption",
    "What are the key trends in generative AI for 2024?",
    "Explain the impact of large language models on software development",
]

POLL_INTERVAL = 3   # seconds between status checks
MAX_POLLS = 60      # give up after 3 minutes per task


async def submit_task(client: httpx.AsyncClient, query: str) -> str:
    """Submit a task and return the job_id."""
    resp = await client.post("/tasks", json={"query": query}, timeout=15)
    resp.raise_for_status()
    return resp.json()["job_id"]


async def poll_until_done(client: httpx.AsyncClient, job_id: str) -> dict:
    """Poll GET /tasks/{job_id} until the job reaches a terminal state."""
    terminal = {"done", "failed", "rejected"}
    for _ in range(MAX_POLLS):
        await asyncio.sleep(POLL_INTERVAL)
        resp = await client.get(f"/tasks/{job_id}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") in terminal:
                return data
    return {"job_id": job_id, "status": "timeout"}


async def run_one(client: httpx.AsyncClient, index: int, query: str) -> dict:
    t0 = time.perf_counter()
    print(f"[{index+1}/5] Submitting: {query[:60]}...")
    job_id = await submit_task(client, query)
    print(f"[{index+1}/5] job_id={job_id}")
    result = await poll_until_done(client, job_id)
    elapsed = round(time.perf_counter() - t0, 2)
    result["elapsed_seconds"] = elapsed
    result["query"] = query
    print(f"[{index+1}/5] {job_id} → {result['status']} in {elapsed}s")
    return result


async def main(base_url: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Load test — 5 concurrent requests → {base_url}")
    print(f"  Started at {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(base_url=base_url) as client:
        # Verify the API is up
        health = await client.get("/health", timeout=5)
        print("Health:", health.json(), "\n")

        # Submit all 5 tasks concurrently
        t_total = time.perf_counter()
        tasks = [run_one(client, i, q) for i, q in enumerate(QUERIES)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = round(time.perf_counter() - t_total, 2)

    # Normalise exceptions
    clean = []
    for r in results:
        if isinstance(r, Exception):
            clean.append({"status": "error", "error": str(r)})
        else:
            clean.append(r)

    print(f"\n{'='*60}")
    print(f"  All tasks finished in {total_elapsed}s")
    print(f"{'='*60}")

    latencies = [r.get("elapsed_seconds", 0) for r in clean if isinstance(r, dict)]
    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
        p95 = latencies_sorted[p95_idx]
        print(f"  p95 latency : {p95}s")
        print(f"  avg latency : {round(sum(latencies)/len(latencies), 2)}s")
        print(f"  max latency : {max(latencies)}s")

    output_file = "load_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {"total_elapsed_seconds": total_elapsed, "results": clean},
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Results saved to {output_file}")
    print("  Cross-reference with your LangSmith/Phoenix dashboard for p95 and cost.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    args = parser.parse_args()
    asyncio.run(main(args.url))
