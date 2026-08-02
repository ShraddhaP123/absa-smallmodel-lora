"""Minimal concurrent load test against a served variant's OpenAI-compatible endpoint.

    python serve/loadtest.py --endpoint http://localhost:8000/v1 --model absa-lora-merged \
        --domain restaurants --split test --concurrency 8

Prints latency p50/p95 and throughput; feed the numbers into results.csv via
the /log-run skill (this script does not write to results.csv itself, since
serving hardware/cost details need a human to record accurately).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.runner import build_prompt  # noqa: E402  (needs sys.path set first)

PROMPT_TEMPLATE = "variants/prompts/absa_extract.txt"


def load_texts(domain: str, split: str, limit: int) -> list[str]:
    path = Path("data/processed") / f"{domain}_{split}.jsonl"
    texts = []
    with path.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= limit:
                break
    return texts


def call(endpoint: str, model: str, domain: str, text: str) -> float:
    # Plain /completions, not /chat/completions -- matches eval/runner.py's
    # run_vllm_backend and the raw-text prompt format training actually used.
    prompt = build_prompt(PROMPT_TEMPLATE, domain, text)
    start = time.perf_counter()
    resp = requests.post(
        f"{endpoint}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", default="restaurants")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    texts = load_texts(args.domain, args.split, args.n)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        latencies = list(pool.map(lambda t: call(args.endpoint, args.model, args.domain, t), texts))
    wall = time.perf_counter() - start

    print(f"n={len(latencies)} concurrency={args.concurrency}")
    print(f"latency_p50_ms={percentile(latencies, 0.50):.1f}")
    print(f"latency_p95_ms={percentile(latencies, 0.95):.1f}")
    print(f"throughput_req_per_s={len(latencies) / wall:.2f}")


if __name__ == "__main__":
    main()
