"""Run a variant end-to-end against an eval split and log the result.

    python -m eval.runner --variant variants/base.yaml

Pipeline: load gold examples -> generate a prediction per example via the
variant's backend -> score with eval.absa_eval.compute_metrics -> append one
row to results/results.csv, tagged with the git SHA and a hash of the config.
Predictions are also saved to results/predictions/ so a run can be re-scored
without re-generating if absa_eval.py is deliberately re-versioned.

Backend imports (transformers/peft/torch, openai, anthropic) are lazy so this
file imports cleanly on a machine that only has the ones it needs installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

from data.schema import GoldExample
from eval.absa_eval import compute_metrics

RESULTS_CSV = Path("results/results.csv")
PREDICTIONS_DIR = Path("results/predictions")

CSV_FIELDS = [
    "timestamp", "git_sha", "config_hash", "variant_name", "backend", "model",
    "quantization", "hardware", "domain", "split", "n_examples",
    "aspect_f1", "sentiment_acc", "joint_f1", "parse_rate",
    "latency_p50_ms", "latency_p95_ms", "cost_per_1k_examples_usd", "notes",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def load_gold(domain: str, split: str) -> list[GoldExample]:
    path = Path("data/processed") / f"{domain}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m data.load_semeval --domain {domain}"
        )
    golds = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                golds.append(GoldExample.from_json(json.loads(line)))
    return golds


def build_prompt(template_path: str, domain: str, text: str) -> str:
    template = Path(template_path).read_text()
    return template.format(domain=domain, text=text)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def run_api_backend(config: dict, golds: list[GoldExample]) -> tuple[dict[str, str], list[float]]:
    provider = config["api"]["provider"]
    model = config["api"]["model"]
    predictions: dict[str, str] = {}
    latencies: list[float] = []

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()

        def call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_new_tokens", 256),
            )
            return resp.choices[0].message.content or ""
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

        def call(prompt: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=config.get("max_new_tokens", 256),
                temperature=config.get("temperature", 0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
    else:
        raise ValueError(f"unknown api provider: {provider}")

    for gold in golds:
        prompt = build_prompt(config["prompt_template"], config["domain"], gold.text)
        start = time.perf_counter()
        raw = call(prompt)
        latencies.append((time.perf_counter() - start) * 1000)
        predictions[gold.id] = raw

    return predictions, latencies


def run_hf_backend(config: dict, golds: list[GoldExample]) -> tuple[dict[str, str], list[float]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_cfg = config["hf"]
    quant_kwargs = {}
    if hf_cfg.get("quantization") == "4bit":
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif hf_cfg.get("quantization") == "8bit":
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        hf_cfg["base_model"], device_map=hf_cfg.get("device", "cuda"), **quant_kwargs,
    )
    if hf_cfg.get("lora_adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, hf_cfg["lora_adapter"])
    model.eval()

    predictions: dict[str, str] = {}
    latencies: list[float] = []
    for gold in golds:
        prompt = build_prompt(config["prompt_template"], config["domain"], gold.text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", 256),
                temperature=config.get("temperature", 0.0) or None,
                do_sample=config.get("temperature", 0.0) > 0,
            )
        latencies.append((time.perf_counter() - start) * 1000)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions[gold.id] = text

    return predictions, latencies


def run_vllm_backend(config: dict, golds: list[GoldExample]) -> tuple[dict[str, str], list[float]]:
    import requests

    vllm_cfg = config["vllm"]
    predictions: dict[str, str] = {}
    latencies: list[float] = []
    for gold in golds:
        prompt = build_prompt(config["prompt_template"], config["domain"], gold.text)
        start = time.perf_counter()
        resp = requests.post(
            f"{vllm_cfg['endpoint']}/chat/completions",
            json={
                "model": vllm_cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.get("temperature", 0.0),
                "max_tokens": config.get("max_new_tokens", 256),
            },
            timeout=60,
        )
        resp.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)
        predictions[gold.id] = resp.json()["choices"][0]["message"]["content"]

    return predictions, latencies


BACKENDS = {"api": run_api_backend, "hf": run_hf_backend, "vllm": run_vllm_backend}


def append_result_row(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="path to a variants/*.yaml config")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config = load_config(args.variant)
    backend_name = config["backend"]
    if backend_name not in BACKENDS:
        raise ValueError(f"unknown backend: {backend_name}")

    golds = load_gold(config["domain"], config["split"])
    predictions, latencies = BACKENDS[backend_name](config, golds)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PREDICTIONS_DIR / f"{config['name']}_{config['domain']}_{config['split']}.jsonl"
    with pred_path.open("w") as f:
        for gold in golds:
            f.write(json.dumps({"id": gold.id, "raw_output": predictions.get(gold.id, "")}) + "\n")

    metrics = compute_metrics(golds, predictions)

    backend_cfg = config.get(backend_name, {})
    model_name = backend_cfg.get("model") or backend_cfg.get("base_model", "")
    quantization = backend_cfg.get("quantization") or "none"

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "config_hash": config_hash(args.variant),
        "variant_name": config["name"],
        "backend": backend_name,
        "model": model_name,
        "quantization": quantization,
        "hardware": config.get("hardware", "local"),
        "domain": config["domain"],
        "split": config["split"],
        "n_examples": metrics.n_examples,
        "aspect_f1": metrics.aspect_f1,
        "sentiment_acc": metrics.sentiment_acc,
        "joint_f1": metrics.joint_f1,
        "parse_rate": metrics.parse_rate,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 1),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
        "cost_per_1k_examples_usd": "",
        "notes": args.notes,
    }
    append_result_row(row)

    print(f"predictions -> {pred_path}")
    print(f"results row -> {RESULTS_CSV}")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
