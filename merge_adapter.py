"""Merge a LoRA adapter into its base model and save full merged weights.

vLLM (and most serving stacks) want a single set of weights, not a base
model plus a separate adapter to apply at load time -- merging bakes the
adapter's effect directly into the model's weight matrices, producing an
ordinary model directory that any serving stack can load like any other.

Usage:
    python merge_adapter.py --base HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --adapter checkpoints/lora-r8-mlp-synth --out checkpoints/merged-best
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)  # adapter dir also has the tokenizer we saved
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, args.adapter)

    merged = model.merge_and_unload()  # bakes the LoRA delta into the base weights, drops the adapter wrapper
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"merged model saved -> {args.out}")


if __name__ == "__main__":
    main()
