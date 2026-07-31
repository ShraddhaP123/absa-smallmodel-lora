"""LoRA fine-tune of a small instruction model on the ABSA extraction task.

Trains on data/processed/<domain>_{train,val}.jsonl, formatting each example
as (prompt + completion) using the SAME plain-text prompt format
eval/runner.py's `hf` backend uses at inference time (no chat template) --
train/inference format consistency matters more than it looks like it
should; see CLAUDE.md.

Usage:
    python train.py --config variants/lora-r8.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data.schema import GoldExample
from eval.runner import build_prompt


def load_examples(domain: str, split: str) -> list[GoldExample]:
    path = Path("data/processed") / f"{domain}_{split}.jsonl"
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(GoldExample.from_json(json.loads(line)))
    return examples


class ABSADataset(Dataset):
    """(prompt + completion) pairs, tokenized, with the prompt masked out of the loss.

    Every example is padded/truncated to a fixed max_seq_length up front so
    the default data collator can batch them without a custom collate_fn.
    """

    def __init__(
        self,
        examples: list[GoldExample],
        tokenizer,
        prompt_template: str,
        domain: str,
        max_seq_length: int,
    ):
        self.rows = []
        n_truncated = 0
        for ex in examples:
            prompt = build_prompt(prompt_template, domain, ex.text)
            completion = json.dumps(
                [{"term": a.term, "polarity": a.polarity} for a in ex.aspects]
            )
            full_text = prompt + completion + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if len(full_ids) > max_seq_length:
                n_truncated += 1
                full_ids = full_ids[:max_seq_length]

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100  # don't train on predicting the prompt itself

            pad_len = max_seq_length - len(full_ids)
            attention_mask = [1] * len(full_ids) + [0] * pad_len
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # don't train on padding either

            self.rows.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        if n_truncated:
            print(
                f"warning: {n_truncated}/{len(examples)} examples truncated to "
                f"max_seq_length={max_seq_length} (their tail, including possibly "
                "the completion, was cut off)"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ABSADataset(
        load_examples(config["domain"], config["train_split"]),
        tokenizer,
        config["prompt_template"],
        config["domain"],
        config["max_seq_length"],
    )
    val_dataset = ABSADataset(
        load_examples(config["domain"], config["val_split"]),
        tokenizer,
        config["prompt_template"],
        config["domain"],
        config["max_seq_length"],
    )
    print(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(config["base_model"], dtype=torch.float16)

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=train_cfg["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"adapter saved -> {config['output_dir']}")


if __name__ == "__main__":
    main()
