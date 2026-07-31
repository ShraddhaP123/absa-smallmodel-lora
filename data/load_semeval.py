"""Load SemEval-2014 Task 4 ABSA (restaurants/laptops) and convert to our gold JSONL format.

Source: tomaarsen/setfit-absa-semeval-{restaurants,laptops} on the HF Hub.
That dataset is one row per (text, span, label, ordinal) — a candidate aspect
span with its polarity label. Rows whose label isn't a real polarity (e.g.
SetFit's "no aspect" negative-sampling rows) are dropped; the sentence is
still kept, with an empty aspect list, if no valid aspects remain.

GOTCHA: this HF re-hosting ships its "test" split with every label blanked
out (`label == ""`) — it's set up for the SetFit inference-demo tutorial,
not as a scored test set. Only the "train" split has real polarities. So we
ignore the HF split boundary entirely and carve our own train/val/test split
out of the labeled data, with a fixed seed for reproducibility. If a future
version of the dataset ships real test labels, this script will notice (the
"test" split will show up with a nonzero labeled fraction) and warn instead
of silently using it — verify manually before trusting it.

Usage:
    python -m data.load_semeval --domain restaurants --out data/processed
    python -m data.load_semeval --domain laptops --out data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset

from data.schema import Aspect, GoldExample

VALID_POLARITIES = {"positive", "negative", "neutral", "conflict"}
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

HF_DATASET_BY_DOMAIN = {
    "restaurants": "tomaarsen/setfit-absa-semeval-restaurants",
    "laptops": "tomaarsen/setfit-absa-semeval-laptops",
}


def convert_split(rows, domain: str) -> list[GoldExample]:
    by_text: "OrderedDict[str, list[Aspect]]" = OrderedDict()
    dropped = 0
    for row in rows:
        text = row["text"]
        by_text.setdefault(text, [])
        label = row["label"]
        if label not in VALID_POLARITIES:
            dropped += 1
            continue
        by_text[text].append(Aspect(term=row["span"], polarity=label))

    if dropped:
        print(f"[{domain}] dropped {dropped} non-polarity rows (e.g. SetFit 'no aspect' candidates)")

    examples = []
    for i, (text, aspects) in enumerate(by_text.items()):
        examples.append(GoldExample(id=f"{domain}-{i}", text=text, domain=domain, aspects=aspects))
    return examples


def _labeled_fraction(split) -> float:
    if len(split) == 0:
        return 0.0
    return sum(1 for label in split["label"] if label in VALID_POLARITIES) / len(split)


def three_way_split(examples: list[GoldExample], seed: int, ratios: tuple[float, float, float]):
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = [examples[i] for i in order[:n_train]]
    val = [examples[i] for i in order[n_train:n_train + n_val]]
    test = [examples[i] for i in order[n_train + n_val:]]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(HF_DATASET_BY_DOMAIN), required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    ds = load_dataset(HF_DATASET_BY_DOMAIN[args.domain])
    args.out.mkdir(parents=True, exist_ok=True)

    if _labeled_fraction(ds["test"]) > 0.5:
        print(
            f"[{args.domain}] WARNING: HF 'test' split now looks labeled "
            f"({_labeled_fraction(ds['test']):.0%} have a real polarity). "
            "This loader currently ignores it and builds its own split from "
            "'train' only — verify manually and update this script if you "
            "want to use the real test labels instead."
        )
    else:
        print(
            f"[{args.domain}] HF 'test' split has no real labels "
            f"({_labeled_fraction(ds['test']):.0%} labeled) — ignoring it, "
            "building train/val/test from the labeled 'train' split instead "
            f"(seed={SPLIT_SEED}, ratios={SPLIT_RATIOS})."
        )

    examples = convert_split(ds["train"], args.domain)
    train, val, test = three_way_split(examples, SPLIT_SEED, SPLIT_RATIOS)

    for split_name, split_examples in [("train", train), ("val", val), ("test", test)]:
        out_path = args.out / f"{args.domain}_{split_name}.jsonl"
        with out_path.open("w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex.to_json()) + "\n")
        n_aspects = sum(len(ex.aspects) for ex in split_examples)
        print(f"wrote {len(split_examples)} examples ({n_aspects} aspects) -> {out_path}")


if __name__ == "__main__":
    main()
