"""Post-hoc error analysis: what KINDS of mistakes does a variant make, not
just how many. Reads a variant's raw predictions and the matching gold file,
categorizes every disagreement, and writes a report with counts and sample
examples per category.

This is read-only analysis over files that already exist -- it never writes
to results/results.csv and never influences a training or eval run. Per
CLAUDE.md, this is the one place in the project where an LLM is allowed
outside the measurement path: an optional pass (--llm-summary) that asks
Claude to read a sample of the hardest category (wrong-polarity confusions)
and summarize common themes in plain language. That pass is diagnostic only
-- it does not change any score and is never used to decide a result.

Usage:
    python error_analysis.py \
        --predictions results/predictions/lora-r8-mlp-synth-eval_restaurants_test.jsonl \
        --gold data/processed/restaurants_test.jsonl \
        --out results/error_analysis/lora-r8-mlp-synth.json

    # add a qualitative pass over wrong-polarity cases:
    python error_analysis.py ... --llm-summary
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from data.schema import GoldExample
from eval.absa_eval import normalize_term, parse_prediction

load_dotenv()


def load_gold(path: str) -> list[GoldExample]:
    golds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                golds.append(GoldExample.from_json(json.loads(line)))
    return golds


def load_predictions(path: str) -> dict[str, str]:
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["id"]] = d["raw_output"]
    return preds


def categorize(golds: list[GoldExample], raw_predictions: dict[str, str]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = defaultdict(list)

    for gold in golds:
        raw = raw_predictions.get(gold.id, "")
        parsed = parse_prediction(raw)
        if parsed is None:
            categories["unparseable_output"].append({
                "id": gold.id, "text": gold.text, "raw_output": raw[:300],
            })
            continue

        gold_remaining = [(normalize_term(a.term), a.term, a.polarity) for a in gold.aspects]
        pred_items = [(normalize_term(p["term"]), p["term"], p["polarity"]) for p in parsed]

        for pred_norm, pred_term, pred_polarity in pred_items:
            match_idx = next((i for i, g in enumerate(gold_remaining) if g[0] == pred_norm), None)
            if match_idx is None:
                categories["hallucinated_aspect"].append({
                    "id": gold.id, "text": gold.text,
                    "pred_term": pred_term, "pred_polarity": pred_polarity,
                })
                continue
            _, gold_term, gold_polarity = gold_remaining.pop(match_idx)
            if gold_polarity != pred_polarity:
                categories[f"wrong_polarity:{gold_polarity}->{pred_polarity}"].append({
                    "id": gold.id, "text": gold.text, "term": gold_term,
                    "gold_polarity": gold_polarity, "pred_polarity": pred_polarity,
                })
            # else: correct, not an error -- not recorded

        for _, gold_term, gold_polarity in gold_remaining:
            categories["missed_aspect"].append({
                "id": gold.id, "text": gold.text, "term": gold_term, "gold_polarity": gold_polarity,
            })

    return categories


def llm_summarize_wrong_polarity(categories: dict[str, list[dict]], sample_size: int = 15) -> str:
    import random

    import anthropic

    wrong_polarity_examples = []
    for cat, examples in categories.items():
        if cat.startswith("wrong_polarity:"):
            wrong_polarity_examples.extend(examples)

    if not wrong_polarity_examples:
        return "No wrong-polarity errors to summarize."

    rng = random.Random(42)
    sample = rng.sample(wrong_polarity_examples, min(sample_size, len(wrong_polarity_examples)))

    lines = [
        f'{i + 1}. "{ex["text"]}" -- term: "{ex["term"]}", gold: {ex["gold_polarity"]}, predicted: {ex["pred_polarity"]}'
        for i, ex in enumerate(sample)
    ]
    prompt = (
        "These are aspect-based sentiment analysis errors from a model: it correctly found the "
        "aspect term but assigned the wrong polarity. Read them and identify the 2-4 most common "
        "underlying patterns (e.g. sarcasm the model read literally, genuine ambiguity, conflict "
        "cases where the model picked only one side, understatement). Be concrete and cite which "
        "example numbers support each pattern. Keep it to a short paragraph, no preamble.\n\n"
        + "\n".join(lines)
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--llm-summary", action="store_true")
    parser.add_argument("--sample-per-category", type=int, default=5)
    args = parser.parse_args()

    golds = load_gold(args.gold)
    preds = load_predictions(args.predictions)
    categories = categorize(golds, preds)

    total_errors = sum(len(v) for v in categories.values())
    report = {
        "predictions_file": args.predictions,
        "gold_file": args.gold,
        "n_examples": len(golds),
        "total_errors": total_errors,
        "category_counts": {k: len(v) for k, v in sorted(categories.items(), key=lambda kv: -len(kv[1]))},
        "samples": {k: v[: args.sample_per_category] for k, v in categories.items()},
    }

    if args.llm_summary:
        report["wrong_polarity_llm_summary"] = llm_summarize_wrong_polarity(categories)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"n_examples={len(golds)}  total_errors={total_errors}")
    print("\ncategory                                   count   % of errors")
    for cat, count in report["category_counts"].items():
        pct = (count / total_errors * 100) if total_errors else 0.0
        print(f"{cat:<42} {count:>5}   {pct:>5.1f}%")
    if args.llm_summary:
        print("\n--- wrong-polarity themes (Claude Sonnet 5) ---")
        print(report["wrong_polarity_llm_summary"])
    print(f"\nfull report -> {out_path}")


if __name__ == "__main__":
    main()
