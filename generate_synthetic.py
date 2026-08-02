"""Generate synthetic ABSA training examples targeting the specific gap found
in Session 4: the model's sentiment_acc lags Sonnet's more than its aspect_f1
does, and generic data augmentation wouldn't necessarily touch that. Instead
of asking for "more restaurant reviews," this asks Claude Sonnet 5 for
specific categories of *hard sentiment judgment* cases: conflict polarity,
sarcasm, and statements that read as neutral but could be mistaken for
positive/negative.

This is self-labeled synthetic data (the same model both writes the sentence
and assigns its own gold labels) -- a real limitation worth being honest
about in results/notes.md: these labels are not independently verified the
way the real SemEval annotations are. Every generated aspect term is
validated to be an exact substring of its sentence (matching the constraint
real training data has) and every polarity is checked against the valid set;
anything that fails validation is dropped, not repaired.

Usage:
    python generate_synthetic.py --domain restaurants --out data/processed/restaurants_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from data.schema import Aspect, GoldExample
from eval.absa_eval import VALID_POLARITIES

load_dotenv()

CATEGORIES = [
    (
        "conflict polarity",
        "Write sentences where a SINGLE aspect gets both praise and criticism in "
        "the same sentence (e.g. 'the service was fast but rude'). The correct "
        "polarity for that aspect is \"conflict\" -- not positive, not negative, "
        "not two separate entries. Do not use aspects that are purely positive "
        "or purely negative in these sentences.",
    ),
    (
        "sarcasm and backhanded compliments",
        "Write sentences where the literal wording sounds positive but the real "
        "sentiment is negative (sarcasm, backhanded compliments), or the reverse "
        "-- wording that sounds harsh but the real sentiment is actually positive "
        "(self-deprecating praise, dry humor). Label the polarity as the model's "
        "best judgment of the REAL intended sentiment, not the literal words.",
    ),
    (
        "ambiguous neutral statements",
        "Write factual, neutral statements about an aspect that a careless reader "
        "might mislabel as positive or negative, but a careful reader would call "
        "neutral (e.g. describing a menu item's price or a wait time factually, "
        "without praise or complaint). The correct polarity is \"neutral\".",
    ),
    (
        "understatement and hedged sentiment",
        "Write sentences using understatement or hedging ('not bad', 'could have "
        "been worse', 'nothing special') where the real sentiment is mild but "
        "clearly leaning positive or negative -- not neutral, not conflict. "
        "Label the polarity as the actual leaning, not literally 'neutral' just "
        "because the wording is soft.",
    ),
]

PROMPT_TEMPLATE = """You are creating training data for an aspect-based sentiment analysis model on {domain} reviews.

Task focus: {category_instruction}

Write {n} distinct, realistic {domain} review sentences fitting that focus. Vary sentence length, aspect terms, and phrasing -- do not repeat the same aspect or sentence structure across examples.

For each sentence, list every aspect term mentioned and its polarity (positive, negative, neutral, or conflict). Each "term" MUST be copied verbatim as it appears in the sentence -- exact substring, same casing, no paraphrasing.

Respond with ONLY a JSON array, no prose, no code fence, in this exact format:
[{{"text": "<sentence>", "aspects": [{{"term": "<exact substring from text>", "polarity": "<positive|negative|neutral|conflict>"}}]}}]
"""


def call_sonnet(prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tokens,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def parse_batch(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[len("json"):]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def validate_example(item: dict) -> GoldExample | None:
    text = item.get("text")
    raw_aspects = item.get("aspects")
    if not isinstance(text, str) or not text.strip() or not isinstance(raw_aspects, list):
        return None

    aspects = []
    text_lower = text.lower()
    for a in raw_aspects:
        if not isinstance(a, dict):
            continue
        term = a.get("term")
        polarity = a.get("polarity")
        if not isinstance(term, str) or not term.strip():
            continue
        if polarity not in VALID_POLARITIES:
            continue
        if term.lower() not in text_lower:  # must be an exact substring, per training format
            continue
        aspects.append(Aspect(term=term, polarity=polarity))

    if not aspects:
        return None
    return GoldExample(id="", text=text, domain="", aspects=aspects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="restaurants")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-batch", type=int, default=25)
    parser.add_argument("--batches-per-category", type=int, default=3)
    args = parser.parse_args()

    kept: list[GoldExample] = []
    n_generated = 0
    n_dropped = 0

    for category_name, category_instruction in CATEGORIES:
        for batch_i in range(args.batches_per_category):
            prompt = PROMPT_TEMPLATE.format(
                domain=args.domain,
                category_instruction=category_instruction,
                n=args.n_per_batch,
            )
            raw = call_sonnet(prompt, max_tokens=4096)
            items = parse_batch(raw)
            n_generated += len(items)
            for item in items:
                ex = validate_example(item)
                if ex is None:
                    n_dropped += 1
                    continue
                kept.append(ex)
            print(f"[{category_name}] batch {batch_i + 1}/{args.batches_per_category}: "
                  f"{len(items)} generated, running total kept={len(kept)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, ex in enumerate(kept):
            ex = GoldExample(id=f"synth-{args.domain}-{i}", text=ex.text, domain=args.domain, aspects=ex.aspects)
            f.write(json.dumps(ex.to_json()) + "\n")

    print(f"\ngenerated {n_generated} raw examples, dropped {n_dropped} "
          f"(failed validation: bad JSON shape, invalid polarity, or term not "
          f"a verbatim substring), kept {len(kept)} -> {out_path}")


if __name__ == "__main__":
    main()
