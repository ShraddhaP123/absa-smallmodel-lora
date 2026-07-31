"""The ABSA scorer. This is the project's measuring instrument.

DO NOT modify this file without reading the note in CLAUDE.md first:
run `pytest eval/test_absa_eval.py -v` before AND after any change. If the
change alters scoring behavior (normalization, matching, metric definitions),
bump SCORER_VERSION, note the change and the reason in results/notes.md, and
re-run every variant already in results/results.csv — old and new rows are
only comparable if they were scored by the same logic.

Prediction format: for each gold example, the model's raw output is a string
that should parse into a JSON list of {"term": str, "polarity": str} objects,
e.g.  [{"term": "battery life", "polarity": "negative"}]

Four metrics (see CLAUDE.md for why they're kept separate):
  - aspect_f1      : F1 on aspect-term extraction alone (term match, polarity ignored)
  - sentiment_acc  : polarity accuracy, conditioned on correctly extracted terms
  - joint_f1       : F1 requiring both term AND polarity to match
  - parse_rate     : fraction of raw outputs that parsed into valid predictions
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data.schema import GoldExample

SCORER_VERSION = "1.0.0"

VALID_POLARITIES = {"positive", "negative", "neutral", "conflict"}

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_term(term: str) -> str:
    term = term.lower().strip()
    term = _PUNCT_RE.sub("", term)
    term = _WHITESPACE_RE.sub(" ", term)
    return term.strip()


def parse_prediction(raw: str) -> list[dict] | None:
    """Parse a raw model output into a list of {"term", "polarity"} dicts.

    Returns None if the output doesn't parse into that shape at all — that's
    what parse_rate measures. Malformed individual items (e.g. a missing
    polarity, or a polarity outside VALID_POLARITIES) are dropped rather than
    failing the whole parse, since that's a partial-credit case, not a
    format failure.
    """
    raw = raw.strip()
    # Be lenient about models wrapping JSON in a code fence.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[len("json"):]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        polarity = item.get("polarity")
        if not isinstance(term, str) or not term.strip():
            continue
        if polarity not in VALID_POLARITIES:
            continue
        result.append({"term": term, "polarity": polarity})
    return result


@dataclass(frozen=True)
class MetricsResult:
    aspect_f1: float
    sentiment_acc: float
    joint_f1: float
    parse_rate: float
    n_examples: int
    n_gold_aspects: int
    n_parsed: int

    def to_dict(self) -> dict:
        return {
            "aspect_f1": round(self.aspect_f1, 4),
            "sentiment_acc": round(self.sentiment_acc, 4),
            "joint_f1": round(self.joint_f1, 4),
            "parse_rate": round(self.parse_rate, 4),
            "n_examples": self.n_examples,
            "n_gold_aspects": self.n_gold_aspects,
            "n_parsed": self.n_parsed,
            "scorer_version": SCORER_VERSION,
        }


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(golds: list[GoldExample], raw_predictions: dict[str, str]) -> MetricsResult:
    """Score a set of raw model outputs against gold examples.

    raw_predictions maps GoldExample.id -> raw model output string. Every
    gold id must have an entry (use "" for a missing/failed generation —
    it will correctly parse as None and count against parse_rate).
    """
    aspect_tp = aspect_fp = aspect_fn = 0
    joint_tp = joint_fp = joint_fn = 0
    n_parsed = 0
    n_gold_aspects = 0

    for gold in golds:
        raw = raw_predictions.get(gold.id, "")
        parsed = parse_prediction(raw)
        if parsed is not None:
            n_parsed += 1
        pred_aspects = parsed or []

        gold_terms = [normalize_term(a.term) for a in gold.aspects]
        gold_joint = [(normalize_term(a.term), a.polarity) for a in gold.aspects]
        n_gold_aspects += len(gold_terms)

        pred_terms = [normalize_term(p["term"]) for p in pred_aspects]
        pred_joint = [(normalize_term(p["term"]), p["polarity"]) for p in pred_aspects]

        gold_terms_remaining = list(gold_terms)
        for t in pred_terms:
            if t in gold_terms_remaining:
                aspect_tp += 1
                gold_terms_remaining.remove(t)
            else:
                aspect_fp += 1
        aspect_fn += len(gold_terms_remaining)

        gold_joint_remaining = list(gold_joint)
        for j in pred_joint:
            if j in gold_joint_remaining:
                joint_tp += 1
                gold_joint_remaining.remove(j)
            else:
                joint_fp += 1
        joint_fn += len(gold_joint_remaining)

    aspect_f1 = _f1(aspect_tp, aspect_fp, aspect_fn)
    joint_f1 = _f1(joint_tp, joint_fp, joint_fn)
    # sentiment_acc: of the aspects we got the term right on, how often did
    # we also get the polarity right. aspect_tp counts term-matches (with
    # duplicates removed per-example the same way joint_tp does), so this is
    # joint_tp / aspect_tp by construction.
    sentiment_acc = (joint_tp / aspect_tp) if aspect_tp else 0.0
    parse_rate = (n_parsed / len(golds)) if golds else 0.0

    return MetricsResult(
        aspect_f1=aspect_f1,
        sentiment_acc=sentiment_acc,
        joint_f1=joint_f1,
        parse_rate=parse_rate,
        n_examples=len(golds),
        n_gold_aspects=n_gold_aspects,
        n_parsed=n_parsed,
    )


def _load_gold_jsonl(path: str) -> list[GoldExample]:
    golds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                golds.append(GoldExample.from_json(json.loads(line)))
    return golds


def _load_predictions_jsonl(path: str) -> dict[str, str]:
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["id"]] = d["raw_output"]
    return preds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="gold JSONL (data/processed/*.jsonl)")
    parser.add_argument("--pred", required=True, help="predictions JSONL with {id, raw_output} per line")
    args = parser.parse_args()

    golds = _load_gold_jsonl(args.gold)
    preds = _load_predictions_jsonl(args.pred)
    metrics = compute_metrics(golds, preds)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
