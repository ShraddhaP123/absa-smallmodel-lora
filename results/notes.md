# Notes

Freeform log of decisions and observations that don't belong in results.csv
rows but affect how to read them. Append, don't rewrite history.

## Decisions

- Polarity classes: positive, negative, neutral, conflict (kept as-is, not
  folded together). Revisit here if that changes.
- Aspect matching in eval/absa_eval.py is exact-match on normalized term
  (lowercased, punctuation stripped), not span-position-based — the model
  only sees/produces text, not character offsets.

## Human baseline

n=3 restaurants test examples (restaurants-1161, -1561, -1372), sampled with
seed=7. Scored with eval/absa_eval.py (SCORER_VERSION 1.0.0):

  aspect_f1=0.167  sentiment_acc=1.0  joint_f1=0.167  parse_rate=1.0

NOT a statistically meaningful ceiling at n=3 — this was a qualitative check,
and it surfaced something more useful than the number itself:

**Finding:** the low F1 wasn't from getting sentiment wrong (sentiment_acc=1.0
on everything matched) — it was granularity mismatch. Gold annotation is
atomistic (e.g. "zucchini", "mashed potatoes", "garlic", "butter" are 4
separate aspects in one sentence about a single dish). The human instinct —
and likely a small model's default instinct too, absent explicit prompting —
is to summarize at the sentence/dish level ("the dish is positive") rather
than enumerate every noun phrase gold treats as its own aspect. One hit was
notable: independently reasoning "service = conflict" from "attentive, yet
unimposing" landed exactly on gold's label without ever seeing it.

Action item for Session 3+ prompting: the extraction prompt
(variants/prompts/absa_extract.txt) should probably say something like
"extract every distinct noun phrase mentioned, even if several belong to the
same dish/topic" — otherwise expect models to under-generate aspects the same
way this manual pass did, which would show up as a low aspect_f1 that looks
like a capability gap but is actually a granularity-instruction gap.

Disagreements on record:
  - restaurants-1161: labeled the dish as one positive unit; gold wants 4
    separate ingredient-level aspects (zucchini, mashed potatoes, garlic,
    butter), all positive.
  - restaurants-1561: caught service=conflict correctly; missed food,
    wine list, and "priced" as separate aspects entirely (all positive in
    gold).
  - restaurants-1372: gave an overall sentence judgment ("positive"); gold's
    actual aspects are "value" and "lunch", not "place" or the town.

## Observations
