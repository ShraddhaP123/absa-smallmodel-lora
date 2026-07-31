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

### Session 2 — Claude Sonnet 5 API baseline (ceiling-from-above)

Full test splits, not a sample. See results.csv for exact numbers.

  restaurants (n=304): aspect_f1=0.785  sentiment_acc=0.874  joint_f1=0.686  parse_rate=0.997
  laptops     (n=223): aspect_f1=0.788  sentiment_acc=0.867  joint_f1=0.683  parse_rate=1.000

Strikingly consistent across domains (~0.69 joint_f1 either way). Two things
worth carrying into later sessions:

1. **sentiment_acc isn't 1.0 even for a frontier model** (0.87 both domains).
   So a chunk of the joint_f1 gap vs aspect_f1 is genuine polarity-judgment
   difficulty in this dataset (sarcasm, mixed sentences), not just something
   a bigger/better model would trivially fix. Don't assume a small model's
   sentiment errors are purely a capability gap — some of this ceiling is
   irreducible from the data itself.
2. **parse_rate is a red herring for API models, but won't be for the LoRA
   target model.** Sonnet parses at ~99-100% basically for free. The small
   model in Session 3+ is a genuinely open question here — if its parse_rate
   comes in noticeably below Sonnet's, that's a real signal, not noise.

Technical gotcha hit while building this (fixed in eval/runner.py, not a
data issue): claude-sonnet-5 runs extended thinking by default and it can
eat the entire max_tokens budget before emitting the actual JSON answer
(saw stop_reason=max_tokens with 230/256 tokens spent thinking, output
empty). Fixed by passing thinking={"type":"disabled"} for the anthropic
backend. Also: this model rejects an explicit `temperature` param outright
(400, "deprecated for this model") rather than ignoring it — runner.py only
sends it when the config's temperature is non-zero.

### Session 3 — first LoRA fine-tune (r=8, SmolLM2-1.7B-Instruct, restaurants only)

  LoRA r=8 (n=304): aspect_f1=0.762  sentiment_acc=0.803  joint_f1=0.612  parse_rate=0.990

As a fraction of the Session 2 Sonnet ceiling on the same domain (0.686 joint_f1):
89.2% of Sonnet's joint_f1, 97.0% of aspect_f1, 91.9% of sentiment_acc, 99.3%
of parse_rate — from a 1.7B model, 3.1M trainable LoRA params (0.18% of the
model), 18 minutes of training on a free Kaggle T4.

**Finding:** aspect_f1 and parse_rate are both within ~1-3 points of Sonnet —
the small model finds nearly as many correct aspects and produces valid JSON
almost as reliably. The real gap is sentiment_acc (0.803 vs 0.874): once it
finds the right aspect, it's noticeably more likely to get the polarity wrong
than Sonnet is. That's what compounds into the wider joint_f1 gap. Suggests
Session 4's ablations should focus on things that might help polarity
judgment specifically (more epochs, MLP target modules, or more training
data) rather than assuming aspect extraction is the bottleneck.

Training was healthy and reproducible: two independent runs (one lost to a
Kaggle session reset, retrained from scratch) produced near-identical loss
curves. Both train and eval loss decreased every epoch (eval_loss:
0.142 -> 0.126 -> 0.124) with no sign of overfitting at 3 epochs, though the
epoch-2-to-3 improvement was small — 3 epochs is a reasonable stopping point
for this first pass, not obviously under- or over-trained.

**Known gap in this row:** `latency_p50_ms`/`latency_p95_ms` are blank —
Kaggle's interactive notebook session recycled its kernel between the
training+eval cell finishing and a follow-up cell trying to read
results.csv back (a known Kaggle free-tier quirk: the interactive kernel can
be recycled between cell executions, wiping /kaggle/working, independent of
whether the notebook's own outputs are still visible). The four accuracy
metrics come directly from the run's own printed console output, which
*was* preserved; git_sha and config_hash were computed locally against the
exact commit (186f584) Kaggle had cloned, since nothing changed in the repo
in between. Latency wasn't guessed or backfilled — left blank per the
project's rule against fabricating unmeasured fields. If per-example latency
is needed later, rerun with a version of the notebook that captures
`!cat results/results.csv` inside the same cell as training, not a
follow-up one.
