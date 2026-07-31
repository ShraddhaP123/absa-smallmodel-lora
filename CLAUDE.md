# ABSA Small-Model Project

Aspect-Based Sentiment Analysis (ABSA) on SemEval-2014 Task 4 (restaurants +
laptops). The deliverable is a **tradeoff table**: F1/accuracy vs. latency vs.
cost vs. hardware footprint, across a base model, a LoRA fine-tune, a
synthetic-data-augmented fine-tune, and a quantized/served version.

Everything in this file is a hard constraint, not a suggestion. Read it before
touching code.

## The three rules that keep this project comparable

1. **`eval/absa_eval.py` is the measuring instrument. Never modify it without
   re-running its self-test first.**
   Run `pytest eval/test_absa_eval.py -v` before AND after any change to
   `eval/absa_eval.py`. If you change scoring logic (normalization, matching,
   metric definitions) mid-project, every earlier row in `results/results.csv`
   becomes incomparable to every later row, and the whole project's central
   claim — "here is how these variants trade off" — is void. If a change to
   the scorer is genuinely necessary, it must be a deliberate decision (bump
   `SCORER_VERSION` in `eval/absa_eval.py`, note it in `results/notes.md`,
   and re-run every prior variant), never a silent "improvement."

2. **No agent frameworks (LangGraph, DSPy, Pydantic-AI, custom multi-agent
   harnesses, etc.) anywhere in the measurement path** — not in `eval/`, not
   in `data/`, not in the training or serving scripts. Batch ML pipelines
   should be boring and deterministic: same input, same output, every run.
   Nondeterministic routing/retry/chunking makes it impossible to attribute a
   metric change to the variable you're actually testing (quantization,
   synthetic data, LoRA rank, ...). The one exception is a standalone
   error-analysis tool (Session 6+) that reads failure logs after the fact —
   it must never write to `results/results.csv` or influence a run in
   progress.

3. **Every run appends one row to `results/results.csv`, tagged with the git
   SHA and a config hash.** No result should exist only in terminal
   scrollback or a notebook cell. Use the `/run-eval` skill, which does this
   automatically. If you score something outside that skill (e.g. a manual
   serving load test), use `/log-run` to record it in the same schema.

## Repo layout

```
data/       loaders + gold-format conversion (data/schema.py, data/load_semeval.py)
eval/       absa_eval.py (the scorer — protected), test_absa_eval.py (self-test), runner.py
variants/   one YAML config per model+quantization combo
serve/      Dockerfile + vLLM config for the served, quantized variant
results/    results.csv (append-only log), notes.md (freeform observations)
PLAN.md     the 7-session plan
```

## The four metrics, and why they're separate

ABSA here means two chained subtasks: **extract aspect terms**, then
**classify polarity** for each. Collapsing this into one number hides *where*
a model is failing, so we always report four:

- **`aspect_f1`** — F1 on aspect-term span extraction alone (ignores polarity).
  Tells you if the model can even find what's being talked about.
- **`sentiment_acc`** — polarity accuracy, computed only over aspects the
  model correctly extracted. Tells you, conditional on a correct extraction,
  whether it gets the sentiment right.
- **`joint_f1`** — F1 requiring both the span AND the polarity to match. This
  is the real end-to-end metric; the other two exist to explain it.
- **`parse_rate`** — fraction of model outputs that parse into valid
  structured predictions at all. A model can have a great `joint_f1` on the
  outputs it parses and still be bad in practice if `parse_rate` is low. Watch
  the gap between `parse_rate` and the other three — a low parse rate with
  decent scores-on-parsed-output usually means a prompting/format problem,
  not a capability problem, and shouldn't be read as "the small model can't
  do ABSA."

All four are computed by `eval/absa_eval.py::compute_metrics`. Do not compute
metrics ad hoc elsewhere.

## Dataset

`tomaarsen/setfit-absa-semeval-restaurants` and `-laptops` on the HF Hub
(SemEval-2014 Task 4, re-hosted in SetFit's ABSA format: one row per
`(text, span, label, ordinal)`). `data/load_semeval.py` converts this into our
gold JSONL format (one line per sentence, with a list of aspects) under
`data/processed/`. Polarity labels: `positive`, `negative`, `neutral`,
`conflict`. Decide up front whether `conflict` is folded into a class or
dropped, and record that decision in `results/notes.md` — don't let it drift
per-variant.

## Working discipline

- One week (one `PLAN.md` session) per Claude Code session. Commit at the end
  of each session.
- Training only happens on a GPU (Lightning AI Studio or Kaggle — see
  `PLAN.md`). Sessions 1, 2, and 7 are local/CPU-only; don't let a GPU
  requirement creep into those.
- When asked to write a training script (Session 3+), explain the LoRA config
  choices (rank, alpha, target_modules, dropout) in prose before/alongside the
  code. Rank and target modules are exactly what gets asked about later —
  generated-and-unread code teaches nothing.
- Don't add retries, fallbacks, or defensive error handling for scenarios that
  can't happen in a local script run by one person. Trust the config.

## Skills

- `/run-eval <variant> <domain> <split>` — run a variant end-to-end against
  an eval split and append the row to `results/results.csv`.
- `/log-run` — manually record a run (e.g. a serving/load test) in the same
  schema, when `/run-eval` doesn't apply.
- `/new-variant <name>` — scaffold a new `variants/<name>.yaml` from the
  template, keeping every config structured identically.
