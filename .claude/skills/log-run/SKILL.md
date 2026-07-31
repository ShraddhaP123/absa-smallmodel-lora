---
name: log-run
description: Manually record a run in results/results.csv when /run-eval doesn't apply — e.g. a serving/load test, a hardware/cost measurement, or any result produced outside eval/runner.py. Use whenever the user reports numbers from outside the automated pipeline that should still land in the shared results table.
---

# log-run

`eval/runner.py` (used by `/run-eval`) only knows how to produce accuracy
metrics from a variant config. Some results come from elsewhere — a
`serve/loadtest.py` run, a manual cost calculation from a cloud bill, a
hand-timed benchmark — and still need to end up in the same
`results/results.csv` table so the final tradeoff comparison has every
variant in one place. This skill is that manual path.

## Steps

1. **Get the raw numbers from the user or the tool output** — don't invent
   or estimate anything (latency percentiles, throughput, cost, hardware
   type). If a number wasn't measured, leave that CSV field blank rather than
   guessing.

2. **Check whether this variant already has a row from `/run-eval`.** If so,
   reuse its `aspect_f1` / `sentiment_acc` / `joint_f1` / `parse_rate` rather
   than re-deriving them — this row is adding serving/cost data to an
   existing accuracy result, not re-scoring the model. If there's no prior
   accuracy row, leave those fields blank rather than fabricating them.

3. **Append one row** to `results/results.csv` matching the exact column
   order in the header (`timestamp,git_sha,config_hash,variant_name,backend,
   model,quantization,hardware,domain,split,n_examples,aspect_f1,
   sentiment_acc,joint_f1,parse_rate,latency_p50_ms,latency_p95_ms,
   cost_per_1k_examples_usd,notes`). Use:
   - `timestamp`: now, ISO-ish (`%Y-%m-%dT%H:%M:%S`)
   - `git_sha`: `git rev-parse --short HEAD`
   - `config_hash`: `nomanual` if there's no variant YAML behind this row
   - `backend`: `manual` for anything not run through `eval/runner.py`
   - `notes`: what was measured and how (e.g. "vLLM load test, concurrency=8,
     A10G, serve/loadtest.py")

4. **Also append a line to `results/notes.md`** if there's any caveat a bare
   CSV row can't capture (e.g. "cold-start excluded from p95", "cost
   estimated from Lightning's per-credit rate, not a real invoice").

5. Never overwrite or delete existing rows — append only.
