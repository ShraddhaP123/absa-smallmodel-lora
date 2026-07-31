# Seven-session plan

One session per week, ~4 hours each. Commit at the end of every session.
GPU column tells you whether you need Lightning AI / Kaggle or can work
locally — see the GPU note at the bottom.

| Session | Focus | GPU? |
|---|---|---|
| 1 | Data | No |
| 2 | API baselines | No |
| 3 | LoRA fine-tune (first pass) | Yes |
| 4 | LoRA fine-tune (iterate) | Yes |
| 5 | Synthetic data + retrain | Yes |
| 6 | Quantize + serve + load test | Yes |
| 7 | Held-out test + writeup | No |

---

## Session 1 — Data (no GPU)

Goal: gold data exists and you have a human baseline. Nothing else.

```
python -m data.load_semeval --domain restaurants
python -m data.load_semeval --domain laptops
```

Then, by hand: pick 3 examples from `data/processed/restaurants_test.jsonl`,
label them yourself (aspects + polarity) before looking at gold, score your
labels against gold using `eval/absa_eval.py`, and write the result — and
your disagreements with gold — into `results/notes.md` under "Human
baseline." This number is the ceiling every later model result gets compared
against; skipping it means you have no idea whether a 0.72 F1 is good.

Claude Code prompt to paste:
> Run the two `data.load_semeval` commands above, then show me 3 restaurant
> test examples I can hand-label (don't show me their gold labels yet).

Don't touch model code this session.

## Session 2 — API baselines (no GPU)

Goal: know what a strong general model scores on this task before you
fine-tune anything smaller. This is your ceiling-from-above.

- `/new-variant api-gpt4o-mini` (backend: api, provider: openai)
- `/run-eval` it against restaurants test, then laptops test
- Optionally a second API variant (different provider/model) for comparison

Claude Code prompt to paste:
> Set up a new variant for gpt-4o-mini as an API backend and run it against
> both domains' test splits.

## Session 3 — LoRA fine-tune, first pass (GPU)

Goal: a working training script and one trained adapter, not a good score.

Ask Claude Code to write `train.py` (LoRA fine-tune of a small base model,
e.g. Qwen2.5-1.5B-Instruct, on `data/processed/restaurants_train.jsonl`
formatted with `variants/prompts/absa_extract.txt`) — and to explain the
`rank`, `alpha`, `target_modules`, and `dropout` choices before writing the
code. Read that explanation; it's the part of this project that's actually
new to you.

Then `/new-variant lora-r8` (backend: hf, lora_adapter pointing at the
checkpoint) and `/run-eval` it.

## Session 4 — LoRA fine-tune, iterate (GPU)

Goal: at least one deliberate ablation, isolating one variable.

Pick one axis to vary (rank, target_modules, epochs, or learning rate) and
create a second variant (`/new-variant lora-r16` or similar) that changes
only that axis. `/run-eval` both, compare against the Session 3 row and the
Session 2 API baseline in `results/results.csv`.

## Session 5 — Synthetic data + retrain (GPU)

Goal: test whether synthetic augmentation helps, isolated from every other
variable changed so far.

Generate synthetic ABSA examples (e.g. via an API model, since that's already
wired up from Session 2), retrain the best Session 3/4 config on
train+synthetic, `/new-variant synth-augmented`, `/run-eval`.

## Session 6 — Quantize, serve, load test (GPU)

Goal: the serving-economics row(s) of the tradeoff table.

- Merge the best LoRA adapter into the base model, quantize it
- Build `serve/Dockerfile`, run it, `/new-variant` with `backend: vllm`
  pointing at the served endpoint, `/run-eval` for accuracy
- `python serve/loadtest.py ...` for latency/throughput under concurrency,
  then `/log-run` to record hardware + cost + latency alongside the accuracy
  numbers already on that variant's row

This is also where a standalone error-analysis tool (reads
`results/predictions/*.jsonl`, categorizes failures) earns its keep — build
it as a separate script, not inside `eval/` or `data/`, and it must never
write to `results/results.csv`. See CLAUDE.md's rule #2 for why.

## Session 7 — Held-out test + writeup (no GPU)

Goal: final numbers and the tradeoff table, not new code.

Re-run every finalist variant (base, best LoRA, synthetic-augmented,
quantized-served) against a held-out split you haven't looked at yet, then
build the final table from `results/results.csv` — don't recompute numbers
by hand, pull directly from the log. Write up: the tradeoffs, the human
baseline comparison, the parse-rate-vs-capability distinction (CLAUDE.md), and
whatever the error-analysis pass surfaced as the interesting failure taxonomy.

---

## GPU note

Kaggle gives you no root/SSH — browser notebook only, incompatible with using
Claude Code directly against the GPU machine. For Sessions 3–6, prefer
**Lightning AI Studios** (persistent dev environment, SSH, VS Code-like UI,
15 free credits/month ≈ 22 hours on a T4, survives between sessions unlike
Colab/Kaggle). Install Claude Code inside the Studio and work there directly.

If you use Kaggle instead (30 GPU hrs/week, more hours but worse dev
experience), keep the notebook to five lines — clone the repo, install
requirements, run a script — and do zero real logic in notebook cells. All
code stays in this repo, written locally with Claude Code, pushed to GitHub
first. Checkpoint to the HF Hub, not local disk — Kaggle has no persistent
volume.

If you switch GPU providers, update the `hardware:` field convention in
`variants/*.yaml` and this file so `results.csv` stays honest about where
numbers came from.
