---
name: run-eval
description: Run a variant end-to-end against an eval split and log the result to results/results.csv. Use whenever the user wants to evaluate a model variant (base, fine-tuned, quantized) on the ABSA task.
---

# run-eval

Runs `eval/runner.py` for a given variant config and appends one row to
`results/results.csv`, tagged with the current git SHA and a hash of the
config file. This is the standard way to produce a comparable result in this
project — see `CLAUDE.md` for why ad hoc scoring isn't allowed.

## Steps

1. **Protect the scorer.** Before running anything, confirm
   `eval/absa_eval.py` hasn't been edited uncommitted (`git status`). If it
   has, stop and run `pytest eval/test_absa_eval.py -v` first — do not score
   anything with an unverified scorer.

2. **Resolve the variant.** The user will name a variant (e.g. `base-api`,
   `lora-r8`) or give a path. Resolve it to `variants/<name>.yaml`. If it
   doesn't exist, suggest `/new-variant <name>` instead of guessing at one.

3. **Confirm the gold data exists.** Check
   `data/processed/<domain>_<split>.jsonl` exists (domain/split come from the
   variant config). If missing, run:
   ```
   python -m data.load_semeval --domain <domain>
   ```

4. **Run it.**
   ```
   python -m eval.runner --variant variants/<name>.yaml --notes "<one-line context>"
   ```
   The `--notes` field should say what's being tested (e.g. "lora rank=8
   before synthetic augmentation") — it lands directly in the results.csv row
   and is the only place that context survives.

5. **Report the four metrics** (`aspect_f1`, `sentiment_acc`, `joint_f1`,
   `parse_rate`) to the user, and flag anything that looks off — in
   particular a `parse_rate` well below the others, which usually means a
   prompt/format issue rather than a capability issue (see CLAUDE.md).

6. **Do not hand-edit `results/results.csv`.** If a row needs correcting,
   re-run the variant rather than editing the CSV — the git SHA / config hash
   pairing is what makes rows trustworthy.
