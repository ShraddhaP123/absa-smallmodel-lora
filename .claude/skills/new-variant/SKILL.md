---
name: new-variant
description: Scaffold a new variants/<name>.yaml config from the variants/base.yaml template. Use whenever the user wants to set up a new model/quantization/LoRA combination to evaluate.
---

# new-variant

Every variant config must have the same shape so runs stay comparable and
`eval/runner.py` can dispatch on `backend` without special-casing. This skill
creates `variants/<name>.yaml` from `variants/base.yaml` rather than writing
one from scratch each time.

## Steps

1. **Get the name and backend from the user.** Name should be short and
   descriptive of what's being varied (e.g. `lora-r8`, `lora-r16-4bit`,
   `api-gpt4o-mini`, `synth-augmented`). Backend is one of `api`, `hf`,
   `vllm` (see `CLAUDE.md` / `variants/base.yaml` for what each needs).

2. **Copy `variants/base.yaml` to `variants/<name>.yaml`.**

3. **Set `name:` inside the file to match the filename exactly** — the
   runner and results.csv both key off this field, and a mismatch produces
   confusing rows.

4. **Fill in only the backend section that applies**; leave the other two
   backend blocks in place (unused, harmless) rather than deleting them —
   keeping the same structure across files makes diffs between variants
   meaningful (e.g. `diff variants/lora-r8.yaml variants/lora-r16.yaml`
   should show exactly the one thing that changed).

5. **State back to the user what changed vs. the template** (domain, model,
   quantization, LoRA rank/target_modules if applicable) so it's clear this
   variant isolates one variable. If it's a LoRA variant, this is also the
   moment to ask what's being tested (rank? target_modules? both?) — don't
   let two things vary in one new variant, or the results won't isolate
   anything.

6. Don't run the variant as part of this skill — that's `/run-eval`.
