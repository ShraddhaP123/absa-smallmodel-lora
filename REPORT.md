# ABSA Small-Model LoRA Project — Final Report

**Task:** Aspect-based sentiment analysis (ABSA) on restaurant reviews (SemEval-2014 Task 4).
**Question:** How close can a small (1.7B), LoRA-fine-tuned, and eventually quantized/served model get to a frontier model's accuracy, and what do you give up along the way?
**Compute:** Free-tier Kaggle T4 GPU for all fine-tuning and serving; Claude Sonnet 5 API for the ceiling baseline and synthetic data generation.

All numbers below are pulled directly from [`results/results.csv`](results/results.csv) and [`results/notes.md`](results/notes.md) — nothing here is recomputed by hand.

---

## 1. The final tradeoff table

Evaluated on the **val split** (302 restaurant reviews) — the one split in this project never used to make a comparison or a design decision during development, so these numbers are a genuine held-out estimate rather than a number this project optimized against. See §3 for why "test" (used everywhere else in this report) couldn't serve that role.

| Variant | Params trained | joint_f1 | % of ceiling | aspect_f1 | sentiment_acc | parse_rate | latency p50 | latency p95 |
|---|---|---|---|---|---|---|---|---|
| **Claude Sonnet 5** (ceiling, zero-shot) | — | **0.682** | 100% | 0.775 | 0.879 | 1.000 | 1365 ms¹ | 2682 ms¹ |
| **LoRA fp16** (r=8, +MLP modules, +290 synthetic examples) | 9.0M (0.53%) | **0.618** | 90.6% | 0.763 | 0.810 | 0.997 | 1303 ms | 3807 ms |
| **4-bit quantized, served (vLLM)** | *(same adapter)* | **0.568** | 83.3% | 0.715 | 0.794 | 0.960 | 803 ms | 2468 ms |

¹ Sonnet's latency is a network API round-trip; the LoRA/quantized rows are local single-GPU inference. Not directly comparable as "hardware cost," but informative as "what a user actually waits for."

**The one number that matters most and doesn't show up in this table:** under a **concurrency=8** load test (same quantized model, `results.csv` row `served-vllm-loadtest`), p50 latency rose to **3968 ms** and p95 to **16,490 ms** — a single T4 cannot serve this model to even a handful of simultaneous users without severe degradation. Throughput topped out at 1.34 req/s. This is the real hardware-footprint finding of the project: the small model's per-request speed advantage over an API (803ms vs Sonnet's 1365ms) is real, but it only holds at low concurrency.

---

## 2. What was actually built

- A **LoRA adapter** (rank 8, targeting all attention projections and the MLP/feedforward layers) fine-tuned on 1,413 real SemEval training examples plus 290 targeted synthetic examples, using [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) as the base model.
- A **merged, 4-bit quantized version** of that adapter, served through vLLM's OpenAI-compatible API (`--quantization bitsandbytes --load-format bitsandbytes`, no calibration dataset).
- A **synthetic data generator** ([`generate_synthetic.py`](generate_synthetic.py)) that used Claude Sonnet 5 to write and self-label 290 examples specifically targeting conflict polarity, sarcasm, ambiguous-neutral statements, and understatement — categories chosen because they were underrepresented in the real training data (`conflict` was only 2.4% of real training aspects) and because they were hypothesized to explain the model's sentiment-judgment gap.
- A **standalone error-analysis tool** ([`error_analysis.py`](error_analysis.py)) that categorizes every prediction/gold disagreement (missed aspect, hallucinated aspect, specific wrong-polarity confusions) and optionally asks Claude to summarize qualitative themes in a sample — read-only, never affects a score, never runs during training.
- A reproducible eval pipeline ([`eval/runner.py`](eval/runner.py)) supporting three backends (Anthropic API, local HuggingFace/PEFT, vLLM), all scored by the same protected scorer ([`eval/absa_eval.py`](eval/absa_eval.py), unit-tested, versioned).

---

## 3. Methodology notes worth stating explicitly

**The scorer never changed mid-project.** All numbers across all seven sessions were produced by the same `eval/absa_eval.py` (`SCORER_VERSION 1.0.0`), guarded by 12 unit tests that must pass before and after any edit — this is what makes numbers from Session 2 and Session 7 comparable at all.

**"test" is not a clean held-out set, despite the name.** It was used repeatedly across Sessions 2–6 to compare every variant and make real decisions (add MLP modules because test `joint_f1` went up; add synthetic data for the same reason). That makes it a de facto *development* set. The **"val" split** was only ever touched by the training loop's internal loss monitoring (a different metric — language-model loss, not `aspect_f1`) and was never used to compare variants — so it's the split used for the final table in §1. The Session 7 val numbers came back consistent with the test-split numbers throughout this report (e.g. Sonnet: 0.682 vs 0.686), which is itself a reassuring sign the earlier test-split numbers weren't overfit flukes — but the val numbers, not the test numbers, are the honest final estimate.

**Data pipeline bug caught early.** The Hugging Face re-hosting of this dataset (`tomaarsen/setfit-absa-semeval-restaurants`) ships its "test" split with every label blanked out — an artifact of the tutorial it was built for. Caught in Session 1 and fixed by building a seeded 70/15/15 train/val/test split from the labeled data instead of trusting the source's own split boundary.

**Human baseline was informative, not a statistic.** Hand-labeling 3 restaurant sentences (Session 1) is not a sample size that supports a real ceiling estimate — but it surfaced a genuine, useful finding anyway: 100% sentiment accuracy against 16.7% joint F1, because gold annotation is far more atomistic than natural human summarization (one dish → four separate ingredient-level aspects). That finding shaped how the extraction prompt was worded from Session 2 onward.

---

## 4. Development path (test split, all four sessions of iteration)

These are the *development* numbers — used throughout the project to decide what to try next. Not the final numbers (see §1), but the record of how the project got there.

| Session | Variant | joint_f1 | Δ vs prior |
|---|---|---|---|
| 2 | Claude Sonnet 5 (ceiling) | 0.686 | — |
| 3 | LoRA r=8, attention only | 0.612 | (first pass) |
| 4 | + MLP target modules | 0.632 | **+3.2%** |
| 5 | + 290 synthetic examples | 0.636 | +0.8% |
| 6 | + 4-bit quantization, served | 0.573 | −9.9% |

**Session 4 (MLP modules):** hypothesized this would specifically help sentiment judgment (attention decides *what* relates to *what*; reasoned this wouldn't touch *how well the model judges tone*). It didn't confirm that hypothesis — `aspect_f1` improved more (+2.4%) than `sentiment_acc` (+0.8%). Cost: ~2.9x more trainable parameters (9.0M vs 3.1M), +19% training time.

**Session 5 (synthetic data):** targeted the sentiment-judgment gap directly — 290 examples built specifically around conflict polarity, sarcasm, and understatement, more than doubling the model's exposure to the `conflict` class. `sentiment_acc` again barely moved (0.810 → 0.813, +0.35% relative). Two independent interventions in a row failed to meaningfully close that specific gap — a genuine negative result, not a null one, since it's backed by two different attempted mechanisms.

**Session 6 (quantization):** the first genuinely *lossy* tradeoff in the project — everything else had been a close-to-free improvement. 4-bit quantization cost 9.9% relative `joint_f1`, the single largest accuracy drop from any change made.

---

## 5. What kind of mistakes each model makes (not just how many)

Aggregate F1 scores looked like a gentle, unremarkable decline across variants (aspect_f1: 0.775 → 0.763 → 0.715). The error-analysis tool showed that aggregate number was masking something much more specific:

| Failure category | Sonnet | LoRA fp16 | 4-bit quantized |
|---|---|---|---|
| Missed aspect (under-generation) | **59.4%** | 44.2% | 27.7% |
| Hallucinated aspect (over-generation) | 22.5% | 31.4% | **47.8%** |
| Wrong polarity | 18.0% | 24.1% | 21.6% |
| Unparseable output | 0.0% | 0.3% | 4.0% |

**As the model shrinks and compresses, its error character flips almost monotonically.** A frontier model is conservative — it mostly fails by *not mentioning* an aspect at all (implicit or subtly-phrased ones). A heavily compressed small model is the opposite — it mostly fails by *flagging things that aren't really aspects*. Similar overall `aspect_f1` numbers across variants were hiding a real shift in precision/recall balance.

**Digging into *why* the quantized model hallucinates so much more revealed two separate, differently-fixable problems, not one:**

1. **Repetition-loop degeneration (100% of the quantized model's parse failures).** Every one of its 12 unparseable outputs is the same mechanism: the model gets stuck repeating a single term (`"staff"` ×20, `"price"` ×19, `"place"` ×16...) until truncated by the token limit before the JSON can close. One trigger example (`restaurants-1539`) causes this in *both* the fp16 and quantized versions of the same adapter — the instability already exists in fp16; quantization amplifies it roughly **12x** (0.3% → 4.0% of examples), it doesn't create a new failure mode. This is plausibly fixable cheaply (a repetition penalty, a stricter stop sequence) without touching the model weights.
2. **Genuinely distributed over-prediction, separate from #1.** Checked directly whether a handful of degenerate examples were skewing the 47.8% hallucination rate — they're not. Examples with 5+ hallucinations each account for only 13.8% of the total; the rest is spread thinly across many examples each predicting one or two distinct wrong terms. This is a real calibration problem that quantization worsens, not a decoding artifact.

**Qualitative themes (Claude-assisted reading of a wrong-polarity sample, full text in `results/error_analysis/*.json`):** Sonnet's errors cluster around literal reading of negation/irony and sentiment "bleeding" from nearby context onto neutral aspects. The LoRA model (both fp16 and quantized) shares a largely overlapping set of confusions — implied/indirect sentiment missed, negation/litotes misread, contrastive structures resolved to the wrong clause, mild/hedged language over-amplified — suggesting quantization intensifies existing weaknesses rather than introducing qualitatively new ones.

---

## 6. Key findings, ranked by how much they'd change a real deployment decision

1. **A single T4 cannot serve this model under real concurrent load.** 803ms → 3968ms p50 latency (1 → 8 concurrent requests). Any real deployment plan needs to budget for this explicitly, not extrapolate from single-request latency.
2. **Quantization has a real, non-trivial accuracy cost here** (−9.9% relative `joint_f1`), and it's driven by two distinguishable causes needing two different fixes — a cheap decoding fix (repetition loops) and a harder calibration problem (broad over-prediction) that shouldn't be conflated.
3. **The sentiment-judgment gap to Sonnet resisted two different, well-targeted interventions** (more LoRA capacity, then targeted synthetic data). That's a specific, falsifiable claim about where this 1.7B model's limits are on ambiguous-sentiment judgment (sarcasm, mixed cues, subtle neutrality) — not just "the small model is worse."
4. **Aggregate F1 hid a real shift in failure character.** Two models can post similar `aspect_f1` numbers for entirely different reasons (one missing things, one hallucinating things) — a finding that only exists because of building the error-analysis tool rather than stopping at the four headline metrics.
5. **Self-labeled synthetic data is a real, acknowledged limitation.** The 290 synthetic examples were generated *and* labeled by Claude Sonnet 5 — if Sonnet has systematic blind spots on exactly the judgment calls (sarcasm, conflict) this project cared most about, that data could reinforce rather than correct them. Not independently verified the way the real SemEval annotations were.

---

## 7. Limitations and what would come next

- **Single domain for fine-tuning.** All LoRA training and quantization work targeted the restaurants domain only; Sonnet's laptops-domain baseline exists (`results.csv`) but no fine-tuned variant was trained or evaluated on laptops.
- **No calibration-dataset quantization method was tried** (e.g. AWQ/GPTQ) — only bitsandbytes' on-the-fly 4-bit, chosen for zero setup cost. A calibrated method might close some of the 9.9% accuracy gap; untested here.
- **The repetition-loop fix is a hypothesis, not a validated result.** A repetition penalty or stricter stop sequence is the natural next experiment, not something this project actually tried.
- **Kaggle's ephemeral session model cost real data** — Session 3's concurrent-load latency and one training run had to be redone after an interactive kernel recycled mid-project (documented in `results/notes.md`); later sessions worked around this by combining setup+train+eval+result-capture into single notebook cells.
- **n=3 human baseline** is a qualitative signal, not a statistically supported ceiling estimate.

---

## 8. Repository map

```
data/              loaders, gold-format schema, the SemEval blanked-test-label fix
eval/              absa_eval.py (protected scorer), runner.py (3-backend eval pipeline)
variants/          one YAML config per model/quantization/split combination
train.py           LoRA fine-tuning (supports mixing in synthetic data)
merge_adapter.py   bakes a LoRA adapter into full model weights for serving
generate_synthetic.py  targeted synthetic data generation + validation
error_analysis.py  post-hoc failure categorization (missed/hallucinated/wrong-polarity)
serve/             Dockerfile (real-deployment reference) + concurrent load-test script
results/           results.csv (every run, git-SHA-tagged), notes.md (the full narrative),
                   error_analysis/ (per-variant failure breakdowns), predictions/ (raw outputs)
```

Full narrative, including every dead end, bug, and mid-project correction, is in [`results/notes.md`](results/notes.md) — this report is the summary; that file is the record.
