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

### Session 4 — ablation: add MLP target modules (r8-mlp vs r8)

Single-variable change from Session 3: same rank (8), alpha (16), data,
epochs (3), learning rate — only target_modules changed, adding
gate_proj/up_proj/down_proj (the MLP/feedforward layers) alongside the
same four attention projections.

  r8 attn-only  (n=304): aspect_f1=0.762  sentiment_acc=0.803  joint_f1=0.612  parse_rate=0.990
  r8 + MLP      (n=304): aspect_f1=0.780  sentiment_acc=0.810  joint_f1=0.632  parse_rate=0.993

As a fraction of Sonnet's ceiling (0.686 joint_f1 / 0.785 aspect_f1 / 0.874
sentiment_acc / 0.997 parse_rate): joint_f1 92.0% (was 89.2%), aspect_f1
99.3% (was 97.0%) -- essentially matches Sonnet's aspect-finding now --
sentiment_acc 92.7% (was 91.9%), parse_rate 99.7% (was 99.3%).

**Finding, and it's more nuanced than the hypothesis predicted.** The
hypothesis going in was "MLP adaptation should specifically help polarity
judgment, since attention only decides what relates to what." That's not
actually what happened: aspect_f1 improved the most (+2.4% relative),
sentiment_acc barely moved (+0.8% relative). So the sentiment-accuracy gap
vs Sonnet is still essentially unclosed after this change -- adding MLP
capacity made the model better at *finding* aspects, not meaningfully
better at *judging sentiment* once it found them. That's a real result
even though it doesn't confirm the hypothesis: it suggests the remaining
sentiment_acc gap looks more like a genuine capability ceiling for this
1.7B model on this task than something more LoRA capacity alone fixes.
Worth testing directly in a future session (more/better training data
specifically for ambiguous-sentiment examples, rather than more model
capacity).

**Cost of this change, not free:** trainable params went from 3,145,728
(0.18% of the model) to 9,043,968 (0.53%) -- roughly 2.9x more -- and wall-
clock training time went from ~1090s to 1297s (+19%). Both train and eval
loss were lower at every epoch than the attn-only run (eval_loss:
0.133 -> 0.116 -> 0.113, vs 0.142 -> 0.126 -> 0.124), so the extra capacity
is buying real generalization, not just overfitting -- but it's a real
tradeoff to weigh against a ~3% relative joint_f1 gain, especially once
Session 6's serving-cost numbers are in.

This run's Kaggle cell combined setup+train+eval+`cat results.csv` into a
single cell (lesson from Session 3's session-reset losses) and captured
real latency for the first time: p50=1396.6ms, p95=3618.1ms. Not
apples-to-apples against Sonnet's API latency (different serving
substrate entirely -- local `model.generate()` vs a network API call) and
there's no comparable number for the Session 3 r8 attn-only run (lost to
the session reset), so this can't yet be compared against that variant's
latency, only against Sonnet's as a rough reference point.

### Session 5 — synthetic data targeting conflict/sarcasm/neutral/understatement

Single-variable change from Session 4 (the r8+MLP config): added 290
synthetic examples (see generate_synthetic.py) to the real 1,413-example
training set, everything else held constant. real conflict-class aspects
were only 2.4% of the real training set (62/2588); synthetic added 72 more,
specifically targeting the sentiment-judgment gap Session 4 surfaced.

  r8+MLP           (n=304): aspect_f1=0.780  sentiment_acc=0.810  joint_f1=0.632  parse_rate=0.993
  r8+MLP+synth     (n=304): aspect_f1=0.783  sentiment_acc=0.813  joint_f1=0.636  parse_rate=1.000

As a fraction of Sonnet's ceiling: joint_f1 92.8% (was 92.0%), aspect_f1
99.7% (was 99.3%), sentiment_acc 93.0% (was 92.7%), parse_rate now
literally 100% -- fractionally *higher* than Sonnet's own 99.7% on this
domain.

**Finding, and it's a repeat of Session 4's pattern -- worth taking
seriously rather than explaining away.** This intervention was specifically
designed to fix the sentiment_acc gap (72 extra conflict examples, plus
sarcasm/neutral/understatement categories chosen exactly because Session 4
didn't move that number). It barely moved: 0.810 -> 0.813, a 0.35% relative
gain -- smaller than the improvement from adding MLP target modules in
Session 4, and this time with 20% more training data and 25 more minutes
of compute. Two different interventions in a row (more model capacity,
then targeted data) have both failed to meaningfully close sentiment_acc's
gap to Sonnet (93.0% of ceiling now vs 91.9% in Session 3 -- a ~1 point
total movement across two full sessions of work aimed directly at it).

**Working conclusion for Session 7's writeup:** the sentiment_acc gap
increasingly looks like a real capability ceiling of this 1.7B base model
on ambiguous-sentiment judgment (sarcasm, mixed cues, subtle neutrality),
not a data-quantity or adapter-capacity problem that this project's scale
of intervention can close. That's a legitimate, useful conclusion for the
final report -- it's a specific, falsifiable claim (backed by two
independent negative results) about *where* a 1.7B model's limits are
relative to a frontier model on this task, which is more informative than
if everything had just gotten better.

**What actually did move, unexpectedly:** parse_rate hit a perfect 1.0
(0/304 unparseable), up from 0.993. Possible explanation: the synthetic
examples are somewhat more templated/formulaic in structure than the real
SemEval sentences (each written to a fairly narrow category prompt), which
may have reinforced output-formatting consistency more than it improved
semantic sentiment reasoning -- consistent with the aspect_f1/parse_rate
axis moving while sentiment_acc doesn't. Worth checking in Session 6's
error analysis whether the *kinds* of sentiment errors changed at all,
even if the aggregate rate didn't.

**Self-labeled synthetic data caveat:** these gold labels were assigned by
Claude Sonnet 5 itself, not independently verified the way the real
SemEval annotations were (see the Session 1 human-baseline entry above for
what that verification process even looks like). A model grading its own
homework on exactly the judgment calls we're trying to teach is a real
limitation -- if Sonnet's own conflict/sarcasm labels have systematic
blind spots, this data could reinforce rather than correct them. Manual
spot-checking a sample of the 290 synthetic examples would be worth doing
before trusting this dataset for anything beyond this one experiment.

### Session 6 — quantize + serve (best adapter, 4-bit, vLLM)

Merged the Session 5 adapter (r8+MLP+synth, the best checkpoint so far)
into the base model via merge_adapter.py, then served it through vLLM with
on-the-fly 4-bit bitsandbytes quantization (--quantization bitsandbytes
--load-format bitsandbytes -- no calibration dataset needed). Ran on
Kaggle directly (no Docker; Kaggle notebooks have no Docker daemon --
serve/Dockerfile remains a "how you'd really deploy this" reference, not
something actually run for this project's own numbers).

Real bug caught before running: eval/runner.py's vLLM backend and
serve/loadtest.py were both hitting vLLM's /chat/completions endpoint,
which auto-applies the tokenizer's chat template -- but training used a
raw, unwrapped prompt (matching the hf backend exactly). Fixed both to use
plain /completions with a "prompt" field, so the served model sees the
exact format it was trained on, same principle as the train/inference
consistency rule from Session 3.

**Accuracy cost of quantization** (same adapter, same eval set, only
precision changed):

  unquantized fp16 (Session 5): aspect_f1=0.783  sentiment_acc=0.813  joint_f1=0.636  parse_rate=1.000
  4-bit quantized  (Session 6): aspect_f1=0.740  sentiment_acc=0.775  joint_f1=0.574  parse_rate=0.957

joint_f1 dropped 9.9% relative -- the largest single-cause accuracy drop in
the whole project so far, bigger than any gain from Sessions 4 or 5
combined. parse_rate also regressed (13/304 unparseable, vs 0 before).
This is a real, structural tradeoff -- unlike the earlier ablations, this
one is squarely "give something up to get something else," not a free
improvement.

**Latency: sequential vs. concurrent, and this is the most informative
number in the whole project so far.**

  sequential (eval run, 1 request at a time): p50=798ms   p95=2183ms
  concurrent load test (concurrency=8, n=100): p50=3968ms  p95=16490ms  throughput=1.34 req/s

Run one request at a time, this quantized 1.7B model is *faster* than
Sonnet's API (798ms vs Sonnet's 1581ms in Session 2) -- unsurprising, no
network hop, much smaller model. But at just 8 concurrent requests, p50
jumps 5x and p95 blows out to 16.5 seconds. **A single T4 cannot serve
this model to even a handful of simultaneous users without severe
degradation.** This is exactly the kind of finding the "hardware
footprint" axis of the tradeoff table exists to surface -- the small
model's per-request cost advantage over an API is real, but only holds at
low concurrency; serving it to real traffic would need either more/bigger
GPUs or a lower concurrency ceiling than the API alternative, which has
its own (opaque, presumably much larger) serving infrastructure behind it.

Both measurements are logged as separate rows (served-vllm for the
sequential/eval numbers, served-vllm-loadtest for the concurrent numbers)
per the /log-run convention -- the load-test row reuses the same accuracy
metrics rather than re-scoring, since it's the same served model.

**Pip noise, not a real failure:** installing vllm produced a large block
of "ERROR: pip's dependency resolver..." messages about version conflicts
in packages this project never uses (bigframes, google-adk, gradio, the
cudf/cuml/dask-cuda RAPIDS stack -- all part of Kaggle's base image). The
server started and served correctly regardless; these were pip being
noisy about unrelated parts of the environment, not a vllm install
failure. Worth remembering if this comes up again: check whether the
package in the warning is actually imported anywhere in this project
before treating a pip dependency-conflict report as blocking.

**Still to do:** the error-analysis tool (read results/predictions/*.jsonl
across variants, categorize failure types) hasn't been built yet --
flagged in CLAUDE.md as belonging in Session 6 alongside quantize/serve,
but the serving work took priority. Worth doing before Session 7's
writeup, now that there are 4 restaurant-domain prediction sets to compare
(r8, r8+mlp, r8+mlp+synth, and the quantized-served version) plus Sonnet's.

### Session 7 — held-out final numbers + error-analysis tool built + biggest finding of the project

**Methodology note, stated plainly for the writeup:** "test" was reused
across Sessions 2-6 to compare every variant and make real decisions (add
MLP modules, add synthetic data) based on its numbers -- that makes it a
de facto dev set, not a clean held-out set, despite the filename. "val"
was only ever used for the Trainer's internal loss monitoring (a
different metric -- LM loss, not ABSA F1) during training, never for
cross-variant comparisons, so it's the genuinely untouched split. All
numbers below are on val (restaurants, n=302), for exactly the three
finalist variants: Sonnet 5 (ceiling), the best fp16 LoRA
(r8+MLP+synth), and that same adapter merged + 4-bit quantized + served
via vLLM.

  Sonnet 5 (ceiling):        aspect_f1=0.775  sentiment_acc=0.880  joint_f1=0.682  parse_rate=1.000
  LoRA fp16 (r8+MLP+synth):  aspect_f1=0.763  sentiment_acc=0.810  joint_f1=0.618  parse_rate=0.997
  4-bit quantized, served:   aspect_f1=0.715  sentiment_acc=0.794  joint_f1=0.568  parse_rate=0.960

As % of Sonnet's ceiling: LoRA fp16 reaches 90.6% joint_f1 (98.4%
aspect_f1, 92.1% sentiment_acc); quantized drops to 83.3% joint_f1 (92.3%
aspect_f1, 90.3% sentiment_acc). These are consistent with the test-split
numbers throughout the project (89-92% range) -- the held-out check
confirms the earlier numbers weren't split-specific flukes.

**The error-analysis tool (error_analysis.py) is built and validated** --
categorizes every prediction/gold disagreement into missed_aspect,
hallucinated_aspect, and wrong_polarity:X->Y confusion pairs, with an
optional Claude-assisted qualitative pass over the wrong-polarity sample
(read-only, never affects a score, per CLAUDE.md's one sanctioned
LLM-outside-the-measurement-path exception). One real bug fixed along the
way: the LLM pass failing (no ANTHROPIC_API_KEY on Kaggle) was crashing
the whole script *before* the reliable deterministic report got saved or
printed -- fixed so the optional step can never take the reliable core
down with it.

**The biggest single finding in this entire project** -- run on all three
finalists' held-out val predictions:

  category           Sonnet   LoRA fp16   4-bit quantized
  missed_aspect       59.4%      44.2%         27.7%
  hallucinated_aspect 22.5%      31.4%         47.8%
  wrong_polarity      18.0%      24.1%         21.6%
  unparseable          0.0%       0.3%          2.9%

This is not noise -- missed_aspect and hallucinated_aspect trade off
almost perfectly monotonically as the model shrinks/compresses (Sonnet ->
fp16 LoRA -> 4-bit quantized). **A frontier model is conservative: its
main failure mode is under-generating, missing implicit or subtly-phrased
aspects. A heavily compressed small model is the opposite: its main
failure mode is over-generating, flagging things as aspects that aren't
one at all.** This is a categorically different failure signature, not a
scaled-down version of the same weaknesses -- similar overall aspect_f1
numbers across variants (0.775 / 0.763 / 0.715) were masking a real shift
in *how* that number is composed (different precision/recall balance),
which the aggregate F1 alone completely hides. This is exactly the kind
of result the error-analysis tool exists to surface, and it's a much
stronger, more specific claim for the final writeup than "the small model
is somewhat worse."

**Secondary finding, worth stating as a hypothesis, not a proven cause:**
in the LoRA fp16 breakdown, `neutral->positive` is the single largest
specific wrong-polarity confusion (7.6% of all its errors), and
`conflict->positive` shows up prominently (2.7%) -- neither is a notable
pattern for Sonnet. Session 5's synthetic data included an "understatement"
category specifically teaching the model to read soft/hedged language as
leaning positive; it's plausible that generalized into over-reading
positive sentiment where gold says neutral or conflict. Not confirmed --
would need a targeted follow-up (e.g. an ablation removing just the
understatement category) to actually test this causally.

**Known gap:** the LLM-assisted qualitative theme summary only ran
successfully for Sonnet (has ANTHROPIC_API_KEY locally); the LoRA and
quantized variants' runs were on Kaggle, which has no API key configured,
so only their deterministic category counts are logged -- the specific
sample examples/themes for those two remain on that Kaggle session
if deeper qualitative digging is wanted later. The deterministic counts
above are the load-bearing finding regardless; the qualitative pass adds
color, not the core result.

**Update: the gap above is now closed.** The LoRA and quantized variants'
raw predictions were retrieved from Kaggle (pasted back into chat and
reconstructed locally -- 302 lines each, verified byte-identical to
Kaggle's own run by re-deriving the exact same category counts locally).
Both now have full qualitative theme summaries. This also let us test a
specific mechanistic hypothesis about *why* hallucinated_aspect jumps so
much under quantization, not just observe that it does.

**Refined finding: quantization causes two distinct, separately-diagnosed
failure modes, not one.**

1. **Repetition-loop degeneration, causing parse failures.** All 12 of
   the quantized variant's unparseable_output cases (100%, not "some of
   them") are the exact same mechanism: the model gets stuck repeating a
   single term (`"staff"` x20, `"price"` x19, `"food"` x16, `"place"`
   x16, `"portraits"` x5...) until truncated by the 256-token limit
   before the JSON array can close. This is not scattered formatting
   mistakes -- it's one specific, mechanistically identifiable pathology
   (precision loss destabilizing autoregressive decoding under greedy
   sampling), confirmed by checking every unparseable case individually.
   Critically, `restaurants-1539` (the `"staff"` loop) triggers this
   failure in **both** the fp16 and quantized versions of the same
   adapter -- the underlying instability already exists in the fp16
   model on this specific input; 4-bit quantization doesn't invent a new
   failure mode, it amplifies an existing one roughly **12x** (fp16:
   1/302 = 0.3%, quantized: 12/302 = 4.0%).
2. **Genuinely distributed over-prediction, separate from #1.** The
   47.8% hallucinated_aspect share is *not* an artifact of a few
   degenerate examples inflating the count -- checked directly: examples
   contributing 5+ hallucinations each account for only 13.8% of the
   total hallucination count (27 of 195). The rest is spread thinly
   across many examples, each predicting one or two genuinely distinct
   (not repeated) extra/wrong terms. This is a real, broad precision
   problem, not a statistical artifact of a handful of pathological
   generations.

These are different problems needing different fixes: #1 (repetition
loops) could likely be mitigated cheaply -- a repetition penalty, a
lower max_tokens with a stricter stop sequence, or a light constrained-
decoding grammar -- without touching the LoRA weights at all. #2 (broad
over-prediction) is a genuine capability/calibration issue that
quantization is making worse, not a decoding-parameter fix.

**Full category breakdown, all three finalists, held-out val split:**

  category             Sonnet   LoRA fp16   4-bit quantized
  missed_aspect          59.4%      44.2%           27.7%
  hallucinated_aspect    22.5%      31.4%           47.8%
  wrong_polarity         18.0%      24.1%           21.6%
  unparseable_output      0.0%       0.3%            4.0% (was 2.9% of
                                                       errors; 4.0% of examples)

**Qualitative wrong-polarity themes, condensed across all three** (full
text in results/error_analysis/*.json):

- **Sonnet:** literal reading of negation/irony (missing that "NO more
  reservations" is praise); negative sentiment "bleeding" onto neutral
  aspects just mentioned nearby; conflict cases collapsed to one
  polarity; comparative sentiment misjudged.
- **LoRA fp16:** implied/indirect sentiment missed (situational cues
  without explicit sentiment words); negation/litotes misread ("don't
  look like leafy road kill" = praise, misread as neutral); comparative/
  backhanded-compliment structures where the model picks one clause's
  polarity over the sentence's actual target; mild/hedged descriptions
  over-amplified into a stronger polarity than warranted.
- **4-bit quantized:** the same core patterns as fp16 (hedged language
  read as too strong, spillover sentiment from adjacent context,
  contrastive/comparative structures resolved to the wrong clause) --
  quantization doesn't introduce new *qualitative* confusion types, it
  makes the *existing* ones (already present in fp16) somewhat more
  frequent, on top of the two structural failure modes above.

**Bottom line for the Session 7 writeup:** three independent lines of
evidence -- the missed/hallucinated F1 trade-off, the qualitative theme
overlap between fp16 and quantized, and the shared repetition-loop
trigger example -- all point the same direction: going from Sonnet to a
small fine-tuned model to a quantized version of that same model doesn't
just make errors *more frequent*, it changes *what kind* of model you're
dealing with (conservative and under-generating vs. liberal and
over-generating, plus a decoding-stability cost specific to
quantization). That's a genuinely different, more actionable story than
"accuracy goes down as you compress the model," and it's a direct
product of building the error-analysis tool rather than stopping at
aggregate metrics.
