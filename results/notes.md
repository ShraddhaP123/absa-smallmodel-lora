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

(Fill in during Session 1: hand-label 3 examples, score your own labels
against gold with eval/absa_eval.py, record disagreements here. This is the
ceiling every model result gets compared against.)

## Observations
