"""Self-test for eval/absa_eval.py — the scorer's contract.

Run this before AND after any edit to absa_eval.py:
    pytest eval/test_absa_eval.py -v

These are hand-computed expected values on toy examples. If a change to the
scorer breaks one of these, that change alters what every existing row in
results/results.csv means — see CLAUDE.md before proceeding.
"""
from data.schema import Aspect, GoldExample
from eval.absa_eval import compute_metrics, normalize_term, parse_prediction


def test_normalize_term_strips_case_punct_whitespace():
    assert normalize_term("Battery  Life!") == "battery life"
    assert normalize_term("  screen ") == "screen"


def test_parse_prediction_valid_json():
    raw = '[{"term": "food", "polarity": "positive"}]'
    assert parse_prediction(raw) == [{"term": "food", "polarity": "positive"}]


def test_parse_prediction_handles_code_fence():
    raw = '```json\n[{"term": "food", "polarity": "positive"}]\n```'
    assert parse_prediction(raw) == [{"term": "food", "polarity": "positive"}]


def test_parse_prediction_rejects_non_list():
    assert parse_prediction('{"term": "food", "polarity": "positive"}') is None


def test_parse_prediction_rejects_garbage():
    assert parse_prediction("the food was great") is None


def test_parse_prediction_drops_invalid_items_keeps_valid():
    raw = '[{"term": "food", "polarity": "positive"}, {"term": "x", "polarity": "amazing"}]'
    assert parse_prediction(raw) == [{"term": "food", "polarity": "positive"}]


def _gold(id_, text, aspects):
    return GoldExample(id=id_, text=text, domain="test", aspects=aspects)


def test_perfect_prediction_scores_one_everywhere():
    golds = [_gold("1", "great food", [Aspect("food", "positive")])]
    preds = {"1": '[{"term": "food", "polarity": "positive"}]'}
    m = compute_metrics(golds, preds)
    assert m.aspect_f1 == 1.0
    assert m.sentiment_acc == 1.0
    assert m.joint_f1 == 1.0
    assert m.parse_rate == 1.0


def test_right_term_wrong_polarity():
    golds = [_gold("1", "bad food", [Aspect("food", "negative")])]
    preds = {"1": '[{"term": "food", "polarity": "positive"}]'}
    m = compute_metrics(golds, preds)
    assert m.aspect_f1 == 1.0        # term matched
    assert m.sentiment_acc == 0.0    # polarity wrong
    assert m.joint_f1 == 0.0         # term+polarity must both match
    assert m.parse_rate == 1.0


def test_missed_aspect_and_hallucinated_aspect():
    golds = [_gold("1", "food and service", [
        Aspect("food", "positive"), Aspect("service", "negative"),
    ])]
    # model finds "food" correctly, misses "service", hallucinates "price"
    preds = {"1": '[{"term": "food", "polarity": "positive"}, {"term": "price", "polarity": "neutral"}]'}
    m = compute_metrics(golds, preds)
    # aspect: tp=1 (food), fp=1 (price), fn=1 (service) -> P=0.5 R=0.5 F1=0.5
    assert m.aspect_f1 == 0.5
    assert m.joint_f1 == 0.5
    assert m.sentiment_acc == 1.0    # of the term-matches (food), polarity was right


def test_unparseable_output_counts_as_all_missed_and_lowers_parse_rate():
    golds = [_gold("1", "great food", [Aspect("food", "positive")])]
    preds = {"1": "not json at all"}
    m = compute_metrics(golds, preds)
    assert m.parse_rate == 0.0
    assert m.aspect_f1 == 0.0
    assert m.joint_f1 == 0.0
    assert m.sentiment_acc == 0.0


def test_missing_prediction_id_treated_as_empty():
    golds = [_gold("1", "great food", [Aspect("food", "positive")])]
    m = compute_metrics(golds, {})
    assert m.parse_rate == 0.0
    assert m.n_gold_aspects == 1


def test_parse_rate_is_independent_of_scoring_correctness():
    golds = [
        _gold("1", "great food", [Aspect("food", "positive")]),
        _gold("2", "bad service", [Aspect("service", "negative")]),
    ]
    preds = {
        "1": '[{"term": "food", "polarity": "negative"}]',  # parses, wrong polarity
        "2": "garbage",                                       # doesn't parse
    }
    m = compute_metrics(golds, preds)
    assert m.parse_rate == 0.5
    assert m.n_examples == 2
