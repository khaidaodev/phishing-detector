"""
Tests for src/combined_url_eval.py. Only covers the aggregation logic (evaluate_combined and
old_combine_scores), using fake stand-in models instead of the real trained ones, no real
joblib file or scikit-learn version match needed to run these.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combined_url_eval import evaluate_combined, old_combine_scores  # noqa: E402


class FakeTextModel:
    """Returns one fixed phishing probability per message, in the order predict_proba is
    called, so a test can control exactly what "the text model said" about each message."""

    def __init__(self, probas):
        self.probas = list(probas)
        self.calls = 0

    def predict_proba(self, X):
        proba = self.probas[self.calls]
        self.calls += 1
        return [[1 - proba, proba] for _ in X]


class FakeUrlModel:
    """Returns fixed phishing probabilities per category, keyed by call order, matching how
    tests/test_real_url_eval.py's FakeModel works."""

    def __init__(self, probas_by_category):
        self.probas_by_category = probas_by_category
        self.calls = 0

    def predict_proba(self, features):
        category_probas = list(self.probas_by_category.values())[self.calls]
        self.calls += 1
        proba_phishing = np.array(category_probas)
        return np.column_stack([1 - proba_phishing, proba_phishing])


def test_old_combine_scores_takes_the_higher_one():
    assert old_combine_scores(0.2, 0.9) == 0.9
    assert old_combine_scores(0.9, 0.2) == 0.9


def test_evaluate_combined_counts_false_positives_for_both_rules():
    # one boring message the text model is very sure is legit (0.02), one URL the url model is
    # only barely over the line on (0.6): the old "take the higher" rule flags it, the new
    # confidence-weighted one shouldn't
    text_model = FakeTextModel([0.02])
    url_model = FakeUrlModel({"gov_uk": [0.6]})
    urls_by_category = {"gov_uk": ["https://www.gov.uk/some-page"]}

    results = evaluate_combined(text_model, url_model, urls_by_category, ["boring message"])

    assert results["gov_uk"]["count"] == 1
    assert results["gov_uk"]["old_false_positives"] == 1
    assert results["gov_uk"]["new_false_positives"] == 0


def test_evaluate_combined_pairs_every_message_with_every_url():
    text_model = FakeTextModel([0.02, 0.03])
    url_model = FakeUrlModel({"wikipedia": [0.1, 0.2]})
    urls_by_category = {"wikipedia": ["https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B"]}

    results = evaluate_combined(
        text_model, url_model, urls_by_category, ["message one", "message two"]
    )

    # 2 messages x 2 urls = 4 pairs, none of them anywhere near phishing under either rule
    assert results["wikipedia"]["count"] == 4
    assert results["wikipedia"]["old_false_positives"] == 0
    assert results["wikipedia"]["new_false_positives"] == 0


def test_evaluate_combined_overall_aggregates_across_categories():
    text_model = FakeTextModel([0.15])
    url_model = FakeUrlModel({"a": [0.95], "b": [0.1]})
    urls_by_category = {"a": ["https://a.com"], "b": ["https://b.com"]}

    results = evaluate_combined(text_model, url_model, urls_by_category, ["boring message"])

    assert results["_overall"]["count"] == 2
    assert results["_overall"]["old_false_positives"] == 1  # only the 0.95 one, under either rule
    assert results["_overall"]["new_false_positives"] == 1
