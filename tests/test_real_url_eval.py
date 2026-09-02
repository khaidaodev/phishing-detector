"""
Tests for src/real_url_eval.py. Only covers the aggregation logic (evaluate_urls), using a fake
stand-in model instead of the real trained one, no real joblib file or scikit-learn version match
needed to run these.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from real_url_eval import evaluate_urls  # noqa: E402


class FakeModel:
    """Stands in for the real Random Forest. predict_proba just returns whatever probabilities
    were configured for it, keyed by call order, so tests can control exactly what "the model
    said" without needing real features or a real model."""

    def __init__(self, probas_by_category):
        self.probas_by_category = probas_by_category
        self.calls = 0

    def predict_proba(self, features):
        # real_url_eval.py calls this once per category, in the same order the dict iterates in
        category_probas = list(self.probas_by_category.values())[self.calls]
        self.calls += 1
        proba_phishing = np.array(category_probas)
        return np.column_stack([1 - proba_phishing, proba_phishing])


def test_evaluate_urls_counts_false_positives_correctly():
    model = FakeModel({
        "wikipedia": [0.1, 0.2, 0.05],
        "file_sharing": [0.9, 0.6, 0.4],
    })
    urls_by_category = {
        "wikipedia": ["https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B", "https://en.wikipedia.org/wiki/C"],
        "file_sharing": ["https://drive.google.com/a", "https://drive.google.com/b", "https://drive.google.com/c"],
    }

    results = evaluate_urls(model, urls_by_category)

    assert results["wikipedia"]["false_positives"] == 0
    assert results["wikipedia"]["false_positive_rate"] == 0.0
    assert results["file_sharing"]["false_positives"] == 2  # 0.9 and 0.6 are both >= 0.5
    assert results["file_sharing"]["false_positive_rate"] == 2 / 3


def test_evaluate_urls_boundary_counts_as_false_positive():
    model = FakeModel({"cat": [0.5]})
    results = evaluate_urls(model, {"cat": ["https://example.com"]})
    assert results["cat"]["false_positives"] == 1


def test_evaluate_urls_overall_aggregates_across_categories():
    model = FakeModel({
        "a": [0.1, 0.9],
        "b": [0.8, 0.2, 0.3],
    })
    urls_by_category = {
        "a": ["https://a.com/1", "https://a.com/2"],
        "b": ["https://b.com/1", "https://b.com/2", "https://b.com/3"],
    }

    results = evaluate_urls(model, urls_by_category)

    assert results["_overall"]["count"] == 5
    assert results["_overall"]["false_positives"] == 2  # 0.9 from a, 0.8 from b
    assert results["_overall"]["false_positive_rate"] == 2 / 5


def test_evaluate_urls_handles_all_correct():
    model = FakeModel({"cat": [0.0, 0.1, 0.2]})
    results = evaluate_urls(model, {"cat": ["https://x.com/1", "https://x.com/2", "https://x.com/3"]})
    assert results["cat"]["false_positives"] == 0
    assert results["_overall"]["false_positive_rate"] == 0.0
