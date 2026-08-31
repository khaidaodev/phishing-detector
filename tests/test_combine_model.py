"""
Tests for src/combine_model.py. Uses tiny fake stand-in models instead of the real trained ones,
so these run instantly and don't need text_baseline.joblib/url_baseline.joblib to exist on disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from combine_model import combine_scores, predict_combined  # noqa: E402


class FakeModel:
    """Stands in for a real sklearn model, always predicts the same probability regardless of
    input, just enough to check combine_model.py wires things together correctly."""

    def __init__(self, phishing_proba):
        self.phishing_proba = phishing_proba

    def predict_proba(self, X):
        return [[1 - self.phishing_proba, self.phishing_proba] for _ in X]


def test_combine_scores_with_no_url_just_uses_text_score():
    assert combine_scores(text_proba=0.8, url_proba=None) == 0.8


def test_combine_scores_takes_the_higher_of_the_two():
    assert combine_scores(text_proba=0.2, url_proba=0.9) == 0.9
    assert combine_scores(text_proba=0.9, url_proba=0.2) == 0.9


def test_predict_combined_without_a_url_skips_the_url_model():
    result = predict_combined("boring email", url=None, text_model=FakeModel(0.1))
    assert result["url_proba"] is None
    assert result["combined_proba"] == 0.1
    assert result["prediction"] == "legitimate"


def test_predict_combined_flags_phishing_when_either_model_is_confident():
    result = predict_combined(
        "boring sounding text",
        url="http://paypa1-secure-login.com",
        text_model=FakeModel(0.1),
        url_model=FakeModel(0.95),
    )
    assert result["text_proba"] == 0.1
    assert result["url_proba"] == 0.95
    assert result["combined_proba"] == 0.95
    assert result["prediction"] == "phishing"


def test_predict_combined_legitimate_when_both_models_agree_its_fine():
    result = predict_combined(
        "hey, lunch tomorrow?",
        url="https://www.google.com",
        text_model=FakeModel(0.05),
        url_model=FakeModel(0.02),
    )
    assert result["prediction"] == "legitimate"
