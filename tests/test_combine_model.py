"""
Tests for src/combine_model.py. Uses tiny fake stand-in models instead of the real trained ones,
so these run instantly and don't need text_baseline.joblib/url_baseline.joblib to exist on disk.
"""

import sys
from pathlib import Path

import pytest

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


def test_combine_scores_lands_between_the_two_scores():
    # a confidence-weighted average should never go outside the range the two inputs span
    combined = combine_scores(text_proba=0.2, url_proba=0.9)
    assert 0.2 <= combined <= 0.9


def test_combine_scores_lets_a_confident_legit_text_score_overrule_a_barely_over_the_line_url_score():
    # this is the actual bug stage 6 found: the URL model sometimes only just tips over 0.5 on a
    # real, legitimate link (gov.uk URLs averaged 0.64, see README). A text model that's very
    # sure the message itself is boring and legit should be able to pull that back under 0.5,
    # rather than one barely-confident wrong signal deciding the whole thing
    combined = combine_scores(text_proba=0.03, url_proba=0.6)
    assert combined < 0.5


def test_combine_scores_still_flags_a_url_the_model_is_very_confident_about():
    # the flip side: a URL score that's *very* confident (not just barely over 0.5) should still
    # carry real weight even against fairly boring-sounding text, an obviously alarming link
    # shouldn't get diluted away just because the wording around it is calm
    combined = combine_scores(text_proba=0.1, url_proba=0.97)
    assert combined >= 0.5


def test_combine_scores_weighs_by_distance_from_a_coin_flip_not_which_is_higher():
    # two scores the same distance from 0.5 but on opposite sides should land almost exactly in
    # the middle, since they're equally confident and disagree completely, not just "whichever
    # happened to be entered second" or "whichever is numerically higher"
    combined = combine_scores(text_proba=0.2, url_proba=0.8)
    assert combined == pytest.approx(0.5)


def test_predict_combined_without_a_url_skips_the_url_model():
    result = predict_combined("boring email", url=None, text_model=FakeModel(0.1))
    assert result["url_proba"] is None
    assert result["combined_proba"] == 0.1
    assert result["prediction"] == "legitimate"


def test_predict_combined_flags_phishing_when_either_model_is_very_confident():
    result = predict_combined(
        "boring sounding text",
        url="http://paypa1-secure-login.com",
        text_model=FakeModel(0.1),
        url_model=FakeModel(0.95),
    )
    assert result["text_proba"] == 0.1
    assert result["url_proba"] == 0.95
    # no longer just equal to whichever score was higher, it's a blend, but the very confident
    # url score should still be enough to tip the combined score into "phishing"
    assert result["combined_proba"] > 0.5
    assert result["prediction"] == "phishing"


def test_predict_combined_legitimate_when_both_models_agree_its_fine():
    result = predict_combined(
        "hey, lunch tomorrow?",
        url="https://www.google.com",
        text_model=FakeModel(0.05),
        url_model=FakeModel(0.02),
    )
    assert result["prediction"] == "legitimate"
