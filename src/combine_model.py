"""
Combines the text model (src/text_baseline.py) and the link model (src/url_baseline.py) into
one phishing score for a real message.

First thing I tried: stick the two models' training data together and train one model on top
of both (proper "stacking"). Turns out that doesn't actually work here, checked it properly
before writing any of this. The email dataset has a `urls` column saying whether a message had
a link in it, but not the actual URL, most of the real URLs got stripped out when the original
emails were converted from HTML to plain text (links became text like "Update Your Account"
with the underlying href gone). Only a small fraction of rows still have an actual usable URL
sitting in the text. On top of that, the URL model was trained on the completely separate
PhiUSIIL dataset, so there's no shared set of examples that have both a real email body AND a
real URL to combine features from in the first place.

So instead this combines the two models at prediction time, not training time: for a brand new
message someone actually pastes in (with its link still intact, unlike the training data),
run the text model on the message body and the URL model on the link, then combine the two
scores. No joint training data needed for that, and it's arguably more realistic anyway, this
is exactly what a live detector would see.

Run it with:
    python src/combine_model.py
"""

from pathlib import Path

import joblib
import pandas as pd

from url_features import urls_to_feature_frame

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# used by combine_scores below. A score sitting right on 0.5 shouldn't count for nothing (that'd
# divide by zero if both models landed exactly on 0.5), this floor just keeps every score worth
# at least a small vote.
CONFIDENCE_FLOOR = 0.05


def combine_scores(text_proba: float, url_proba: float | None) -> float:
    """Combines the two models' phishing probabilities into one score for the message.

    If there's no URL in the message at all, the URL model never got a vote, so this is just
    the text model's score.

    If there is a URL, this used to just take whichever of the two scores was higher. Stage 6
    (see README) found the URL model is still confidently wrong on some real, everyday links,
    GitHub and Stack Overflow URLs especially, sometimes scoring a totally normal link as 90%+
    phishing. "Take the higher score" means one confidently wrong model is all it takes to flag
    someone's boring email as phishing, the text model doesn't get a say at all.

    So this now does a confidence-weighted average instead: each model's score is weighted by
    how far it sits from 0.5, a score of 0.5 is a coin flip and barely worth listening to, a
    score of 0.97 is the model actually being sure about something. That way a text model that's
    very sure a message is boring and legit can outvote a URL model that's only mildly over the
    0.5 line on a link it's misreading, but a URL model that's *very* confident (a real phishing
    link, or a link the URL model badly misjudges) still carries real weight, an obviously
    alarming link doesn't just get diluted away by boring text wrapped around it.

    This isn't a full fix, a URL the model is confidently, extremely wrong about (like GitHub
    links, still ~99% phishing after stage 6) can still tip a boring message over 0.5, weighting
    by confidence can't fix a model that's confidently wrong, only a model that's mildly wrong.
    Measured with src/combined_url_eval.py against the real URLs from stage 6, see the README
    for the actual before/after numbers.
    """
    if url_proba is None:
        return text_proba
    text_weight = abs(text_proba - 0.5) + CONFIDENCE_FLOOR
    url_weight = abs(url_proba - 0.5) + CONFIDENCE_FLOOR
    return (text_proba * text_weight + url_proba * url_weight) / (text_weight + url_weight)


def predict_combined(message_text: str, url: str | None = None, text_model=None, url_model=None) -> dict:
    """Scores one message (and optionally a URL found inside it) and returns everything: each
    model's own score plus the combined one. text_model/url_model can be passed in directly
    (mainly so tests don't need the real trained models on disk), otherwise this loads the
    saved ones from models/.
    """
    if text_model is None:
        text_model = joblib.load(MODELS_DIR / "text_baseline.joblib")
    text_proba = float(text_model.predict_proba([message_text])[0][1])

    url_proba = None
    if url:
        if url_model is None:
            url_model = joblib.load(MODELS_DIR / "url_baseline.joblib")
        features = urls_to_feature_frame(pd.Series([url]))
        url_proba = float(url_model.predict_proba(features)[0][1])

    combined_proba = combine_scores(text_proba, url_proba)

    return {
        "text_proba": text_proba,
        "url_proba": url_proba,
        "combined_proba": combined_proba,
        "prediction": "phishing" if combined_proba >= 0.5 else "legitimate",
    }


if __name__ == "__main__":
    examples = [
        {
            "message_text": "Your account has been suspended, click here to verify your identity immediately.",
            "url": "http://paypa1-secure-login.com/verify?redirect=http://paypal.com",
        },
        {
            "message_text": "Hey, are we still on for lunch tomorrow at 1pm?",
            "url": None,
        },
        {
            "message_text": "Hi, following up on the invoice I sent last week, let me know if you have questions.",
            "url": "https://www.google.com",
        },
    ]
    for ex in examples:
        result = predict_combined(**ex)
        print(ex["message_text"][:60])
        print(f"  url: {ex['url']}")
        print(f"  text_proba={result['text_proba']:.3f}  url_proba={result['url_proba']}  "
              f"combined={result['combined_proba']:.3f}  -> {result['prediction']}")
        print()
