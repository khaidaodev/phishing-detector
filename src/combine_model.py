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


def combine_scores(text_proba: float, url_proba: float | None) -> float:
    """Combines the two models' phishing probabilities into one score for the message.

    If there's no URL in the message at all, the URL model never got a vote, so this is just
    the text model's score. If there is a URL, this takes whichever of the two scores is
    higher rather than averaging them: an obviously alarming link shouldn't get diluted by
    boring-sounding text wrapped around it, and the other way round, a boring-looking link in
    a screaming "your account is suspended!!" email shouldn't make the message look safer.
    A weighted average is probably worth trying later once there's a proper way to evaluate
    which combining rule actually works best (see the limitation noted in the README), this
    is the simple version to start with.
    """
    if url_proba is None:
        return text_proba
    return max(text_proba, url_proba)


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
