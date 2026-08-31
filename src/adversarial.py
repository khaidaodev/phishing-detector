"""
Trying to break my own models on purpose. A lot of student projects just train something, get a
good accuracy number, and stop there, I don't want to do that.

This takes a small set of messages I've written by hand (so I know the right answer for each
one), applies a few tricks real phishers actually use to dodge filters, and checks how much
worse `combine_model.py`'s predictions get.

Doing three tricks in this pass, all "character level" stuff that's easy to generate a bunch of
automatically:
    - typos (inject_typos): swapping or dropping a letter here and there
    - extra/broken up spacing (add_extra_spacing): "v e r i f y" instead of "verify", an old spam
      filter dodge that relies on matching exact keywords
    - homoglyphs (homoglyph_substitute): swapping a letter in a URL's domain for one that looks
      almost the same, like a zero instead of a capital O, e.g. paypal.com -> payp4l.com

A fourth one is still to come in a follow up pass: reworded urgent/scary language. That one needs
actually hand-rewriting each phishing example rather than a generic function, so it's its own
piece of work rather than something to rush in here.

This also doubles as the "proper evaluation" for the combined model that src/combine_model.py's
notes said was still missing, since it's a set of full messages with intact URLs and known right
answers, exactly what was needed.

Run it with:
    python src/adversarial.py
"""

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import tldextract

from combine_model import predict_combined

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# A small handwritten test set with known right answers. label: 1 = phishing, 0 = legitimate,
# same convention as everywhere else in this project.
BASELINE_EXAMPLES = [
    {
        "message_text": "Your account has been suspended, click here to verify your identity immediately.",
        "url": "http://paypa1-secure-login.com/verify?redirect=http://paypal.com",
        "label": 1,
    },
    {
        "message_text": "URGENT: your bank account will be locked in 24 hours unless you confirm your details now.",
        "url": "http://192.168.45.12/secure-banking/confirm.php",
        "label": 1,
    },
    {
        "message_text": "Congratulations! You've won a $500 gift card, claim it before it expires today.",
        "url": "http://bit.ly/claim-prize-now",
        "label": 1,
    },
    {
        "message_text": "We noticed unusual sign-in activity on your account. Verify now to avoid permanent suspension.",
        "url": "http://amaz0n-account-security.com/signin",
        "label": 1,
    },
    {
        "message_text": "Your package could not be delivered. Update your shipping address to reschedule delivery.",
        "url": "http://usps-redelivery-update.info/track",
        "label": 1,
    },
    {
        "message_text": "Final notice: your subscription payment failed. Update your payment details within 48 hours or it will be cancelled.",
        "url": None,
        "label": 1,
    },
    {"message_text": "Hey, are we still on for lunch tomorrow at 1pm?", "url": None, "label": 0},
    {
        "message_text": "Hi, following up on the invoice I sent last week, let me know if you have questions.",
        "url": "https://www.google.com",
        "label": 0,
    },
    {"message_text": "Thanks for your help with the report yesterday, really appreciate it.", "url": None, "label": 0},
    {"message_text": "Reminder: team standup moved to 10am tomorrow, see you there.", "url": None, "label": 0},
    {
        "message_text": "Here's the recording from today's meeting, let me know if you have questions.",
        "url": "https://drive.google.com/file/d/abc123/view",
        "label": 0,
    },
    {"message_text": "Happy birthday! Hope you have a great day, let's catch up soon.", "url": None, "label": 0},
]

# only a handful of letters actually have a convincing lookalike digit, no point mapping the rest
HOMOGLYPHS = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}


def inject_typos(text: str, rate: float = 0.15, seed: int = 0) -> str:
    """Randomly messes up a handful of words, swapping two adjacent letters or dropping one.
    Happens by accident all the time, but it's also a real trick people use on purpose to dodge
    keyword-based filters."""
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for word in words:
        if len(word) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(word) - 1)
            if rng.choice(["swap", "drop"]) == "swap":
                chars = list(word)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                word = "".join(chars)
            else:
                word = word[:i] + word[i + 1 :]
        out.append(word)
    return " ".join(out)


def add_extra_spacing(text: str, rate: float = 0.2, seed: int = 0) -> str:
    """Breaks a few longer words up into single letters with spaces between them, e.g. "verify"
    becomes "v e r i f y". An old spam filter dodge, since it stops anything matching on the
    exact word "verify" from catching it."""
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for word in words:
        if len(word) > 4 and rng.random() < rate:
            word = " ".join(list(word))
        out.append(word)
    return " ".join(out)


def homoglyph_substitute(url, num_swaps: int = 2, seed: int = 0):
    """Swaps a couple of letters in the URL's domain for lookalike characters, e.g.
    paypal.com -> payp4l.com. Only touches the domain, not the whole URL, since that's how real
    typosquatting actually works, one dodgy-looking domain, not the whole link turned to noise."""
    if not url:
        return url
    rng = random.Random(seed)
    ext = tldextract.extract(url)
    domain = ext.domain
    swappable_positions = [i for i, c in enumerate(domain) if c in HOMOGLYPHS]
    rng.shuffle(swappable_positions)
    chars = list(domain)
    for i in swappable_positions[:num_swaps]:
        chars[i] = HOMOGLYPHS[chars[i]]
    new_domain = "".join(chars)
    if new_domain == domain:
        return url
    return url.replace(domain, new_domain, 1)


# each one takes an example dict + a seed, and returns a new example dict with one field
# tweaked. Returns the *same* object (checked with `is` below) when the trick genuinely doesn't
# apply, e.g. no URL to homoglyph-swap, so that case doesn't get counted as a pass or a fail.
PERTURBATIONS = {
    "typos": lambda ex, seed: {**ex, "message_text": inject_typos(ex["message_text"], seed=seed)},
    "extra_spacing": lambda ex, seed: {**ex, "message_text": add_extra_spacing(ex["message_text"], seed=seed)},
    "homoglyph_url": lambda ex, seed: (
        {**ex, "url": homoglyph_substitute(ex["url"], seed=seed)} if ex.get("url") else ex
    ),
}


def _predict_is_correct(example: dict, text_model, url_model) -> bool:
    result = predict_combined(
        example["message_text"], example.get("url"), text_model=text_model, url_model=url_model
    )
    expected = "phishing" if example["label"] == 1 else "legitimate"
    return result["prediction"] == expected


def run_adversarial_eval():
    text_model = joblib.load(MODELS_DIR / "text_baseline.joblib")
    url_model = joblib.load(MODELS_DIR / "url_baseline.joblib")

    report = {}

    baseline_correct = sum(_predict_is_correct(ex, text_model, url_model) for ex in BASELINE_EXAMPLES)
    baseline_accuracy = baseline_correct / len(BASELINE_EXAMPLES)
    report["baseline"] = {"accuracy": baseline_accuracy, "n": len(BASELINE_EXAMPLES)}
    print(f"baseline (no tricks): {baseline_correct}/{len(BASELINE_EXAMPLES)} correct ({baseline_accuracy:.0%})")

    for name, perturb in PERTURBATIONS.items():
        correct = 0
        applicable = 0
        newly_wrong = []
        for i, ex in enumerate(BASELINE_EXAMPLES):
            perturbed = perturb(ex, i)
            if perturbed is ex:
                continue  # trick didn't apply to this example (e.g. no URL), skip it
            applicable += 1
            if _predict_is_correct(perturbed, text_model, url_model):
                correct += 1
            else:
                newly_wrong.append(ex["message_text"][:50])

        accuracy = correct / applicable if applicable else None
        report[name] = {"accuracy": accuracy, "n": applicable, "now_wrong": newly_wrong}
        if applicable:
            print(f"{name}: {correct}/{applicable} correct ({accuracy:.0%}), now getting these wrong: {newly_wrong}")
        else:
            print(f"{name}: didn't apply to any of the examples")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "adversarial_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    labels = ["baseline"] + list(PERTURBATIONS.keys())
    values = [report[k]["accuracy"] or 0 for k in labels]
    colors = ["#4c72b0"] + ["#c44e52"] * len(PERTURBATIONS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy on the 12 handwritten examples")
    ax.set_title("How much each trick breaks the combined model")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "adversarial_accuracy.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved metrics + plot to {RESULTS_DIR}")
    return report


if __name__ == "__main__":
    run_adversarial_eval()
