"""
Stage 6 measured the URL model's own false-positive rate on 100 real legitimate URLs (46%
overall, see README), but that's the URL model scoring a URL on its own, not what someone
pasting a real message actually gets back from predict_combined() in src/combine_model.py.

This pairs those same 100 URLs with a few different boring, obviously-legitimate messages and
scores the *combined* result instead, comparing the old "take the higher score" combining rule
against the confidence-weighted one it got replaced with, to check the new rule is an actual,
measured improvement and not just a change that sounds reasonable.

Run it with:
    python src/combined_url_eval.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combine_model import combine_scores
from real_url_eval import REAL_LEGITIMATE_URLS
from url_features import urls_to_feature_frame

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# a few different, deliberately unremarkable messages, so this isn't just measuring how the
# text model happens to score one specific sentence
BORING_MESSAGES = [
    "Hi, just sharing this link, let me know what you think.",
    "Here's the one I mentioned earlier, take a look when you get a chance.",
    "Thought this might be useful, thanks!",
]


def old_combine_scores(text_proba: float, url_proba: float) -> float:
    """The combining rule stage 3 originally shipped with: whichever score was higher, no
    weighting. Kept here only so this script can measure the improvement against it,
    combine_model.py itself doesn't use this anymore, see the note in combine_scores() there."""
    return max(text_proba, url_proba)


def evaluate_combined(text_model, url_model, urls_by_category: dict, messages: list) -> dict:
    """Pairs every message with every URL in every category, scores each pair with both
    combining rules, and reports the false-positive rate for each. Every URL here is genuinely
    legitimate (see real_url_eval.py), so any "phishing" prediction, under either rule, is a
    false positive."""
    text_probas = [float(text_model.predict_proba([m])[0][1]) for m in messages]

    results = {}
    old_total_fp = new_total_fp = total = 0
    for category, urls in urls_by_category.items():
        features = urls_to_feature_frame(pd.Series(urls))
        url_probas = url_model.predict_proba(features)[:, 1]

        old_fp = new_fp = 0
        for url_proba in url_probas:
            for text_proba in text_probas:
                if old_combine_scores(text_proba, float(url_proba)) >= 0.5:
                    old_fp += 1
                if combine_scores(text_proba, float(url_proba)) >= 0.5:
                    new_fp += 1
        pair_count = len(urls) * len(messages)

        results[category] = {
            "count": pair_count,
            "old_false_positives": old_fp,
            "old_false_positive_rate": old_fp / pair_count,
            "new_false_positives": new_fp,
            "new_false_positive_rate": new_fp / pair_count,
        }
        old_total_fp += old_fp
        new_total_fp += new_fp
        total += pair_count

    results["_overall"] = {
        "count": total,
        "old_false_positives": old_total_fp,
        "old_false_positive_rate": old_total_fp / total if total else 0.0,
        "new_false_positives": new_total_fp,
        "new_false_positive_rate": new_total_fp / total if total else 0.0,
    }
    return results


def plot_results(results: dict, out_path: Path) -> None:
    categories = [c for c in results if not c.startswith("_")]
    old_rates = [results[c]["old_false_positive_rate"] * 100 for c in categories]
    new_rates = [results[c]["new_false_positive_rate"] * 100 for c in categories]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, old_rates, width, label="old rule (take the higher score)", color="#c0392b")
    ax.bar(x + width / 2, new_rates, width, label="new rule (confidence-weighted average)", color="#4c72b0")
    ax.set_ylabel("False positive rate (%)")
    ax.set_title("Combined-model false positives on real legitimate URLs + boring text, old vs new rule")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    text_model = joblib.load(MODELS_DIR / "text_baseline.joblib")
    url_model = joblib.load(MODELS_DIR / "url_baseline.joblib")

    results = evaluate_combined(text_model, url_model, REAL_LEGITIMATE_URLS, BORING_MESSAGES)

    for category, stats in results.items():
        if category == "_overall":
            continue
        print(
            f"{category:15s} old {stats['old_false_positives']:3d}/{stats['count']} "
            f"({stats['old_false_positive_rate']:.0%})  ->  new {stats['new_false_positives']:3d}/"
            f"{stats['count']} ({stats['new_false_positive_rate']:.0%})"
        )
    overall = results["_overall"]
    print(
        f"\noverall: old {overall['old_false_positives']}/{overall['count']} "
        f"({overall['old_false_positive_rate']:.1%})  ->  new {overall['new_false_positives']}/"
        f"{overall['count']} ({overall['new_false_positive_rate']:.1%})"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "combined_url_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, RESULTS_DIR / "combined_url_eval.png")
    print(f"\nsaved results to {RESULTS_DIR}")
