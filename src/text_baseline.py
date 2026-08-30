"""
My first model: TF-IDF + Logistic Regression, just looking at the email text.

TF-IDF stands for Term Frequency - Inverse Document Frequency, which sounds complicated but it's
basically just a way of scoring how unusual/distinctive a word is to a message, instead of just
counting how often it shows up. A word like "the" scores low because it's in everything, but
something like "wire transfer" scores high because it's rare and stands out.

Run it with:
    python src/text_baseline.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loading import load_combined_dataset

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Sets up the TF-IDF step + the logistic regression model. Using mostly default settings for now."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate():
    df = load_combined_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["legitimate", "phishing"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    print(f"ROC-AUC: {roc_auc:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "tfidf_logreg_baseline",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": roc_auc,
        "report": report,
    }
    with open(RESULTS_DIR / "text_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix plot (shows what it got right/wrong)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Text baseline: confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "text_baseline_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve plot (another way of showing how good the model is)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="TF-IDF + LogReg")
    ax.set_title(f"Text baseline: ROC curve (AUC = {roc_auc:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "text_baseline_roc_curve.png", dpi=150)
    plt.close(fig)

    joblib.dump(pipeline, MODELS_DIR / "text_baseline.joblib")
    print(f"\nSaved metrics + plots to {RESULTS_DIR}, model to {MODELS_DIR}")

    return pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
