"""
My second model: Random Forest on features pulled out of the URL itself (see src/url_features.py).
This one doesn't look at the email text at all, just whatever link is inside it.

A Random Forest is basically a big pile of decision trees that each get a vote, and you go with
whatever most of them say. It suits this better than the TF-IDF/logistic regression setup from
text_baseline.py, because here I've got a small table of hand picked numbers (url length, has an
IP address, etc.) rather than a huge pile of raw text.

Run it with:
    python src/url_baseline.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from url_data_loading import load_url_dataset
from url_features import urls_to_feature_frame

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42


def build_model() -> RandomForestClassifier:
    """Sets up the Random Forest. Using mostly default settings for now, same as text_baseline.py."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_and_evaluate():
    df = load_url_dataset()
    features = urls_to_feature_frame(df["url"])

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["legitimate", "phishing"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    print(f"ROC-AUC: {roc_auc:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "url_random_forest",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": roc_auc,
        "report": report,
    }
    with open(RESULTS_DIR / "url_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix plot (shows what it got right/wrong)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("URL model: confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "url_baseline_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve plot (another way of showing how good the model is)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="Random Forest")
    ax.set_title(f"URL model: ROC curve (AUC = {roc_auc:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "url_baseline_roc_curve.png", dpi=150)
    plt.close(fig)

    # feature importance plot: which features the forest actually leaned on most
    importances = pd.Series(model.feature_importances_, index=features.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.plot.barh(ax=ax)
    ax.set_title("URL model: which features mattered most")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "url_baseline_feature_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump(model, MODELS_DIR / "url_baseline.joblib")
    print(f"\nSaved metrics + plots to {RESULTS_DIR}, model to {MODELS_DIR}")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
