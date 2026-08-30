"""
Downloads and combines the email dataset I'm using for this project.

It's a mirror of the Kaggle "Phishing Email Dataset" (by naserabdullahalam), hosted on GitHub at
https://github.com/rokibulroni/Phishing-Email-Dataset. That page is really 6 smaller email
datasets bundled together, each with its own `label` column (1 = phishing/spam, 0 = normal email).

How to use it:
    python src/data_loading.py             # downloads everything, builds data/processed/combined.csv
    from src.data_loading import load_combined_dataset
    df = load_combined_dataset()           # just gives you back the dataframe (downloads it first if needed)
"""

import subprocess
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/rokibulroni/Phishing-Email-Dataset.git"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "Phishing-Email-Dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
COMBINED_PATH = PROCESSED_DIR / "combined.csv"

# these are the 6 files that make up the combined dataset, see data/README.md for what's in each one
SOURCE_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]


def fetch_raw_data(force: bool = False) -> Path:
    """Downloads the dataset into data/raw/ if it's not already there."""
    if RAW_DIR.exists() and not force:
        return RAW_DIR

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists() and force:
        subprocess.run(["rm", "-rf", str(RAW_DIR)], check=True)

    print(f"Cloning {REPO_URL} into {RAW_DIR} (first run only, ~250MB)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(RAW_DIR)],
        check=True,
    )
    return RAW_DIR


def _load_one(path: Path) -> pd.DataFrame:
    """Loads one of the 6 CSVs and mashes subject + body together into one `text` column."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    subject = df["subject"].fillna("") if "subject" in df.columns else ""
    body = df["body"].fillna("") if "body" in df.columns else ""
    text = (subject + " " + body).astype(str).str.strip() if "subject" in df.columns else body.astype(str)

    out = pd.DataFrame({"text": text, "label": df["label"]})
    out["source"] = path.stem
    return out


def build_combined_dataset(force_refetch: bool = False) -> pd.DataFrame:
    """Downloads the data if needed, glues all 6 files together, cleans it up, saves it."""
    raw_dir = fetch_raw_data(force=force_refetch)

    frames = [_load_one(raw_dir / fname) for fname in SOURCE_FILES]
    combined = pd.concat(frames, ignore_index=True)

    # cleaning up: get rid of blank messages, duplicate messages, and messy whitespace
    combined["text"] = combined["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset="text")
    combined["label"] = combined["label"].astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"Combined dataset: {len(combined):,} rows -> {COMBINED_PATH}")
    print(combined["label"].value_counts(normalize=True).rename("share").to_string())
    return combined


def load_combined_dataset(force_refetch: bool = False) -> pd.DataFrame:
    """Gives you the combined dataset. Builds it first if this is the first time running it."""
    if COMBINED_PATH.exists() and not force_refetch:
        return pd.read_csv(COMBINED_PATH)
    return build_combined_dataset(force_refetch=force_refetch)


if __name__ == "__main__":
    build_combined_dataset()
