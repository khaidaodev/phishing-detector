"""
Downloads the URL dataset I'm using to train the link-checking model: PhiUSIIL, about 236,000 real
URLs (roughly half legit, half phishing), from the UCI Machine Learning Repository.

PhiUSIIL actually comes with about 50 features already worked out for you (whether the page has a
favicon, how many images are on it, that sort of thing). I'm ignoring basically all of that on
purpose, those features need someone to have actually crawled the live webpage, which isn't
something you can do for a link sitting in someone's inbox (the page might be down, blocked, or
you might not even have internet access at the point you're checking it). So I only keep the `URL`
column and the label, and build my own features from just the URL text in `src/url_features.py`.

How to use it:
    python src/url_data_loading.py           # downloads and builds data/processed/url_dataset.csv
    from src.url_data_loading import load_url_dataset
    df = load_url_dataset()                  # gives you back url + label columns
"""

import shutil
import ssl
import zipfile
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd

DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "phiusiil"
PROCESSED_DIR = ROOT / "data" / "processed"
COMBINED_PATH = PROCESSED_DIR / "url_dataset.csv"


def _find_raw_csv():
    if not RAW_DIR.exists():
        return None
    matches = list(RAW_DIR.glob("*.csv"))
    return matches[0] if matches else None


def fetch_raw_data(force: bool = False) -> Path:
    """Downloads + unzips the dataset into data/raw/ if it's not already there."""
    existing = _find_raw_csv()
    if existing and not force:
        return existing

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / "phiusiil.zip"
    print(f"Downloading PhiUSIIL dataset (~15MB zipped) from {DATASET_ZIP_URL}...")

    # Macs installed via python.org don't always ship a working list of trusted certificates for
    # Python to check https sites against, so this points it at the one from the `certifi` package
    # instead of relying on the system to have one set up already.
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(DATASET_ZIP_URL, context=context) as response, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    zip_path.unlink()

    csv_path = _find_raw_csv()
    if csv_path is None:
        raise FileNotFoundError(f"Couldn't find a CSV after unzipping into {RAW_DIR}")
    return csv_path


def build_url_dataset(force_refetch: bool = False) -> pd.DataFrame:
    """Downloads the raw data if needed, keeps just the URL + label, saves the cleaned version."""
    csv_path = fetch_raw_data(force=force_refetch)
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip() for c in raw.columns]

    url_col = "URL" if "URL" in raw.columns else "url"
    label_col = "label" if "label" in raw.columns else "Label"

    # PhiUSIIL uses label 1 = legitimate, 0 = phishing, the opposite way round to my email dataset
    # (data_loading.py uses 1 = phishing), so flip it here to keep both datasets consistent.
    df = pd.DataFrame({
        "url": raw[url_col].astype(str),
        "label": 1 - raw[label_col].astype(int),
    })
    df = df.drop_duplicates(subset="url")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMBINED_PATH, index=False)
    print(f"URL dataset: {len(df):,} rows -> {COMBINED_PATH}")
    print(df["label"].value_counts(normalize=True).rename("share").to_string())
    return df


def load_url_dataset(force_refetch: bool = False) -> pd.DataFrame:
    """Gives you the URL dataset. Builds it first if this is the first time running it."""
    if COMBINED_PATH.exists() and not force_refetch:
        return pd.read_csv(COMBINED_PATH)
    return build_url_dataset(force_refetch=force_refetch)


if __name__ == "__main__":
    build_url_dataset()
