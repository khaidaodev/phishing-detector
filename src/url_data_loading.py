"""
Downloads the URL dataset I'm using to train the link-checking model: PhiUSIIL, about 236,000 real
URLs (roughly half legit, half phishing), from the UCI Machine Learning Repository.

PhiUSIIL actually comes with about 50 features already worked out for you (whether the page has a
favicon, how many images are on it, that sort of thing). I'm ignoring basically all of that on
purpose, those features need someone to have actually crawled the live webpage, which isn't
something you can do for a link sitting in someone's inbox (the page might be down, blocked, or
you might not even have internet access at the point you're checking it). So I only keep the `URL`
column and the label, and build my own features from just the URL text in `src/url_features.py`.

Found a real problem with this dataset while digging into a wrong prediction from stage 4
(see the README): every single "legitimate" URL in PhiUSIIL is a bare homepage, literally
"https://www.something.tld" with nothing after it, zero exceptions across all ~135,000 of them.
"Phishing" ones do have real paths a lot of the time. That's not a real-world pattern, that's
just how this dataset happened to get collected, but the model latched onto it: "has any path at
all" basically meant "phishing" to it, so genuinely legit links like paypal.com/signin or a
Google Doc link were getting flagged. `_augment_legitimate_urls_with_paths` below fixes the worst
of that by tacking a realistic path onto half the legitimate URLs before training, so the model
can't use "any path = phishing" as a free shortcut any more.

First version of that fix used a fixed list of 18 short, clean template paths. Turned out that
was too narrow, on a proper 100-URL test the model still flagged 84% of real legitimate URLs
(anything with a longer or hyphen-heavy path, basically all of Wikipedia, GitHub, Stack Overflow,
news, e-commerce), because it had just learned "short clean path = fine" rather than "paths are
normal". `_generate_realistic_path` below is the fix for that, it builds a fresh, randomised path
for every row instead of repeating 18 fixed strings. Full details and numbers for both attempts
are in the README's stage 5 and stage 6 sections.

How to use it:
    python src/url_data_loading.py           # downloads and builds data/processed/url_dataset.csv
    from src.url_data_loading import load_url_dataset
    df = load_url_dataset()                  # gives you back url + label columns
"""

import random
import shutil
import ssl
import zipfile
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd

# A pool of words to build paths out of, joined up differently every call (see
# _generate_realistic_path below) rather than a fixed list of templates, so the augmented paths
# actually spread across the range of shapes real URLs come in.
PATH_WORDS = [
    "login", "account", "profile", "settings", "dashboard", "report", "invoice", "order",
    "product", "article", "post", "review", "comment", "photo", "video", "document", "project",
    "team", "event", "ticket", "booking", "payment", "subscription", "download", "upload",
    "search", "results", "category", "collection", "guide", "tutorial", "faq", "support",
    "contact", "about", "terms", "privacy", "blog", "news", "press", "careers", "pricing",
    "plans", "features", "docs", "reference", "changelog", "release", "update", "status",
]


def _random_path_segment(rng: random.Random) -> str:
    """One piece of a path, in one of three shapes real URLs actually use: a single word
    ("login"), a hyphenated multi-word slug ("breaking-news-today"), or an alphanumeric id
    ("a1b2c3d4e5f6", the kind of thing a database row or a file-sharing link generates)."""
    kind = rng.random()
    if kind < 0.4:
        return rng.choice(PATH_WORDS)
    if kind < 0.7:
        words = [rng.choice(PATH_WORDS) for _ in range(rng.randint(2, 4))]
        return "-".join(words)
    length = rng.randint(4, 24)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(length))


def _generate_realistic_path(rng: random.Random) -> str:
    """Builds one full path, 1-4 segments deep, sometimes with a file extension or a query
    string on the end. Randomised per call instead of picked from a fixed list, so across
    thousands of rows the augmented paths end up covering roughly the same spread of length,
    hyphen count and digit content that real URLs do."""
    segments = [_random_path_segment(rng) for _ in range(rng.randint(1, 4))]
    path = "/" + "/".join(segments)
    if rng.random() < 0.15:
        path += rng.choice([".html", ".pdf", ".json", ".php"])
    if rng.random() < 0.2:
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 6)))
        path += "?" + rng.choice(["id=", "ref=", "page=", "q="]) + digits
    return path

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


def _augment_legitimate_urls_with_paths(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Gives half of the legitimate URLs a realistic path, so "does this URL have a path at all"
    stops being a giveaway for "must be phishing" (see the big docstring comment up top for why
    this is needed). Only touches label=0 (legit) rows, leaves phishing rows exactly as they are.
    Seeded, so re-running this on the same data gives the same result every time.

    Version 2: every row that gets a path gets its own freshly generated one
    (_generate_realistic_path) instead of picking from a small fixed list. Version 1 used 18
    fixed templates, all short and hyphen-light, and the model just learned that narrow shape
    instead of "paths are normal" (see the README's stage 6 section for the numbers that showed
    this)."""
    rng = random.Random(seed)
    df = df.copy()

    def maybe_add_path(url: str) -> str:
        if rng.random() < 0.5:
            return url.rstrip("/") + _generate_realistic_path(rng)
        return url

    legit_mask = df["label"] == 0
    df.loc[legit_mask, "url"] = df.loc[legit_mask, "url"].apply(maybe_add_path)
    return df


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
    df = _augment_legitimate_urls_with_paths(df)
    # augmenting can occasionally create a duplicate (two different homepages both getting the
    # same path tacked on), drop those too
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
