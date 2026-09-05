"""
Tests for the augmentation step in src/url_data_loading.py, the part that doesn't need internet
or the real PhiUSIIL dataset on disk.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from url_data_loading import (  # noqa: E402
    _add_real_legitimate_examples,
    _augment_legitimate_urls_with_paths,
)


def _sample_df():
    return pd.DataFrame({
        "url": [
            "https://www.legit-one.com",
            "https://www.legit-two.com",
            "https://www.legit-three.com",
            "https://www.legit-four.com",
            "http://phishy-one.ru",
            "http://phishy-two.tk",
        ],
        "label": [0, 0, 0, 0, 1, 1],
    })


def test_only_touches_legitimate_rows():
    out = _augment_legitimate_urls_with_paths(_sample_df())
    # the two phishing urls come back completely untouched
    assert out.loc[out["label"] == 1, "url"].tolist() == ["http://phishy-one.ru", "http://phishy-two.tk"]


def test_gives_some_but_not_all_legit_urls_a_path():
    out = _augment_legitimate_urls_with_paths(_sample_df())
    legit_urls = out.loc[out["label"] == 0, "url"].tolist()
    has_path = [u for u in legit_urls if u not in _sample_df()["url"].tolist()]
    # with 4 legit rows and a 50/50 chance each, expect a mix rather than all-or-nothing,
    # this isn't a hard guarantee but would need very unlucky luck to fail with a fixed seed
    assert 0 < len(has_path) < len(legit_urls)


def test_is_reproducible_with_the_same_seed():
    out1 = _augment_legitimate_urls_with_paths(_sample_df(), seed=7)
    out2 = _augment_legitimate_urls_with_paths(_sample_df(), seed=7)
    assert out1["url"].tolist() == out2["url"].tolist()


def test_does_not_mutate_the_original_dataframe():
    original = _sample_df()
    original_urls = original["url"].tolist()
    _augment_legitimate_urls_with_paths(original)
    assert original["url"].tolist() == original_urls


def test_add_real_legitimate_examples_appends_them_as_label_zero():
    out = _add_real_legitimate_examples(_sample_df(), ["https://real-site.com/some/real/path"])
    new_row = out[out["url"] == "https://real-site.com/some/real/path"]
    assert len(new_row) == 1
    assert new_row["label"].iloc[0] == 0


def test_add_real_legitimate_examples_keeps_the_original_rows_too():
    out = _add_real_legitimate_examples(_sample_df(), ["https://real-site.com/path"])
    assert len(out) == len(_sample_df()) + 1
    assert "http://phishy-one.ru" in out["url"].tolist()
