"""
Tests for src/real_legit_urls.py. The main thing worth checking here isn't the URLs themselves
(they're just data), it's that this training set stays honestly separate from the 100-URL
evaluation set in src/real_url_eval.py, training on the exact URLs used to measure the model
would make the false-positive numbers look better than they actually are.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from real_legit_urls import ALL_REAL_LEGIT_TRAINING_URLS, REAL_LEGIT_TRAINING_URLS  # noqa: E402
from real_url_eval import REAL_LEGITIMATE_URLS  # noqa: E402


def test_flattened_list_matches_the_per_category_dict():
    expected = [url for urls in REAL_LEGIT_TRAINING_URLS.values() for url in urls]
    assert ALL_REAL_LEGIT_TRAINING_URLS == expected


def test_no_duplicate_urls_within_the_training_set():
    assert len(ALL_REAL_LEGIT_TRAINING_URLS) == len(set(ALL_REAL_LEGIT_TRAINING_URLS))


def test_does_not_overlap_with_the_real_url_eval_test_set():
    # this is the important one: if any of these also showed up in real_url_eval.py, training on
    # them would make stage 6/7/8's false-positive measurements meaningless, the model would just
    # be remembering the exact test URLs rather than actually generalizing
    eval_urls = {url for urls in REAL_LEGITIMATE_URLS.values() for url in urls}
    overlap = set(ALL_REAL_LEGIT_TRAINING_URLS) & eval_urls
    assert overlap == set()


def test_weighted_towards_the_categories_stage_6_and_7_found_still_broken():
    # github and stack overflow were the two categories stage 7 couldn't fix at all (97%/100%
    # false positives even with the confidence-weighted combiner), so this training set should
    # have noticeably more examples of those than the easier categories
    assert len(REAL_LEGIT_TRAINING_URLS["github"]) > len(REAL_LEGIT_TRAINING_URLS["wikipedia"])
    assert len(REAL_LEGIT_TRAINING_URLS["stackoverflow"]) > len(REAL_LEGIT_TRAINING_URLS["wikipedia"])


def test_every_url_looks_like_a_url():
    for url in ALL_REAL_LEGIT_TRAINING_URLS:
        assert url.startswith("https://") or url.startswith("http://")
