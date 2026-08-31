"""
Tests for the perturbation functions in src/adversarial.py, the bits that don't need the real
trained models on disk. The full evaluation (run_adversarial_eval) does need them, so that's
checked by hand by actually running the file, not covered here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adversarial import add_extra_spacing, homoglyph_substitute, inject_typos  # noqa: E402


def test_inject_typos_changes_text_with_high_rate():
    original = "verify your account details immediately please"
    perturbed = inject_typos(original, rate=1.0, seed=1)
    assert perturbed != original
    # still roughly the same length, just some letters swapped or dropped, not a different message
    assert abs(len(perturbed) - len(original)) <= 6


def test_inject_typos_is_reproducible_with_the_same_seed():
    original = "verify your account details immediately please"
    assert inject_typos(original, rate=1.0, seed=5) == inject_typos(original, rate=1.0, seed=5)


def test_inject_typos_leaves_short_words_alone():
    # nothing here is longer than 3 characters, so there's nothing safe to touch
    assert inject_typos("go to it ok", rate=1.0, seed=1) == "go to it ok"


def test_add_extra_spacing_breaks_up_long_words():
    perturbed = add_extra_spacing("please verify your account now", rate=1.0, seed=1)
    assert "v e r i f y" in perturbed
    assert perturbed != "please verify your account now"


def test_homoglyph_substitute_only_touches_the_domain():
    perturbed = homoglyph_substitute("http://paypal.com/login?x=1", num_swaps=2, seed=1)
    assert perturbed != "http://paypal.com/login?x=1"
    assert perturbed.startswith("http://")
    assert perturbed.endswith("/login?x=1")


def test_homoglyph_substitute_handles_no_url():
    assert homoglyph_substitute(None) is None
    assert homoglyph_substitute("") == ""
