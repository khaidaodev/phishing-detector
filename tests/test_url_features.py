"""
A couple of quick tests for src/url_features.py. Doesn't need internet, just checks the feature
extraction picks up on the obvious stuff correctly using a few made up examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from url_features import extract_url_features  # noqa: E402


def test_flags_ip_address_url():
    feats = extract_url_features("http://192.168.1.1/update-account.php")
    assert feats["is_ip_address"] == 1
    assert feats["is_https"] == 0


def test_flags_https_normal_domain():
    feats = extract_url_features("https://www.paypal.com/signin")
    assert feats["is_ip_address"] == 0
    assert feats["is_https"] == 1
    assert feats["has_at_symbol"] == 0


def test_flags_at_symbol_trick():
    feats = extract_url_features("http://real-bank.com@fake-site.ru/login")
    assert feats["has_at_symbol"] == 1


def test_flags_known_shortener():
    feats = extract_url_features("https://bit.ly/3abcXYZ")
    assert feats["is_known_shortener"] == 1


def test_counts_subdomains():
    feats = extract_url_features("https://secure.login.paypal-verify.com")
    assert feats["num_subdomains"] == 2


def test_domain_with_digits_flagged():
    feats = extract_url_features("http://paypa1.com/verify")
    assert feats["domain_has_digits"] == 1
