"""
Turns a URL into a bunch of numbers a model can actually learn from (a "feature vector"). This is
the link-checking half of the project, it doesn't look at the email text at all, just whatever
link is inside it.

Everything here only looks at the URL string itself, not what's actually on the page it points to.
That's on purpose: a link in an email might point to a page that's already been taken down by the
time anyone checks it, and a real detector might not even have internet access to go fetch the
page live. So this only uses stuff you can tell just from reading the URL, the same way a
suspicious person squinting at a link would.

Run it with:
    python src/url_features.py     # quick sanity check on a couple of example URLs
"""

import math
import re
from urllib.parse import urlparse

import pandas as pd
import tldextract

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# words that turn up a lot in phishing links (login pages, "your account is suspended" style urgency)
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "webscr", "ebayisapi", "suspend", "urgent",
]

# a few well known link shorteners, these hide the real destination until you click
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
}


def _shannon_entropy(s: str) -> float:
    """How "random" a string looks. Real words score low, random-looking strings (the kind a script
    spits out to generate a new throwaway phishing domain every day) score higher."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_url_features(url: str) -> dict:
    """Pulls every feature out of a single URL. Returns a plain dict, one number/flag per feature."""
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    ext = tldextract.extract(url)

    hostname = parsed.netloc.split("@")[-1].split(":")[0]  # strip any user@ bit and any :port
    domain = ext.domain  # e.g. "paypal"
    # tldextract 5.1 calls this .registered_domain, newer versions renamed it (same thing)
    registered_domain = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
    subdomain = ext.subdomain  # e.g. "www", or "secure.login" for something dodgier

    path_and_query = (parsed.path or "") + (("?" + parsed.query) if parsed.query else "")
    after_scheme = url.split("://", 1)[-1]
    digits = sum(c.isdigit() for c in url)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path_and_query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / len(url) if url else 0.0,
        "num_subdomains": len([p for p in subdomain.split(".") if p]) if subdomain else 0,
        "is_ip_address": int(bool(IP_PATTERN.match(hostname))),
        "is_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        # a genuine "//" showing up again after the protocol is a classic open-redirect trick
        "has_redirect_slash": int("//" in after_scheme),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9./\-]", url)),
        "num_suspicious_words": sum(word in url.lower() for word in SUSPICIOUS_WORDS),
        # real brands basically never put digits in their own domain name (paypa1.com, amaz0n.com...)
        "domain_has_digits": int(any(c.isdigit() for c in domain)),
        "domain_entropy": _shannon_entropy(domain),
        "is_known_shortener": int(registered_domain in URL_SHORTENERS),
    }


def urls_to_feature_frame(urls: pd.Series) -> pd.DataFrame:
    """Turns a whole column of URLs into a dataframe of features, one row per URL."""
    return pd.DataFrame(urls.apply(extract_url_features).tolist(), index=urls.index)


if __name__ == "__main__":
    examples = [
        "https://www.paypal.com/signin",
        "http://192.168.1.1/update-account.php",
        "http://paypa1-secure-login.com/verify?redirect=http://paypal.com",
    ]
    for u in examples:
        print(u)
        for k, v in extract_url_features(u).items():
            print(f"  {k}: {v}")
        print()
