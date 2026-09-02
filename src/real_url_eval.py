"""
Stage 5 fixed the "any path at all = phishing" shortcut for normal URLs, but the check for it was
thin, 7 URLs I picked by hand and typed straight into a table. This does the same idea properly:
a set of 100 real, legitimate URLs across 10 categories (10 each), including the file-sharing
links that stage 5 found were still broken, and measures the actual false-positive rate per
category instead of eyeballing a handful of examples.

These are hand-assembled from the real shape each site's URLs take, not a live crawl, some of the
specific pages (news articles especially) may have moved or been taken down by the time you read
this. That's fine here: the URL model never fetches the page, it only ever looks at the URL string
itself (src/url_features.py), so a dead link is still a perfectly valid, realistic test case for
"what does a legitimate URL from this site look like".

Run it with:
    python src/real_url_eval.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from url_features import urls_to_feature_frame

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# All of these are genuinely legitimate, every URL below should ideally score under 0.5. The
# file_sharing category is the one stage 5 already found and documented as still broken.
REAL_LEGITIMATE_URLS = {
    "wikipedia": [
        "https://en.wikipedia.org/wiki/Phishing",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Random_forest",
        "https://en.wikipedia.org/wiki/London",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Sign_language",
        "https://en.wikipedia.org/wiki/Greenwich",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Logistic_regression",
        "https://en.wikipedia.org/wiki/Vietnam",
    ],
    "github": [
        "https://github.com/torvalds/linux/blob/master/README",
        "https://github.com/python/cpython/blob/main/README.rst",
        "https://github.com/facebook/react/blob/main/README.md",
        "https://github.com/microsoft/vscode/blob/main/README.md",
        "https://github.com/pytorch/pytorch/blob/main/README.md",
        "https://github.com/huggingface/transformers/blob/main/README.md",
        "https://github.com/scikit-learn/scikit-learn/blob/main/README.rst",
        "https://github.com/pandas-dev/pandas/blob/main/README.md",
        "https://github.com/nodejs/node/blob/main/README.md",
        "https://github.com/khaidaodev/phishing-detector/blob/main/README.md",
    ],
    "gov_uk": [
        "https://www.gov.uk/apply-renew-passport",
        "https://www.gov.uk/register-to-vote",
        "https://www.gov.uk/apply-for-a-visa-to-enter-the-uk",
        "https://www.gov.uk/universal-credit",
        "https://www.gov.uk/student-finance",
        "https://www.gov.uk/browse/driving",
        "https://www.gov.uk/income-tax",
        "https://www.gov.uk/national-insurance",
        "https://www.gov.uk/council-tax",
        "https://www.gov.uk/apply-uk-citizenship",
    ],
    "stackoverflow": [
        "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
        "https://stackoverflow.com/questions/2074687/how-can-i-sort-a-dictionary-by-key",
        "https://stackoverflow.com/questions/419163/what-does-if-name-main-do",
        "https://stackoverflow.com/questions/6034067/random-numbers-with-seed",
        "https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists",
        "https://stackoverflow.com/questions/1832893/python-regex-matching-a-multiline-block-of-text",
        "https://stackoverflow.com/questions/13905741/accessing-python-dict-values-with-a-key",
        "https://stackoverflow.com/questions/509211/how-do-i-convert-a-list-into-a-string",
        "https://stackoverflow.com/questions/312443/how-do-you-convert-a-byte-array-to-a-hex-string",
        "https://stackoverflow.com/questions/100003/what-is-the-scope-of-a-variable",
    ],
    "docs": [
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map",
        "https://docs.python.org/3/library/itertools.html",
        "https://docs.djangoproject.com/en/5.0/topics/db/models/",
        "https://react.dev/reference/react/useState",
        "https://nodejs.org/api/fs.html",
        "https://pypi.org/project/scikit-learn/",
        "https://www.npmjs.com/package/express",
        "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html",
        "https://pkg.go.dev/net/http",
        "https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html",
    ],
    "news": [
        "https://www.bbc.co.uk/news/articles/c93q4z0zq7yo",
        "https://www.bbc.co.uk/news/technology-66000000",
        "https://www.theguardian.com/technology/2026/jan/15/ai-regulation-uk",
        "https://www.reuters.com/technology/artificial-intelligence/",
        "https://www.independent.co.uk/news/uk/politics",
        "https://www.telegraph.co.uk/news/2026/02/10/uk-economy-growth/",
        "https://news.sky.com/story/uk-weather-warning-12345678",
        "https://www.ft.com/content/abcdef12-3456-7890-abcd-ef1234567890",
        "https://www.nytimes.com/2026/03/01/technology/ai-news.html",
        "https://www.standard.co.uk/news/london/london-news-article-b1234567.html",
    ],
    "ecommerce": [
        "https://www.amazon.co.uk/dp/B08N5WRWNW",
        "https://www.amazon.com/dp/B0B2SF7S2G",
        "https://www.ebay.co.uk/itm/295678901234",
        "https://www.asos.com/asos-design/asos-design-oversized-t-shirt/prd/12345678",
        "https://www.argos.co.uk/product/1234567",
        "https://www.johnlewis.com/john-lewis-anyday-sofa/p1234567",
        "https://www.next.co.uk/style/st123456/789012",
        "https://www.currys.co.uk/products/laptop-1234567.html",
        "https://www.marksandspencer.com/womens-clothing/p/clp60123456",
        "https://www.etsy.com/listing/1234567890/handmade-gift",
    ],
    "university": [
        "https://www.ox.ac.uk/admissions/undergraduate/courses/computer-science",
        "https://www.cam.ac.uk/courses/computer-science",
        "https://www.ucl.ac.uk/prospective-students/undergraduate/degrees/computer-science-bsc",
        "https://www.imperial.ac.uk/study/ug/courses/computing-department/",
        "https://www.gre.ac.uk/undergraduate/courses/computing-with-artificial-intelligence-bsc-hons",
        "https://www.manchester.ac.uk/study/undergraduate/courses/2026/00050/bsc-computer-science/",
        "https://www.kcl.ac.uk/study/undergraduate/courses/computer-science-bsc",
        "https://www.bristol.ac.uk/study/undergraduate/2026/computer-science/",
        "https://www.leeds.ac.uk/course/undergraduate/108/computer-science",
        "https://www.ed.ac.uk/studying/undergraduate/degrees",
    ],
    "blog": [
        "https://medium.com/@blogs-world/how-to-learn-programming-in-2026-a-complete-practical-guide-for-beginners-and-career-switchers-794d70d8b24a",
        "https://parmardevendra23.medium.com/programming-in-2026-the-game-changing-languages-you-need-to-know-84c3ccfc76fe",
        "https://dev.to/bishnu_thakur_06ba044717b/why-java-still-rules-the-programming-world-in-2026-2a09",
        "https://medium.com/@SE_KE/2026-the-end-of-coding-or-its-greatest-big-bang-realities-post-gpt-5-3-codex-08113a391808",
        "https://medium.com/@atulprogrammer/the-future-of-programming-2026-will-change-everything-b53a50afee36",
        "https://dev.to/t/python",
        "https://joelgrus.com/2019/05/13/livecoding-madness-implement-your-own-sklearn-part-1/",
        "https://blog.pragmaticengineer.com/software-engineering-salaries-in-the-netherlands/",
        "https://danluu.com/input-lag/",
        "https://martinfowler.com/articles/microservices.html",
    ],
    "file_sharing": [
        "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view",
        "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit",
        "https://docs.google.com/spreadsheets/d/1a2b3c4d5e6f7g8h9i0j/edit",
        "https://docs.google.com/presentation/d/1a2b3c4d5e6f7g8h9i0j/edit",
        "https://www.dropbox.com/s/qw3rty12uiop45/report.pdf",
        "https://onedrive.live.com/view.aspx?resid=1234ABCD5678EFGH!123",
        "https://app.box.com/s/abc123def456ghi789",
        "https://wetransfer.com/downloads/abc123def456ghi789",
        "https://raw.githubusercontent.com/khaidaodev/phishing-detector/main/README.md",
        "https://www.icloud.com/iclouddrive/0abc123def456ghi789",
    ],
}


def evaluate_urls(model, urls_by_category: dict[str, list[str]]) -> dict:
    """Runs the URL model on every URL in every category. Since every URL here is genuinely
    legitimate, any prediction of "phishing" (proba >= 0.5) is a false positive. Returns a
    per-category breakdown plus an overall total under the "_overall" key."""
    results = {}
    total, total_fp = 0, 0

    for category, urls in urls_by_category.items():
        features = urls_to_feature_frame(pd.Series(urls))
        probas = model.predict_proba(features)[:, 1]
        fp_count = int((probas >= 0.5).sum())

        results[category] = {
            "count": len(urls),
            "false_positives": fp_count,
            "false_positive_rate": fp_count / len(urls),
            "mean_phishing_proba": float(probas.mean()),
        }
        total += len(urls)
        total_fp += fp_count

    results["_overall"] = {
        "count": total,
        "false_positives": total_fp,
        "false_positive_rate": total_fp / total if total else 0.0,
    }
    return results


def plot_results(results: dict, out_path: Path) -> None:
    categories = [c for c in results if not c.startswith("_")]
    rates = [results[c]["false_positive_rate"] * 100 for c in categories]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(categories, rates, color="#c0392b")
    plt.ylabel("False positive rate (%)")
    plt.title("Real legitimate URLs wrongly flagged as phishing, by category")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.0f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    model = joblib.load(MODELS_DIR / "url_baseline.joblib")
    results = evaluate_urls(model, REAL_LEGITIMATE_URLS)

    for category, stats in results.items():
        if category == "_overall":
            continue
        print(
            f"{category:15s} {stats['false_positives']:2d}/{stats['count']:2d} flagged as phishing "
            f"({stats['false_positive_rate']:.0%}), mean score {stats['mean_phishing_proba']:.3f}"
        )
    overall = results["_overall"]
    print(
        f"\noverall: {overall['false_positives']}/{overall['count']} false positives "
        f"({overall['false_positive_rate']:.1%})"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "real_url_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, RESULTS_DIR / "real_url_eval_accuracy.png")
    print(f"\nsaved results to {RESULTS_DIR}")
