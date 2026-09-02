# Phishing Email Detector

A model that looks at an email and predicts whether it's phishing (a scam trying to steal your info) or a normal, legit email.

## Why this problem

Pretty much everyone's had a dodgy email or text land in their inbox trying to trick them at some point. Most phishing detectors I looked at online only check one thing, either the wording of the message or the link inside it, but not both. Real phishing usually uses both together (urgent scary wording + a dodgy link), so I want to eventually check both. I also don't want to just show a nice accuracy score and call it done, I want to actually try to break my own model and see where it fails.

## Where it's at right now

I've got the first working piece done: a model that reads just the text of an email (no links yet) and guesses phishing or not.

How it works: first the text gets turned into numbers using something called TF-IDF, which basically scores how unusual or suspicious a word is in a message rather than just counting how often it shows up. Then a logistic regression model (a fairly simple, beginner-level type of ML model) learns from those numbers to make its guess.

Results on the test set (about 82,000 emails total, kept 20% aside for testing):

| Metric | Legit emails | Phishing emails |
|---|---|---|
| Precision | 0.99 | 0.99 |
| Recall | 0.98 | 0.99 |

ROC-AUC came out to 0.999, which is basically a score out of 1 for how well the model can tell the two apart.

Honestly, that score is probably too good to fully trust. My dataset is actually 6 different smaller datasets glued together, and a couple of them are 100% phishing or 100% real emails on their own with nothing mixed in. So there's a chance the model is partly just learning "which of the 6 datasets does this sound like" rather than actually spotting what makes something phishing. That's exactly why I want to do the "try to trick it on purpose" testing later, to see if it actually holds up or if it's cheating in a way.

Plots and the full numbers are in the `results/` folder if you want to look closer.

## Stage 2: the link/URL model

Second piece is done: a separate model that looks only at a URL and guesses phishing or not, no email text involved at all.

How it works: `src/url_features.py` pulls a set of numbers out of the raw URL string, stuff like how long it is, how many subdomains it has, whether it's using a raw IP address instead of a domain name, whether it's HTTPS, whether it has an `@` in it (a classic trick for hiding the real destination), suspicious words like "verify" or "secure-login", how "random-looking" the domain name is, and whether it's a known link shortener. All of that only needs the URL text itself, nothing about the actual webpage, since a real detector might be checking a link to a page that's already offline, or might not have internet access to go check it live anyway.

Those features get fed into a Random Forest (a bunch of decision trees that each vote, then you go with the majority), trained on the PhiUSIIL dataset from UCI, about 236,000 real URLs, roughly half legit and half phishing.

Results on the test set (about 235,000 URLs total, kept 20% aside for testing):

| Metric | Legit URLs | Phishing URLs |
|---|---|---|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 0.99 |

ROC-AUC came out to 0.998.

Honestly, about as suspicious as the text model's score. Seventeen hand picked numbers shouldn't be enough to get a URL right basically every time, that's the kind of score you get when the dataset has some obvious shortcut in it rather than the model actually being that good at spotting phishing in general. Turned out to be exactly that, see stage 5 below for what the shortcut actually was and what I did about it.

Plots (confusion matrix, ROC curve, and which features the forest actually used most) are in `results/`.

## Stage 3: combining the two models (in progress)

Started on this, `src/combine_model.py` now runs a message through both models and gives back one combined phishing score, but it's a small first step, not the full thing yet.

First thing I actually tried was training one model on top of both datasets combined (proper "stacking"), but checked that properly before writing any code and it doesn't work here. The email dataset has a `urls` column saying whether a message had a link in it, but not the actual URL text, most real URLs got stripped out when the original emails were converted from HTML to plain text (a link became text like "Update Your Account" with the href gone). Only a small slice of rows still have a usable URL sitting in the text, nowhere near enough to train on properly. On top of that, the URL model's trained on the completely separate PhiUSIIL dataset, so there's no shared set of examples with both a real email body and a real URL to combine features from in the first place.

So `combine_model.py` combines the two models at prediction time instead: for a new message (with its link still intact, unlike the training data), run the text model on the body and the URL model on the link, then take whichever of the two scores is higher as the combined result. No joint training data needed for that, and it's arguably more realistic anyway, that's exactly what a live detector would actually see.

What's still open: there's no big labeled dataset of full messages with intact URLs to check how well the *combined* score performs (that's the same data gap as above). The handwritten test set built for the adversarial testing below (see stage 4) doubles as this evaluation, but it's only 12 examples, so it's a sanity check rather than a real statistically solid number.

## Stage 4: breaking my own model on purpose

`src/adversarial.py` has a small set of 12 handwritten messages (6 phishing, 6 legit) with the right answer for each one, some with a URL attached, some without. This also doubles as the "proper evaluation" for the combined model that stage 3 above was missing, since it's exactly what was needed: real messages with intact URLs and known right answers.

Four tricks, real ones phishers actually use:

- `inject_typos`: swaps or drops a letter in some words, the sort of thing that happens by accident but also gets used on purpose to dodge keyword filters
- `add_extra_spacing`: breaks a word up like "v e r i f y", an old spam filter trick that only works if the filter's matching on the exact word
- `homoglyph_substitute`: swaps a letter in a URL's domain for a lookalike character, e.g. paypal.com -> payp4l.com, real typosquatting territory
- `apply_reworded_urgency`: a calmer, less panicky hand-rewrite of each phishing example, same scam, none of the "URGENT", "act now", "24 hours" language. This one's hand-written rather than a generic function on purpose, deleting a fixed list of urgent-sounding words would only prove the model depends on those exact words, not whether a more patient-sounding phisher can still get the same scam through.

Results on the 12 handwritten examples:

| Test | Accuracy |
|---|---|
| Baseline (no tricks) | 11/12 (92%) |
| Typos | 11/12 (92%) |
| Extra spacing | 11/12 (92%) |
| Homoglyph URL swap (7 examples that have a URL) | 6/7 (86%) |
| Reworded urgency (6 phishing examples) | 6/6 (100%) |

**What actually happened:** none of the four tricks broke anything new. The only wrong answer anywhere in this table is the same one example, in every row, a legit message with a Google Drive link in it ("Here's the recording from today's meeting...") that's wrong even at baseline with no tricks applied at all. So the actual finding here isn't "the model resisted every attack", it's "none of these four specific tricks moved the needle on this particular model", which is a narrower and more honest claim.

**Why, probably:** the URL model leans hard on structural features (is it HTTPS, is it a raw IP, does the domain have digits in it, that sort of thing), and none of the four tricks touch those in a way that would flip a prediction, homoglyph swapping only changes 1-2 letters, it doesn't turn a normal-looking domain into an IP address or strip the HTTPS. The text model is TF-IDF based, so it scores on word patterns rather than a fixed keyword list, that's probably why reworded_urgency didn't dent it either: a TF-IDF model that's seen enough real phishing emails likely picked up on softer scam patterns too (a link plus a plausible-sounding excuse to click it), not just the shoutiness.

**Is the Google Drive miss fixable:** haven't dug into which of the two models is actually responsible yet, `combine_model.py` would need a small debugging pass (print `text_proba` and `url_proba` separately for that one example) to find out. Worth doing at some point, but it's one example out of twelve, not something to over-tune the model around.

**Honest limitation:** 12 handwritten examples is a sanity check, not a rigorous adversarial evaluation. A real one would need way more examples, ideally covering a wider range of scam types, and probably some tricks that are actually designed to target this specific model's weak points rather than generic ones borrowed from old spam-filter history.

Plot's in `results/adversarial_accuracy.png`.

## Stage 5: found (and partly fixed) a bias in the URL training data

Went back to the one wrong answer from stage 4, the legit Google Drive link that kept getting flagged as phishing across every single trick, to find out why before moving on.

**What I found:** every single one of the ~135,000 "legitimate" URLs in PhiUSIIL is a bare homepage, literally `https://www.something.tld` with nothing after it, not one exception in the whole dataset. "Phishing" ones do have real paths a decent chunk of the time. That's not a real-world pattern about what makes a URL legitimate, it's just an artifact of how this dataset happened to get collected, but the model learned it anyway: "does this URL have a path at all" was basically a free giveaway for "phishing" to it. That's exactly why real links like paypal.com/signin (an actual page, not a homepage) or a Google Doc link were getting flagged. It also explains the suspiciously good 0.998 ROC-AUC back in stage 2, a lot of that score was the model spotting this dataset quirk, not genuinely spotting phishing.

**The fix:** `_augment_legitimate_urls_with_paths` in `src/url_data_loading.py` now tacks a realistic path (`/login`, `/blog/post-1`, `/account/settings`, that sort of thing) onto half of the legitimate URLs before training, so "has a path" stops being a free shortcut. Retrained and checked against a handful of real, definitely-legitimate URLs I picked myself, none of them from PhiUSIIL:

| URL | Phishing probability before | After the fix |
|---|---|---|
| paypal.com/signin | 0.997 | 0.069 |
| bbc.co.uk/news/technology | 1.000 | 0.024 |
| en.wikipedia.org/wiki/Phishing | 1.000 | 0.007 |
| drive.google.com/file/d/.../view | 1.000 | 0.978 |
| docs.google.com/document/d/.../edit | 1.000 | 0.987 |
| dropbox.com/s/.../report.pdf | 1.000 | 0.983 |
| github.com/.../blob/main/README.md | 1.000 | 1.000 |

Fixed it for URLs with normal, word-based paths. Didn't fix it for file-sharing style links.

**What's still broken, and why it's a harder problem:** Google Drive, Docs, Dropbox and GitHub links all use random-looking alphanumeric segments to point at a specific file (`abc123`, `1a2b3c4d5e`), and so do a lot of real phishing kits, for tracking IDs or just to look more official. My features genuinely can't tell those two apart, a random string is a random string whether it's pointing at someone's real file or a fake login page. Properly fixing this would need either a feature that specifically recognises known file-sharing platforms as a strong "probably fine" signal, or a better dataset whose legitimate class actually includes real file-sharing links to begin with. Neither of those is a quick fix, so it's staying as a documented limitation rather than something I patch around.

If you're re-running this yourself: `python src/url_data_loading.py` first to rebuild `data/processed/url_dataset.csv` with whatever augmentation logic is currently in `url_data_loading.py`, then `python src/url_baseline.py` to retrain on it. The dataset doesn't rebuild itself automatically, if the CSV's already there it just gets reused as-is, learned that the hard way in stage 6 below.

## Stage 6: the stage 5 fix didn't actually generalize

Stage 5's own check was 7 URLs I picked by hand and typed into a table. Built something more honest: `src/real_url_eval.py`, 100 real legitimate URLs across 10 categories (Wikipedia, GitHub, gov.uk, Stack Overflow, dev docs, news, e-commerce, university, blog posts, and the file-sharing links stage 5 already knew about), and measured the actual false-positive rate per category instead of eyeballing a handful of examples.

**First result, against the stage 5 model: 84 out of 100 (84%) false positives.** Basically broken outside the handful of examples stage 5 happened to check. Turns out that was partly bad luck in which examples got picked: stage 5's "after" table included `bbc.co.uk/news/technology`, and the fix's augmentation list happened to contain the literal path `/news/technology` as one of its 18 templates. The check wasn't wrong, it just wasn't representative.

**Why:** the 18 template paths stage 5 used to teach the model "paths are normal" were short (13 characters on average) and barely hyphenated (0.17 hyphens on average). Real URLs I tested averaged 29 characters and 1.8 hyphens. So the model didn't learn "a path is normal", it learned "a *short, clean* path is normal", and anything longer or messier (which is most real content, a Wikipedia title, a Stack Overflow question slug, a product page) still read as phishing to it.

**Fix v2:** replaced the fixed list with `_generate_realistic_path` in `src/url_data_loading.py`, a small generator that builds a fresh, randomised path for every augmented row instead of repeating 18 strings, single words, hyphenated multi-word slugs, alphanumeric ids, occasionally a file extension or a query string, so the training data actually spans the range of shapes real paths come in rather than one narrow corner of it.

**Retrained, reran the same 100-URL test: 46 out of 100 (46%) false positives.** Real progress, and not evenly spread:

| Category | Before (18 templates) | After (generated paths) |
|---|---|---|
| Wikipedia | 60% | 10% |
| News | 90% | 10% |
| E-commerce | 100% | 10% |
| University | 100% | 0% |
| File-sharing | 100% | 20% |
| Docs | 80% | 40% |
| Blog | 100% | 80% |
| Gov.uk | 10% | 90% |
| GitHub | 100% | 100% |
| Stack Overflow | 100% | 100% |

GitHub and Stack Overflow didn't move at all, both use long, multi-segment URLs that mix a numeric id with a long hyphenated slug in the same path, a specific combination the generator still doesn't cover well. Gov.uk actually got *worse*, its URLs are short but heavily hyphenated single segments (`/apply-renew-passport`), and whatever the model's decision boundary is now, that specific shape moved to the wrong side of it. Chasing that with a third round of template tweaks would just be whack-a-mole.

**The bigger honest point:** the official train/test split metrics for this model (in the results above) still say 99% precision. That number isn't wrong, but it's not that meaningful either, both the train and test split come from the same synthetic augmentation, so a held-out split from it can't tell you whether the augmentation itself is realistic. That's exactly why this stage's 100-URL check, built from real sites rather than PhiUSIIL, mattered: it caught something the official metric couldn't.

**Where this leaves it:** two rounds of synthetic path augmentation took the real-world false-positive rate from 84% down to 46%. Real, measured improvement, but there's a ceiling on how far this approach can go, PhiUSIIL just doesn't contain real legitimate URLs with real paths, and no amount of synthetic augmentation fully substitutes for that. A proper fix would need a second, real dataset of legitimate URLs that actually have paths on them. Leaving it there as an honestly measured limitation rather than patching a third time.

## What's still left to build

- A real dataset of legitimate URLs with genuine paths, not a synthetic one, to get past the 46% false-positive rate stage 6 landed on, GitHub/Stack Overflow-style URLs and gov.uk-style short hyphenated ones are the two places it's still clearly wrong
- Trying a smarter combining rule than "take the higher score" for stage 3, now that the adversarial test set gives something to measure it against
- Maybe a tiny website at the end where you paste a message in and it tells you phishing or not

Notes for the last one are in `demo/app.py`.

## How to run this yourself

```bash
pip install -r requirements.txt
python src/text_baseline.py     # text model
python src/url_data_loading.py  # rebuilds the URL training data (needed before url_baseline.py)
python src/url_baseline.py      # link/URL model
python src/combine_model.py     # runs a few example messages through both models combined
python src/adversarial.py       # tests the combined model against the tricks in stage 4
python src/real_url_eval.py     # tests the URL model against 100 real legitimate URLs, stage 6
```

First time you run the text or URL model, it downloads the relevant dataset automatically (email data is around 250MB, URL data around 15MB), then trains the model and prints out the results. `combine_model.py` and `adversarial.py` need the text and URL models trained first (they load the saved `.joblib` files from `models/`).

## Where the data's from

Kaggle has a "Phishing Email Dataset" that combines a bunch of older, well known email datasets. I couldn't get Kaggle's own download working from where I was building this, so I grabbed the same data off a GitHub copy of it instead, linked in `data/README.md` along with more detail on where each piece came from.

The URL model uses PhiUSIIL, a dataset of real URLs from the UCI Machine Learning Repository. It actually comes with about 50 features already calculated (stuff about the live webpage, like whether it has a favicon), but I ignore basically all of that and only keep the raw URL text, since a real detector can't always go and crawl the page live. More detail in `data/README.md`.

## Testing and git

`tests/` has pytest tests for the data loading (email and URL), the URL feature extraction, the URL path augmentation from stages 5 and 6, the combining step, the adversarial tricks in stage 4, and the real-URL evaluation from stage 6. I run them before committing anything that touches the models themselves. Commits are broken up by stage too, baseline text model, then the URL model, then combining the two, then adversarial testing, then the two rounds of fixing the URL training data bias, rather than one big dump at the end, so the history actually shows the order this got built in.

## Tools used

Python, pandas, scikit-learn (TF-IDF + logistic regression for the text model, Random Forest for the URL model), tldextract for pulling domains/subdomains out of URLs, matplotlib for the plots. Planning to add Hugging Face's `transformers` library later for a proper transformer model.
