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

Honestly, about as suspicious as the text model's score. Seventeen hand picked numbers shouldn't be enough to get a URL right basically every time, that's the kind of score you get when the dataset has some obvious shortcut in it rather than the model actually being that good at spotting phishing in general. Same plan as the text model: hold off trusting this until I've done the adversarial testing (`src/adversarial.py`) and actually tried to break it with URLs designed to look tricky.

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

## What's still left to build

- Track down which model is flagging the Google Drive example and see if it's an easy fix
- Trying a smarter combining rule than "take the higher score" for stage 3, now that the adversarial test set gives something to measure it against
- Maybe a tiny website at the end where you paste a message in and it tells you phishing or not

Notes for the last one are in `demo/app.py`.

## How to run this yourself

```bash
pip install -r requirements.txt
python src/text_baseline.py     # text model
python src/url_baseline.py      # link/URL model
python src/combine_model.py     # runs a few example messages through both models combined
python src/adversarial.py       # tests the combined model against the tricks in stage 4
```

First time you run the text or URL model, it downloads the relevant dataset automatically (email data is around 250MB, URL data around 15MB), then trains the model and prints out the results. `combine_model.py` and `adversarial.py` need the text and URL models trained first (they load the saved `.joblib` files from `models/`).

## Where the data's from

Kaggle has a "Phishing Email Dataset" that combines a bunch of older, well known email datasets. I couldn't get Kaggle's own download working from where I was building this, so I grabbed the same data off a GitHub copy of it instead, linked in `data/README.md` along with more detail on where each piece came from.

The URL model uses PhiUSIIL, a dataset of real URLs from the UCI Machine Learning Repository. It actually comes with about 50 features already calculated (stuff about the live webpage, like whether it has a favicon), but I ignore basically all of that and only keep the raw URL text, since a real detector can't always go and crawl the page live. More detail in `data/README.md`.

## Testing and git

`tests/` has pytest tests for the data loading, the URL feature extraction, the combining step, and the adversarial tricks in stage 4. I run them before committing anything that touches the models themselves. Commits are broken up by stage too, baseline text model, then the URL model, then combining the two, then the adversarial testing, rather than one big dump at the end, so the history actually shows the order this got built in.

## Tools used

Python, pandas, scikit-learn (TF-IDF + logistic regression for the text model, Random Forest for the URL model), tldextract for pulling domains/subdomains out of URLs, matplotlib for the plots. Planning to add Hugging Face's `transformers` library later for a proper transformer model.
