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

## What's still left to build

- A second model that looks at the actual links in the message (how long the link is, weird characters, fake-looking web addresses, that sort of thing)
- Combining the text model and the link model into one
- Trying to break my own model on purpose, typos, characters that look like letters but aren't, extra spaces, reworded scary language, and seeing how much it messes the model up
- Maybe a tiny website at the end where you paste a message in and it tells you phishing or not

None of that is built yet, but I left notes for myself in each file (`src/url_features.py`, `src/combine_model.py`, `src/adversarial.py`, `demo/app.py`) so I remember the plan when I get to it.

## How to run this yourself

```bash
pip install -r requirements.txt
python src/text_baseline.py
```

First time you run it, it downloads the email dataset automatically (around 250MB, so it takes a minute or two), then trains the model and prints out the results.

## Where the data's from

Kaggle has a "Phishing Email Dataset" that combines a bunch of older, well known email datasets. I couldn't get Kaggle's own download working from where I was building this, so I grabbed the same data off a GitHub copy of it instead, linked in `data/README.md` along with more detail on where each piece came from.

## Tools used

Python, pandas, scikit-learn for the model, matplotlib for the plots. Planning to add Hugging Face's `transformers` library later for a proper transformer model, and XGBoost for the link-based model.
