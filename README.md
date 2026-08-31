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

## What's still left to build

- Combining the text model and the link model into one
- Trying to break my own model on purpose, typos, characters that look like letters but aren't, extra spaces, reworded scary language, and seeing how much it messes the model up
- Maybe a tiny website at the end where you paste a message in and it tells you phishing or not

None of that is built yet, but I left notes for myself in each file (`src/combine_model.py`, `src/adversarial.py`, `demo/app.py`) so I remember the plan when I get to it.

## How to run this yourself

```bash
pip install -r requirements.txt
python src/text_baseline.py     # text model
python src/url_baseline.py      # link/URL model
```

First time you run either one, it downloads the relevant dataset automatically (email data is around 250MB, URL data around 15MB), then trains the model and prints out the results.

## Where the data's from

Kaggle has a "Phishing Email Dataset" that combines a bunch of older, well known email datasets. I couldn't get Kaggle's own download working from where I was building this, so I grabbed the same data off a GitHub copy of it instead, linked in `data/README.md` along with more detail on where each piece came from.

The URL model uses PhiUSIIL, a dataset of real URLs from the UCI Machine Learning Repository. It actually comes with about 50 features already calculated (stuff about the live webpage, like whether it has a favicon), but I ignore basically all of that and only keep the raw URL text, since a real detector can't always go and crawl the page live. More detail in `data/README.md`.

## Tools used

Python, pandas, scikit-learn (TF-IDF + logistic regression for the text model, Random Forest for the URL model), tldextract for pulling domains/subdomains out of URLs, matplotlib for the plots. Planning to add Hugging Face's `transformers` library later for a proper transformer model.
