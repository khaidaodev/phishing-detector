# About the data

I haven't committed the actual data files to this repo (see `.gitignore`). Partly because they're big (around 250MB total), and partly because I'm not 100% sure I'm allowed to redistribute all of it myself, since it's a mix of a few different sources.

## The email text data

`src/data_loading.py` grabs the email data automatically when you run it. It's a combined dataset that's actually 6 smaller, older email datasets stuck together, originally put together by someone on Kaggle called naserabdullahalam ([link here](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)). I ended up pulling it from a GitHub copy of the same files instead ([rokibulroni/Phishing-Email-Dataset](https://github.com/rokibulroni/Phishing-Email-Dataset)) since I couldn't get the Kaggle download working in the environment I was building this in.

The 6 pieces:

| Source | Rows | What it is |
|---|---|---|
| CEAS_08 | 39,154 | mixed real/phishing emails from a 2008 spam challenge |
| Enron | 29,767 | mostly real corporate emails, some spam |
| Ling | 2,859 | the Ling-Spam dataset |
| Nazario | 1,565 | all phishing |
| Nigerian_Fraud | 3,332 | all phishing (the classic "419" scam emails) |
| SpamAssasin | 5,809 | from the SpamAssassin project |

About 82,500 rows once combined and cleaned up. In the `label` column, 1 means phishing/spam, 0 means a real email.

Running `python src/data_loading.py` downloads all of this into `data/raw/` the first time (takes a minute, don't panic if it seems slow), then saves a cleaned up combined version to `data/processed/combined.csv`, which is what the model scripts actually read from.

## Link/URL data

Going with PhiUSIIL ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)), about 236,000 real URLs, roughly half legitimate and half phishing.

`src/url_data_loading.py` downloads and unzips it automatically the first time you run `python src/url_baseline.py` (or `python src/url_data_loading.py` on its own). If the direct download link ever stops working, grab the zip manually from the link above and drop the extracted CSV into `data/raw/phiusiil/`.

PhiUSIIL's own label column is the other way round to mine (1 = legitimate there, vs 1 = phishing in the email dataset), so `url_data_loading.py` flips it to keep both datasets consistent throughout the project.

It also comes with roughly 50 extra features already worked out (whether the page has a favicon, how many images it has, and so on), all based on actually crawling the live webpage. I don't use any of that here, since a real detector might be checking a link that's already offline by the time anyone looks, or might not have live internet access to go check in the first place. `url_data_loading.py` only keeps the raw `URL` column and the label; everything else gets pulled out of just the URL text itself in `src/url_features.py`.
