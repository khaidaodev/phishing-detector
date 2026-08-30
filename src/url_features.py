"""
Haven't started this bit yet, this is where the link-checking model will go.

The idea:
    - Pull out features from any links in the email: how long the link is, how many subdomains
      it has, whether it uses an IP address instead of a normal domain name, whether it's HTTPS,
      suspicious words, and letter-swap tricks (like using a 0 instead of an O).
    - Feed those into a tree-based model (random forest or XGBoost, still deciding).
    - Data: probably the Kaggle "Phishing URLs Dataset with Extracted Features" or the PhiUSIIL
      one from UCI (see data/README.md). Kaggle needs an API key which I didn't have set up
      everywhere I was building this, so might need to download the CSV by hand instead.

To do:
    - [ ] extract_url_features(url) -> some kind of feature dict
    - [ ] turn a list of urls into a dataframe of features
    - [ ] train the model, check it the same way as text_baseline.py
"""

raise NotImplementedError("haven't built this bit yet, see the plan in README.md")
