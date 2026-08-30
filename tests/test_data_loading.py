"""
A couple of quick tests for the cleaning logic in src/data_loading.py.

Doesn't need internet or the real dataset, just checks the subject+body merging works right on a
tiny made up example.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loading import _load_one  # noqa: E402


def test_load_one_combines_subject_and_body(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "subject": ["Win a prize", "Meeting notes"],
            "body": ["Click here now", "See attached agenda"],
            "label": [1, 0],
        }
    ).to_csv(csv_path, index=False)

    out = _load_one(csv_path)

    assert list(out.columns) == ["text", "label", "source"]
    assert out.loc[0, "text"] == "Win a prize Click here now"
    assert out.loc[0, "label"] == 1
    assert out.loc[1, "label"] == 0
    assert (out["source"] == "sample").all()


def test_load_one_handles_missing_subject(tmp_path):
    csv_path = tmp_path / "no_subject.csv"
    pd.DataFrame({"body": ["Just a body"], "label": [0]}).to_csv(csv_path, index=False)

    out = _load_one(csv_path)

    assert out.loc[0, "text"] == "Just a body"
