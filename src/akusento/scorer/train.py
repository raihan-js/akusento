"""M2 training: logistic regression on mora features -> P(H).

Baseline: majority class. Metric: mora-level H/L accuracy on held-out
utterances + word accent-type accuracy (argmax pattern per phrase from
predicted mora labels is reported, not optimized). Deterministic
(random_state fixed, liblinear/lbfgs fixed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xs, y)
    return sc, clf


def accuracy(clf: LogisticRegression, sc: StandardScaler, X: np.ndarray, y: np.ndarray) -> float:
    return float(clf.score(sc.transform(X), y))


def save(path: str | Path, sc: StandardScaler, clf: LogisticRegression, meta: dict) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"scaler": sc, "clf": clf, "meta": meta}, f)


def load(path: str | Path) -> tuple[StandardScaler, LogisticRegression, dict]:
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)  # noqa: S301 - own artifact, documented
    return d["scaler"], d["clf"], d["meta"]


def report_json() -> str:
    return json.dumps({"model": "logistic", "features": "mora_f0"})
