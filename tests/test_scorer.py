"""Scorer unit tests (synthetic features; real-data eval is the M2 script)."""

import numpy as np

from akusento.scorer.features import FEATURE_NAMES, mora_features
from akusento.scorer.train import accuracy, train


def test_features_none_when_unvoiced():
    t = np.arange(0, 1.0, 0.005)
    f0 = np.zeros_like(t)
    assert mora_features(f0, t, 0.2, 0.4, 150.0, 1, 3) is None


def test_features_direction():
    t = np.arange(0, 1.0, 0.005)
    f0 = np.linspace(100, 200, len(t))  # rising
    f = mora_features(f0, t, 0.2, 0.8, 150.0, 2, 3)
    assert f is not None and len(f) == len(FEATURE_NAMES)
    assert f[-1] > 0  # slope positive


def test_logreg_learns_separable():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (50, 9)), rng.normal(2, 1, (50, 9))])
    y = np.array([0] * 50 + [1] * 50)
    sc, clf = train(X, y)
    assert accuracy(clf, sc, X, y) > 0.95
