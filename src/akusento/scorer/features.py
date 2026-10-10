"""Per-mora acoustic features for the pitch-accent scorer (M2).

Features (all from aligned audio + utterance-normalized F0, no labels):
mean, std, min, max of semitone F0 in the mora; voiced rate; log duration;
position index in phrase; phrase length; slope (last-third mean minus
first-third mean, semitones). Target: H/L from OpenJTalk labels
(truth-with-acoustic-agreement; disclosed, not human gold).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def mora_features(f0: npt.NDArray[np.float64], t: npt.NDArray[np.float64], start: float, end: float,
                  med: float, pos_in_phrase: int, phrase_len: int) -> list[float] | None:
    """Feature vector or None when the mora has too little voicing."""
    m = (t >= start) & (t < end) & (f0 > 0)
    frames = int((t < end).sum() - (t < start).sum())
    if m.sum() < 3 or m.sum() / max(frames, 1) < 0.3 or med <= 0:
        return None
    v = 12 * np.log2(f0[m] / med)
    thirds = np.array_split(v, 3)
    slope = float(thirds[-1].mean() - thirds[0].mean()) if len(thirds[-1]) and len(thirds[0]) else 0.0
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
            float(m.sum() / max(frames, 1)), float(np.log10(max(end - start, 1e-3))),
            float(pos_in_phrase), float(phrase_len), slope]


FEATURE_NAMES = ["f0_mean_st", "f0_std", "f0_min", "f0_max", "voiced_rate",
                 "log_dur", "pos_in_phrase", "phrase_len", "slope_st"]
# v2 appends phrase-relative context (promoted 2026-10-11).
FEATURE_NAMES_V2 = FEATURE_NAMES + ["cur_rel_phrase_st", "prev_rel_phrase_st", "next_rel_phrase_st"]


def phrase_feature_matrix(f0: npt.NDArray[np.float64], t: npt.NDArray[np.float64],
                          segs: list[dict[str, Any]],
                          n_morae: int) -> tuple[npt.NDArray[np.float64], list[bool]]:
    """v2 feature rows for one phrase's mora segments (equal order).

    Phrase-median normalization (downstep-robust) + neighbor context.
    Returns (X, valid): rows with too little voicing are zero-filled and
    flagged invalid (caller decides; training skips them).
    """
    pm: list[float | None] = []
    for s in segs:
        m = (t >= float(s["start"])) & (t < float(s["end"])) & (f0 > 0)
        pm.append(float(np.median(f0[m])) if m.sum() >= 3 else None)
    ok = [v for v in pm if v is not None]
    pmed = float(np.median(ok)) if ok else 0.0
    rows: list[list[float]] = []
    valid: list[bool] = []
    for j, s in enumerate(segs):
        f = mora_features(f0, t, float(s["start"]), float(s["end"]), pmed, j + 1, n_morae)
        rows.append(f if f is not None else [0.0] * len(FEATURE_NAMES))
        valid.append(f is not None)
    X = np.array(rows)
    ctx = []
    for j in range(n_morae):
        cur = pm[j] if pm[j] is not None else pmed
        pv = pm[j - 1] if j > 0 else None
        prevm = pv if pv is not None else pmed
        nv = pm[j + 1] if j + 1 < n_morae else None
        nextm = nv if nv is not None else pmed
        if pmed <= 0:
            ctx.append([0.0, 0.0, 0.0])
        else:
            ctx.append([float(12 * np.log2(cur / pmed)), float(12 * np.log2(prevm / pmed)),
                        float(12 * np.log2(nextm / pmed))])
    return np.hstack([X, np.array(ctx)]), valid
