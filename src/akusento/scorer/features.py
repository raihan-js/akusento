"""Per-mora acoustic features for the pitch-accent scorer (M2).

Features (all from aligned audio + utterance-normalized F0, no labels):
mean, std, min, max of semitone F0 in the mora; voiced rate; log duration;
position index in phrase; phrase length; slope (last-third mean minus
first-third mean, semitones). Target: H/L from OpenJTalk labels
(truth-with-acoustic-agreement; disclosed, not human gold).
"""

from __future__ import annotations

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
