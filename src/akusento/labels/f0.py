"""F0 extraction (pyworld DIO + StoneMask, 5ms frames).

Outputs per utterance: f0 (Hz, 0 = unvoiced), time axis, mean/std of voiced.
pyworld is the M1 workhorse (fast, CPU); torchcrepe spot-checks a subset
(if installed) and any systematic bias is logged, not silently kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_f0(wav: np.ndarray, sr: int, frame_ms: float = 5.0) -> dict[str, Any]:
    import pyworld as pw

    x = np.asarray(wav, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame_period = frame_ms
    f0, t = pw.dio(x, sr, frame_period=frame_period)
    f0 = pw.stonemask(x, f0, t, sr)
    voiced = f0[f0 > 0]
    return {
        "f0": f0.astype(np.float32),
        "t": t.astype(np.float32),
        "voiced_rate": float(len(voiced) / max(len(f0), 1)),
        "voiced_mean": float(voiced.mean()) if len(voiced) else 0.0,
        "voiced_std": float(voiced.std()) if len(voiced) else 0.0,
    }


def load_wav_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load + resample to mono target_sr with scipy only (no librosa).

    (librosa 1.x needs numpy 2, which breaks pyopenjtalk's ABI in this env;
    TTS deps live in a separate venv — see docs/decisions.md.)
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, y = wavfile.read(path)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # Normalize int16 -> [-1, 1].
    if y.dtype == np.float32 and np.abs(y).max() > 1.0:
        y = y / 32768.0
    if sr != target_sr:
        from math import gcd

        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
    return y, target_sr
