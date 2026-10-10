"""CER via ASR roundtrip on auto renders (intelligibility floor for M3).

Re-renders group sentences (auto tones), transcribes with faster-whisper
large-v3-turbo, reports CER (character-level, kana-normalized) + per-item
rows. Writes results/cer.json. (jiwer-free: tiny edit-distance inline.)
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tone_swap import load_model
from minimal_pairs import GROUPS
from style_bert_vits2.nlp.japanese.g2p import g2p


def edit(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(dp[j] + 1, ndp[-1] + 1, dp[j - 1] + (ca != cb)))
        dp = ndp
    return dp[-1]


def norm(s: str) -> str:
    return "".join(ch for ch in s if "\u3041" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")


def transcribe_wav(asr, path):
    """Array input bypasses PyAV (version-skewed in this env). Resample to 16k."""
    from math import gcd

    from scipy.signal import resample_poly

    sr, wav = wavfile.read(path)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    if sr != 16000:
        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    segs, _ = asr.transcribe(wav, language="ja")
    return "".join(sg.text for sg in segs)


def main():
    import json

    from faster_whisper import WhisperModel

    asr = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    model = load_model()
    tot_ed = tot_n = 0
    rows = []
    n = 0
    for reading, words in GROUPS:
        for w in words:
            s = f"{w}がいい。"
            try:
                phones, _, _ = g2p(s)
            except Exception:  # noqa: BLE001
                continue
            sr, wav = model.infer(text=s, line_split=False)
            p = Path(f"results/cer_{n:03d}.wav")
            wavfile.write(p, sr, wav)
            hyp = norm(transcribe_wav(asr, str(p)))
            ref = norm(s)
            ed = edit(ref, hyp)
            tot_ed += ed
            tot_n += max(len(ref), 1)
            rows.append({"text": s, "hyp": hyp, "cer": round(ed / max(len(ref), 1), 3)})
            n += 1
            if n >= 60:
                break
        if n >= 60:
            break
    out = {"cer": round(tot_ed / max(tot_n, 1), 4), "n": n, "rows": rows}
    Path("results/cer.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"cer": out["cer"], "n": n}))


if __name__ == "__main__":
    main()
