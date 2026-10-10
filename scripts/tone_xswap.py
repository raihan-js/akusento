"""M3 cross-swap: minimal pairs share phones, differ only in tones.

Render 箸が with 橋が's tones and vice versa (same for 雨/飴). If given_tone
controls output, each render's F0/scorer output follows the TONES, not text.
Metric: our scorer's mora H/L on each render vs forced pattern + F0 flip rate.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tone_swap import load_model, phone_f0
from style_bert_vits2.nlp.japanese.g2p import g2p

PAIRS = [("箸が。", "橋が。"), ("雨が。", "飴が。")]


def main():
    import json
    import sys
    sys.path.insert(0, "src")

    model = load_model()
    out = {}
    for ta, tb in PAIRS:
        pa, ta_tones, _ = g2p(ta)
        pb, tb_tones, _ = g2p(tb)
        assert pa == pb, "minimal pairs must share phones"
        res = {}
        for text, tones, tag in ((ta, ta_tones, "Aauto"), (ta, tb_tones, "AwithB"),
                                 (tb, tb_tones, "Bauto"), (tb, ta_tones, "BwithA")):
            sr, wav = model.infer(text=text, line_split=False,
                                  given_phone=pa, given_tone=tones)
            p = Path(f"results/xswap_{tag}.wav")
            wavfile.write(p, sr, wav)
            f = phone_f0(wav, sr, len(pa))
            res[tag] = {"tones": tones, "f0": [round(float(x), 2) if x == x else None for x in f]}
        # Control score: does AwithB's contour match Bauto better than Aauto?
        def vec(d):
            v = np.array(d, dtype=float)
            return v[~np.isnan(v)]
        a, ab, b, ba = (vec(res[k]["f0"]) for k in ("Aauto", "AwithB", "Bauto", "BwithA"))
        n = min(map(len, (a, ab, b, ba)))
        a, ab, b, ba = a[:n], ab[:n], b[:n], ba[:n]

        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        out[f"{ta}/{tb}"] = {
            "AwithB_vs_Bauto": corr(ab, b), "AwithB_vs_Aauto": corr(ab, a),
            "BwithA_vs_Aauto": corr(ba, a), "BwithA_vs_Bauto": corr(ba, b),
        }
        print(ta, tb, out[f"{ta}/{tb}"])
    Path("results/xswap.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote results/xswap.json")


if __name__ == "__main__":
    main()
