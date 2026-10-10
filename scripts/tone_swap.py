"""M3 control test: same text, auto vs inverted tones -> F0 must flip.

For each minimal-pair sentence: (1) auto tones, (2) fully inverted tones.
Measure per-mora mean F0 in both renders; report flip rate (fraction of morae
whose H/L order follows the FORCED tones) + run our scorer on both.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.tts_model import TTSModel

from tts_spike import BERT, CONFIG, MODEL, STYLE

CASES = ["箸が。", "橋が。", "雨が。", "飴が。"]


def load_model():
    import torch

    bert_models.load_model(Languages.JP, str(BERT))
    bert_models.load_tokenizer(Languages.JP, str(BERT))
    for _m in bert_models.__loaded_models.values():
        _m.float()
    model = TTSModel(model_path=MODEL, config_path=CONFIG,
                     style_vec_path=np.load(STYLE),
                     device="cuda" if torch.cuda.is_available() else "cpu")
    model.load()
    net = getattr(model, "_TTSModel__net_g", None)
    if net is not None:
        net.float()
    return model


def phone_f0(wav, sr, n_phones):
    """Mean F0 per phone slot (equal splits of voiced frames; crude but
    directionally adequate for a control test)."""
    import numpy as np

    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, 16000)
        wav = resample_poly(wav.astype(np.float32), 16000 // g, sr // g)
        sr = 16000
    import sys
    sys.path.insert(0, "src")
    from akusento.labels.f0 import extract_f0

    r = extract_f0(wav.astype(np.float32), sr)
    f0 = r["f0"]
    idx = np.where(f0 > 0)[0]
    if len(idx) < n_phones * 3:
        return [float("nan")] * n_phones
    med = float(np.median(f0[idx]))
    return [float(12 * np.log2(f0[c].mean() / med)) if len(c) else float("nan")
            for c in np.array_split(idx, n_phones)]


def main():
    import json

    model = load_model()
    out = {}
    for text in CASES:
        phones, auto, _ = g2p(text)
        inv = [1 - x for x in auto]
        outs = {}
        for tag, tones in (("auto", None), ("forced_auto", auto), ("forced_inv", inv)):
            kw = {} if tones is None else {"given_phone": phones, "given_tone": tones}
            sr, wav = model.infer(text=text, line_split=False, **kw)
            p = Path(f"results/swap_{tag}_{len(out)}.wav")
            wavfile.write(p, sr, wav)
            outs[tag] = (sr, wav)
        # Robust metric: correlation between forced-tone delta and F0 delta
        # across the utterance (positive = output follows forced tones).
        fa = phone_f0(outs["forced_auto"][1], outs["forced_auto"][0], len(phones))
        fi = phone_f0(outs["forced_inv"][1], outs["forced_inv"][0], len(phones))
        d = np.array(fi) - np.array(fa)
        t = np.array(inv) - np.array(auto)
        mask = ~np.isnan(d)
        corr = float(np.corrcoef(t[mask], d[mask])[0, 1]) if mask.sum() > 2 else None
        out[text] = {"auto_tones": auto, "corr_forced_vs_f0": corr,
                     "mean_abs_f0_shift_st": float(np.abs(d[mask]).mean()) if mask.any() else None}
        print(text, "corr:", corr)
    Path("results/minimal_tone_swap.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote results/minimal_tone_swap.json")


if __name__ == "__main__":
    main()
