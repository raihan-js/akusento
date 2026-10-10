"""Akusento Gradio demo (runs in .venv-tts).

Tab 1 — hear the difference: type a word, hear it with the dictionary accent
  and with the tones flipped (typical learner mistake).
Tab 2 — check yourself: type what you said, record it, see the expected
  pattern vs the heard pattern from the scorer. Single-word/short recordings
  only: segments are equal splits (disclosed approximation, no aligner for
  arbitrary audio).
"""

from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent


def _load_scorer():
    import pickle

    with open(APP_DIR / "models" / "scorer.pkl", "rb") as f:
        d = pickle.load(f)
    return d["scaler"], d["clf"]


def _load_tts():
    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    base = APP_DIR / "data"
    bert = base / "sbv2-bert" / "deberta-v2-large-japanese-char-wwm"
    bert_models.load_model(Languages.JP, str(bert))
    bert_models.load_tokenizer(Languages.JP, str(bert))
    for _m in bert_models.__loaded_models.values():
        _m.float()
    mp = base / "sbv2" / "jvnv-F1-jp"
    model = TTSModel(model_path=mp / "jvnv-F1-jp_e160_s14000.safetensors",
                     config_path=mp / "config.json",
                     style_vec_path=np.load(mp / "style_vectors.npy"),
                     device="cuda" if torch.cuda.is_available() else "cpu")
    model.load()
    net = getattr(model, "_TTSModel__net_g", None)
    if net is not None:
        net.float()
    return model


def hear_pair(word):
    """Render (correct, flipped) wavs for a word. Returns (sr, wav) pairs + patterns."""
    import sys

    sys.path.insert(0, str(APP_DIR / "src"))
    from style_bert_vits2.nlp.japanese.g2p import g2p

    from akusento.labels.accent import extract

    model = hear_pair._model if hasattr(hear_pair, "_model") else setattr(hear_pair, "_model", _load_tts()) or hear_pair._model
    text = word.strip() + "。"
    phrases = extract(text)
    pat = " ".join(p.pattern for p in phrases)
    phones, tones, _ = g2p(text)
    sr, wav_ok = model.infer(text=text, line_split=False)
    flipped = [1 - t if p != "_" else t for p, t in zip(phones, tones)]
    sr2, wav_bad = model.infer(text=text, line_split=False, given_phone=phones, given_tone=flipped)
    return (sr, wav_ok), (sr2, wav_bad), pat


def hear_pair_ui(word):
    (sr, ok), (_, bad), pat = hear_pair(word)
    return (sr, ok), (sr, bad), pat


def check_recording(word, audio):
    """Score a user recording against the expected pattern (equal-split segments)."""
    import sys

    sys.path.insert(0, str(APP_DIR / "src"))
    from akusento.labels.accent import extract
    from akusento.labels.f0 import extract_f0
    from akusento.scorer.features import mora_features

    if not word or not word.strip() or audio is None:
        return "type a word and record audio first", ""
    sc, clf = _load_scorer()
    phrases = extract(word.strip() + "。")
    exp = " ".join(p.pattern for p in phrases)
    sr, wav = audio
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    r = extract_f0(wav, sr)
    f0, t = r["f0"], r["t"]
    idx = np.where(f0 > 0)[0]
    if len(idx) < 30:
        return exp, "too little voicing detected"
    med = float(np.median(f0[idx]))
    heard_bits = []
    pos = 0
    nmorae = sum(len(p.morae) for p in phrases)
    bounds = np.array_split(idx, nmorae)
    for p in phrases:
        for j, m in enumerate(p.morae):
            ch = bounds[pos]
            pos += 1
            s, e = (t[ch[0]], t[ch[-1]]) if len(ch) else (0, 0)
            f = mora_features(f0, t, s, e, med, j + 1, len(p.morae))
            if f is None:
                heard_bits.append("?")
            else:
                heard_bits.append("H" if clf.predict(sc.transform([f]))[0] else "L")
    heard = "".join(heard_bits)
    expflat = "".join(p.pattern for p in phrases)
    mark = "MATCH ✓" if heard.replace("?", "") and all(
        h == e for h, e in zip(heard, expflat) if h != "?") else "differs — see ?/mismatches"
    return exp, f"heard {heard} — {mark} (equal-split segments, approximate)"


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Akusento — Japanese pitch accent") as ui:
        gr.Markdown("# Akusento — pitch accent you can hear\n"
                    "Dictionary accents (OpenJTalk, Tokyo dialect — not human-checked) "
                    "plus a scorer (74% mora accuracy) trained on JSUT read speech.")
        with gr.Tab("Hear the difference"):
            w = gr.Textbox(label="Word (e.g. 箸, 橋, 雨, 飴)", value="箸")
            btn = gr.Button("Synthesize")
            a_ok = gr.Audio(label="Dictionary accent")
            a_bad = gr.Audio(label="Flipped accent (typical mistake)")
            pat = gr.Textbox(label="Expected pattern")
            btn.click(hear_pair_ui, inputs=w, outputs=[a_ok, a_bad, pat])
        with gr.Tab("Check yourself"):
            w2 = gr.Textbox(label="What did you say?")
            mic = gr.Audio(label="Record", sources=["microphone"], type="numpy")
            btn2 = gr.Button("Score")
            exp = gr.Textbox(label="Expected")
            heard = gr.Textbox(label="Heard")
            btn2.click(check_recording, inputs=[w2, mic], outputs=[exp, heard])
    return ui


if __name__ == "__main__":
    build_ui().launch()
