"""M3 spike: given_tone accent control on minimal pairs (base model, no training)."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.tts_model import TTSModel

MODEL = Path("data/sbv2/jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors")
CONFIG = Path("data/sbv2/jvnv-F1-jp/config.json")
STYLE = Path("data/sbv2/jvnv-F1-jp/style_vectors.npy")
BERT = Path("data/sbv2-bert/deberta-v2-large-japanese-char-wwm")


def synth(text, tones=None, tag="out", style="Neutral"):
    import torch
    from style_bert_vits2.nlp import bert_models

    bert_models.load_model(Languages.JP, str(BERT))
    bert_models.load_tokenizer(Languages.JP, str(BERT))
    # DeBERTa checkpoint is fp16; the TTS net is fp32 — unify (documented).
    for _m in bert_models.__loaded_models.values():
        _m.float()
    model = TTSModel(
        model_path=MODEL, config_path=CONFIG, style_vec_path=np.load(STYLE),
        device="cuda" if torch.cuda.is_available() else "cpu")
    model.load()
    # Workaround (upstream pins torch<2.4): the safetensors checkpoint is fp16
    # while the net inits fp32; unify to fp32 for inference on modern torch.
    net = getattr(model, "_TTSModel__net_g", None)
    if net is not None:
        net.float()
    kw = {}
    if tones is not None:
        phones, _, _ = g2p(text)
        assert len(phones) == len(tones), (len(phones), len(tones))
        kw = {"given_phone": phones, "given_tone": tones}
    sr, wav = model.infer(text=text, style=style, line_split=False, **kw)
    out = Path(f"results/spike_{tag}.wav")
    wavfile.write(out, sr, wav)
    print(f"{tag}: sr={sr} len={len(wav)/sr:.1f}s auto_tones={g2p(text)[1] if tones is None else tones} -> {out}")
    return out


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "箸で橋を渡る。"
    synth(text, tag="auto")
