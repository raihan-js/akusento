"""M3 eval v2: phone-level directional agreement (no alignment needed).

For each render: equal-split voiced frames into len(phones) chunks (same
chunking for auto and swapped renders, so chunking bias cancels in the
control comparison). For adjacent phone pairs with a predicted direction
(from tone sequence), compare with measured F0 direction (0.5st deadband).
- auto correctness: agreement with AUTO (OpenJTalk) tones.
- control: agreement of SWAPPED renders with FORCED tones (vs with auto).
Writes results/tts_eval.json.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from style_bert_vits2.nlp.japanese.g2p import g2p
from tone_swap import load_model, phone_f0
from minimal_pairs import GROUPS


def dir_agree(tones, f0):
    ag = co = un = 0
    for i in range(len(tones) - 1):
        a, b = tones[i], tones[i + 1]
        x, y = f0[i], f0[i + 1]
        if a == b or x != x or y != y:
            continue
        d = y - x
        if abs(d) <= 0.5:
            un += 1
        elif (d > 0) == (b > a):
            ag += 1
        else:
            co += 1
    return ag, co, un


def main():
    import json

    model = load_model()
    auto = [0, 0, 0]
    ctrl_forced = [0, 0, 0]
    ctrl_auto = [0, 0, 0]
    kept = dropped = 0
    for reading, words in GROUPS:
        seqs = []
        for w in words:
            s = f"{w}がいい。"
            phones, tones, _ = g2p(s)
            seqs.append((w, s, phones, tones))
        if len({tuple(p) for _, _, p, _ in seqs}) != 1:
            dropped += 1
            continue
        if len({tuple(t) for _, _, _, t in seqs}) < 2:
            dropped += 1
            continue
        kept += 1
        phones = seqs[0][2]
        renders = {}
        for w, s, _, tones in seqs:
            sr, wav = model.infer(text=s, line_split=False)
            p = Path(f"results/eval_{w}.wav")
            wavfile.write(p, sr, wav)
            renders[w] = (sr, wav, tones)
        for w, s, _, tones in seqs:
            f = phone_f0(renders[w][1], renders[w][0], len(phones))
            a, c, u = dir_agree(tones, f)
            auto[0] += a
            auto[1] += c
            auto[2] += u
            others = [t for ww, _, _, t in seqs if t != tones]
            if not others:
                continue
            t2 = others[0]
            sr2, wav2 = model.infer(text=s, line_split=False,
                                    given_phone=phones, given_tone=t2)
            f2 = phone_f0(wav2, sr2, len(phones))
            a, c, u = dir_agree(t2, f2)
            ctrl_forced[0] += a
            ctrl_forced[1] += c
            ctrl_forced[2] += u
            a, c, u = dir_agree(tones, f2)
            ctrl_auto[0] += a
            ctrl_auto[1] += c
            ctrl_auto[2] += u
    out = {"groups_kept": kept, "groups_dropped": dropped,
           "auto": {"agree": auto[0], "contradict": auto[1], "unclear": auto[2]},
           "control_vs_forced": {"agree": ctrl_forced[0], "contradict": ctrl_forced[1], "unclear": ctrl_forced[2]},
           "control_vs_auto": {"agree": ctrl_auto[0], "contradict": ctrl_auto[1], "unclear": ctrl_auto[2]}}
    Path("results/tts_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
