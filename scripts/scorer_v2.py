"""Scorer v2 experiment: per-phrase median normalization + neighbor context.

v1 normalized semitones by the UTTERANCE median (smeared by downstep across
phrases). v2 normalizes by the PHRASE median and adds prev/next mora means
(local contour direction). Same split, same logistic setup — the comparison
is apples to apples. Writes results/scorer_v2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from akusento.align.julius import mora_segments, parse_lab
from akusento.labels.f0 import extract_f0, load_wav_mono
from akusento.scorer.features import mora_features
from akusento.scorer.train import accuracy, train

LAB_DIR = Path("data/jsut-lab/basic5000/lab")
WAV_DIR = Path("data/jsut-basic5000/basic5000")


def build_v2(utt_ids, labels):
    X, y = [], []
    for uid in utt_ids:
        lab = parse_lab(LAB_DIR / f"{uid}.lab")
        segs = mora_segments(lab)
        phrases = labels[uid]
        if sum(len(p["morae"]) for p in phrases) != len(segs):
            continue
        y_wav, _ = load_wav_mono(str(WAV_DIR / f"{uid}.wav"))
        r = extract_f0(y_wav, 16000)
        f0, t = r["f0"], r["t"]
        pos = 0
        for p in phrases:
            # Phrase-local median (downstep-robust).
            pm = []
            for m in p["morae"]:
                s = segs[pos]
                pos += 1
                msk = (t >= s["start"]) & (t < s["end"]) & (f0 > 0)
                pm.append(float(np.median(f0[msk])) if msk.sum() >= 3 else None)
            ok = [v for v in pm if v is not None]
            if not ok:
                continue
            pmed = float(np.median(ok))
            for j, m in enumerate(p["morae"]):
                s = segs[pos - len(p["morae"]) + j]
                f = mora_features(f0, t, s["start"], s["end"], pmed, j + 1, len(p["morae"]))
                if f is None:
                    continue
                prevm = pm[j - 1] if j > 0 and pm[j - 1] is not None else pmed
                nextm = pm[j + 1] if j + 1 < len(pm) and pm[j + 1] is not None else pmed
                cur = pm[j] if pm[j] is not None else pmed
                f = f + [float(12 * np.log2(cur / pmed)), float(12 * np.log2(prevm / pmed)),
                         float(12 * np.log2(nextm / pmed))]
                X.append(f)
                y.append(1 if m["h"] else 0)
    return np.array(X), np.array(y)


def main():
    labels = {}
    for line in Path("results/labels_500.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" not in r:
            labels[r["utt_id"]] = r["phrases"]
    ids = sorted(labels)
    avail = [u for u in ids if (WAV_DIR / f"{u}.wav").exists() and (LAB_DIR / f"{u}.lab").exists()]
    tr, te = avail[:400], avail[400:]
    Xtr, ytr = build_v2(tr, labels)
    Xte, yte = build_v2(te, labels)
    base = max(yte.mean(), 1 - yte.mean())
    sc, clf = train(Xtr, ytr)
    acc = accuracy(clf, sc, Xte, yte)
    out = {"n_train": len(ytr), "n_test": len(yte), "baseline": round(float(base), 4),
           "v2_accuracy": round(acc, 4), "v1_accuracy": 0.7361}
    Path("results/scorer_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
