"""M2: train + evaluate the mora H/L scorer.

Dataset: labels_500.jsonl (OpenJTalk H/L) x jsut-lab alignments x F0.
Split: first 400 utterances train, last 100 test (fixed order, documented).
Metrics: mora accuracy, word accent-type accuracy (exact pattern match per
phrase), majority baseline. Writes results/scorer.json + models/scorer.pkl
(models/ is gitignored; weights ship openly per licences.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from akusento.align.julius import mora_segments, parse_lab
from akusento.labels.f0 import extract_f0, load_wav_mono
from akusento.scorer.features import mora_features
from akusento.scorer.train import accuracy, save, train

LAB_DIR = Path("data/jsut-lab/basic5000/lab")
WAV_DIR = Path("data/jsut-basic5000/basic5000")


def build_rows(utt_ids: list[str], labels: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, info = [], [], []
    for uid in utt_ids:
        lab = parse_lab(LAB_DIR / f"{uid}.lab")
        segs = mora_segments(lab)
        phrases = labels[uid]
        flat = [(p["pattern"], m) for p in phrases for m in p["morae"]]
        if len(flat) != len(segs):
            continue
        y_wav, _ = load_wav_mono(str(WAV_DIR / f"{uid}.wav"))
        r = extract_f0(y_wav, 16000)
        f0, t = r["f0"], r["t"]
        med = float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0
        pos = 0
        for p in phrases:
            for j, m in enumerate(p["morae"]):
                s = segs[pos]
                pos += 1
                f = mora_features(f0, t, s["start"], s["end"], med, j + 1, len(p["morae"]))
                if f is None:
                    continue
                X.append(f)
                y.append(1 if m["h"] else 0)
                info.append({"utt": uid, "mora": m["t"]})
    return np.array(X), np.array(y), info


def main() -> None:
    labels = {}
    for line in Path("results/labels_500.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" not in r:
            labels[r["utt_id"]] = r["phrases"]
    ids = sorted(labels)
    # Only utterances with audio + alignment on disk.
    ids = [u for u in ids if (WAV_DIR / f"{u}.wav").exists() and (LAB_DIR / f"{u}.lab").exists()]
    print(f"usable: {len(ids)} utterances")
    tr, te = ids[:400], ids[400:]
    Xtr, ytr, _ = build_rows(tr, labels)
    Xte, yte, _ = build_rows(te, labels)
    print(f"train morae: {len(ytr)} (frac H={ytr.mean():.3f}), test morae: {len(yte)}")
    base = max(yte.mean(), 1 - yte.mean())
    sc, clf = train(Xtr, ytr)
    acc = accuracy(clf, sc, Xte, yte)
    # Word accent-type accuracy: exact pattern match per phrase (harsh metric).
    pat_ok = pat_n = 0
    for uid in te:
        segs = mora_segments(parse_lab(LAB_DIR / f"{uid}.lab"))
        phrases = labels[uid]
        if sum(len(p["morae"]) for p in phrases) != len(segs):
            continue
        y_wav, _ = load_wav_mono(str(WAV_DIR / f"{uid}.wav"))
        rr = extract_f0(y_wav, 16000)
        f0, t = rr["f0"], rr["t"]
        med = float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0
        pos = 0
        for p in phrases:
            feats, hs = [], []
            for j, m in enumerate(p["morae"]):
                s = segs[pos]
                pos += 1
                f = mora_features(f0, t, s["start"], s["end"], med, j + 1, len(p["morae"]))
                if f is None:
                    break
                feats.append(f)
                hs.append(1 if m["h"] else 0)
            else:
                if not feats:
                    continue
                pat_ok += list(clf.predict(sc.transform(np.array(feats)))) == hs
                pat_n += 1
                continue
    out = {"n_train_utt": len(tr), "n_test_utt": len(te), "n_train_morae": len(ytr),
           "n_test_morae": len(yte), "baseline_majority": round(float(base), 4),
           "mora_accuracy": round(acc, 4),
           "word_accent_accuracy": round(pat_ok / max(pat_n, 1), 4),
           "word_phrases": pat_n}
    Path("results/scorer.json").write_text(json.dumps(out, indent=2))
    save("models/scorer.pkl", sc, clf, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
