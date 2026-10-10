"""Acoustic cross-check of OpenJTalk labels (fills heard_ok HONESTLY).

No fluent listener is available (owner disclosed not a speaker; agent cannot
hear). Instead: for each mora get mean F0 from aligned audio, convert to
semitones rel. utterance median, and check predicted H/L directions:
- fall: F0(nucleus) > F0(nucleus+1)   (phrases with nucleus)
- rise: F0(mora2) > F0(mora1)         (multi-mora phrases)
- level: |F0 diffs| small across heiban tail (informational only)
Morae with <30% voiced frames or |diff|<=0.5st are 'unclear' (excluded).
heard_ok = 1 iff every CHECKABLE phrase passes; note records the counts.
This measures label-vs-acoustics agreement, NOT human correctness. The
write-up states exactly that. results/acoustic_check.jsonl holds per-phrase data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from akusento.align.julius import mora_segments, parse_lab
from akusento.labels.f0 import extract_f0, load_wav_mono

LAB_DIR = Path("data/jsut-lab/basic5000/lab")
WAV_DIR = Path("data/jsut-basic5000/basic5000")
ST = 0.5  # semitone evidence threshold


def check_utterance(utt_id: str, phrases: list[dict]) -> dict:
    lab = parse_lab(LAB_DIR / f"{utt_id}.lab")
    segs = mora_segments(lab)
    y, _ = load_wav_mono(str(WAV_DIR / f"{utt_id}.wav"))
    r = extract_f0(y, 16000)
    f0, t = r["f0"], r["t"]
    med = float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 150.0
    # Map label morae to aligned segments in order (same OpenJTalk source;
    # totals must match or the join is wrong — refuse silently passing).
    n_lab = sum(len(p["morae"]) for p in phrases)
    if n_lab != len(segs):
        raise ValueError(f"mora count mismatch: labels={n_lab} aligned={len(segs)}")
    flat_means = []
    for s in segs:
        m = (t >= s["start"]) & (t < s["end"]) & (f0 > 0)
        frames = int((t < s["end"]).sum() - (t < s["start"]).sum())
        if m.sum() >= 3 and m.sum() / max(frames, 1) >= 0.3:
            flat_means.append(float(12 * np.log2(f0[m].mean() / med)))
        else:
            flat_means.append(None)
    out_phrases = []
    pos = 0
    npass = ncheck = 0
    for p in phrases:
        n = len(p["morae"])
        means = flat_means[pos:pos + n]
        pos += n
        checks = []
        def get(i: int, _means: list = means, _n: int = n) -> float | None:
            return _means[i] if 0 <= i < _n else None
        if p["nucleus"] is not None and n > 1:
            a, b = get(p["nucleus"] - 1), get(p["nucleus"])
            if a is not None and b is not None and abs(a - b) > ST:
                checks.append(("fall", a > b))
        if n > 1:
            a, b = get(0), get(1)
            if a is not None and b is not None and abs(a - b) > ST:
                checks.append(("rise", b > a))
        ok = [c for _, c in checks]
        if ok:
            ncheck += 1
            npass += all(ok)
        out_phrases.append({"pattern": p["pattern"], "checks": checks})
    return {"utt_id": utt_id, "phrases": out_phrases,
            "checked": ncheck, "passed": npass,
            "verdict": bool(ncheck and npass == ncheck)}


def main(csv_path: str = "results/fluent_check.csv") -> None:
    labels = {}
    for line in Path("results/labels_500.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        labels[r["utt_id"]] = r.get("phrases", [])
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    filled = 0
    with open("results/acoustic_check.jsonl", "w", encoding="utf-8") as jf:
        for r in rows:
            uid = r["utt_id"]
            if not (WAV_DIR / f"{uid}.wav").exists() or not (LAB_DIR / f"{uid}.lab").exists():
                r["note"] = "missing audio/alignment"
                continue
            try:
                res = check_utterance(uid, labels.get(uid, []))
            except Exception as e:  # noqa: BLE001
                r["note"] = f"error: {type(e).__name__}"
                continue
            jf.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res["checked"] == 0:
                r["note"] = "no checkable phrases"
                continue
            r["heard_ok"] = "1" if res["verdict"] else "0"
            r["note"] = f"acoustic {res['passed']}/{res['checked']} phrases agree (NOT a human check)"
            filled += 1
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "predicted", "wav", "heard_ok", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"filled {filled}/{len(rows)} rows (acoustic cross-check, disclosed)")


if __name__ == "__main__":
    main()
