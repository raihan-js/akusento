"""Acoustic cross-check of OpenJTalk labels (fills heard_ok HONESTLY).

No fluent listener is available (owner disclosed not a speaker; agent cannot
hear). Instead: for each adjacent mora pair inside a phrase, compare the
PREDICTED direction (from H/L labels) with the MEASURED F0 direction
(semitones rel. utterance median, 0.5st deadband):
- L->H predicted, measured up: agree; measured down: CONTRADICT; else unclear.
- H->L predicted, measured down: agree; measured up: CONTRADICT; else unclear.
- level pairs (H-H, L-L): informational (downdrift makes them unreliable).
heard_ok = 1 iff zero contradictions and >=1 agreement (undershoot is not
contradiction: phonological targets undershoot routinely in running speech).
This measures label-vs-acoustics CONTRADICTION rate, NOT human correctness.
results/acoustic_check.jsonl holds per-transition data.
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
    out_trans = []
    pos = 0
    agree = contra = unclear = 0
    for p in phrases:
        n = len(p["morae"])
        means = flat_means[pos:pos + n]
        pos += n
        for i in range(n - 1):
            a, b = means[i], means[i + 1]
            la = p["morae"][i]["h"]
            lb = p["morae"][i + 1]["h"]
            if a is None or b is None:
                unclear += 1
                continue
            d = b - a
            if la == lb:
                out_trans.append({"pair": (p["morae"][i]["t"], p["morae"][i + 1]["t"]),
                                  "pred": "level", "diff_st": round(d, 2)})
                continue
            expect_up = (not la and lb)
            if abs(d) <= ST:
                unclear += 1
                status = "unclear"
            elif (d > 0) == expect_up:
                agree += 1
                status = "agree"
            else:
                contra += 1
                status = "CONTRADICT"
            out_trans.append({"pair": (p["morae"][i]["t"], p["morae"][i + 1]["t"]),
                              "pred": "up" if expect_up else "down",
                              "diff_st": round(d, 2), "status": status})
    verdict = bool(contra == 0 and agree >= 1)
    return {"utt_id": utt_id, "transitions": out_trans, "agree": agree,
            "contradict": contra, "unclear": unclear, "verdict": verdict}


def main(csv_path: str = "results/fluent_check.csv") -> None:
    from akusento.labels.accent import extract_from_lab

    labels = {}
    for line in Path("results/labels_500.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        labels[r["utt_id"]] = r.get("phrases", [])
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = 0
    with open("results/acoustic_check.jsonl", "w", encoding="utf-8") as jf:
        for r in rows:
            uid = r["utt_id"]
            if not (WAV_DIR / f"{uid}.wav").exists() or not (LAB_DIR / f"{uid}.lab").exists():
                r["note"] = "missing audio/alignment"
                continue
            try:
                # Train/eval only on cross-version-agreed utterances: file
                # labels (2017 OpenJTalk, era-consistent with audio) vs current
                # pyopenjtalk must give identical patterns, else skip honestly.
                file_pats = [p.pattern for p in extract_from_lab(str(LAB_DIR / f"{uid}.lab"))]
                mine_pats = [p["pattern"] for p in labels.get(uid, [])]
                if file_pats != mine_pats:
                    r["note"] = "version drift (label sets disagree) — skipped"
                    continue
                res = check_utterance(uid, labels.get(uid, []))
            except Exception as e:  # noqa: BLE001
                r["note"] = f"error: {type(e).__name__}"
                continue
            jf.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res["agree"] + res["contradict"] == 0:
                r["note"] = "no checkable transitions"
                continue
            r["heard_ok"] = "1" if res["verdict"] else "0"
            r["note"] = (f"acoustic: {res['agree']} agree, {res['contradict']} contradict, "
                         f"{res['unclear']} unclear (NOT a human check)")
            filled += 1
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "predicted", "wav", "heard_ok", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"filled {filled}/{len(rows)} rows (acoustic cross-check, disclosed)")


if __name__ == "__main__":
    main()
