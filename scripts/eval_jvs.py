"""JVS held-out-speaker eval (the idea's headline metric).

JSUT-trained scorer (models/scorer.pkl, single female speaker) evaluated on
JVS parallel100 utterances from unseen speakers (9568 mixed in step 2).
Labels from JVS .lab files (same grouping code); F0 from 24kHz audio.
Writes results/jvs_eval.json: per-speaker mora accuracy + overall.
"""

from pathlib import Path

import numpy as np

JVS = Path("data/jvs_ver1")
SPEAKERS = ["jvs001", "jvs002", "jvs003", "jvs004", "jvs005",
            "jvs007", "jvs008", "jvs009", "jvs010", "jvs011"]  # jvs006 ships empty lab/ful


def main():
    import json
    import pickle
    import sys
    sys.path.insert(0, "src")
    from akusento.align.julius import mora_segments, parse_lab
    from akusento.labels.accent import accent_phrase
    from akusento.labels.f0 import extract_f0, load_wav_mono
    from akusento.scorer.features import phrase_feature_matrix

    d = pickle.load(open("models/scorer.pkl", "rb"))
    sc, clf = d["scaler"], d["clf"]
    per_spk = {}
    for spk in SPEAKERS:
        labdir = JVS / spk / "parallel100" / "lab" / "ful"
        wavdir = JVS / spk / "parallel100" / "wav24kHz16bit"
        if not labdir.exists():
            print(spk, "missing, skip")
            continue
        ok = tot = 0
        for labf in sorted(labdir.glob("*.lab")):
            uid = labf.stem
            wav = wavdir / f"{uid}.wav"
            if not wav.exists():
                continue
            rows = parse_lab(labf)
            # Phrase segmentation shared with training path.
            from akusento.labels.accent import phrases_from_rows

            triples = [(r["phon"], r["a"], r["fid"]) for r in rows]
            try:
                groups = phrases_from_rows(triples)
            except Exception:  # noqa: BLE001
                continue
            y, _ = load_wav_mono(str(wav))
            r = extract_f0(y, 16000)
            f0, t = r["f0"], r["t"]
            # Segment alignment morae from the same rows.
            segs = mora_segments(rows)
            pos = 0
            for g in groups:
                mg = []
                for a in g:
                    if mg and mg[-1][0][1] == a[1]:
                        mg[-1].append(a)
                    else:
                        mg.append([a])
                ap = accent_phrase(mg, ["?"] * len(mg))
                n = len(mg)
                sl = segs[pos:pos + n]
                pos += n
                if len(sl) != n:
                    continue
                Xm, valid = phrase_feature_matrix(f0, t, sl, n)
                pred = clf.predict(sc.transform(Xm))
                for j, h in enumerate([1 if m.high else 0 for m in ap.morae]):
                    if not valid[j]:
                        continue
                    tot += 1
                    ok += pred[j] == h
        per_spk[spk] = {"acc": round(ok / max(tot, 1), 4), "n": tot}
        print(spk, per_spk[spk])
    tot_n = sum(v["n"] for v in per_spk.values())
    overall = round(sum(v["acc"] * v["n"] for v in per_spk.values()) / max(tot_n, 1), 4)
    out = {"speakers": per_spk, "overall": overall, "n": tot_n}
    Path("results/jvs_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
