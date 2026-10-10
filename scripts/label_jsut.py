"""M1: mora-level accent labels (+F0 stats) for downloaded JSUT utterances.

Reads data/jsut-basic5000/basic5000/{transcript_utf8.txt,wav/*.wav}.
Writes results/labels_500.jsonl (one row per utterance, committed).
Failures (pron/label mismatch) are recorded, never coerced.
"""

from __future__ import annotations

import json
from pathlib import Path

from akusento.labels.accent import extract
from akusento.labels.f0 import extract_f0, load_wav_mono


def main(n: int = 500) -> None:
    data = Path("data/jsut-basic5000/basic5000")
    transcripts = {}
    for line in (data / "transcript_utf8.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            transcripts[k.strip()] = v.strip()
    ids = sorted(transcripts)[:n]
    out = Path("results/labels_500.jsonl")
    ok = fail = 0
    with open(out, "w", encoding="utf-8") as f:
        for uid in ids:
            text = transcripts[uid]
            row: dict = {"utt_id": uid, "text": text}
            try:
                phrases = extract(text)
                row["phrases"] = [
                    {"pattern": p.pattern, "nucleus": p.nucleus,
                     "morae": [{"t": m.text, "h": m.high} for m in p.morae]}
                    for p in phrases
                ]
                wav = data / f"{uid}.wav"
                if wav.exists():
                    y, sr = load_wav_mono(str(wav))
                    r = extract_f0(y, sr)
                    row["f0_voiced_rate"] = round(r["voiced_rate"], 3)
                    row["f0_mean"] = round(r["voiced_mean"], 1)
                ok += 1
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                fail += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labeled {ok}/{len(ids)} ok, {fail} failed -> {out}")


if __name__ == "__main__":
    main()
