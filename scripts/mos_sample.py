"""M3 MOS-naturalness sample: 20 varied renders for owner rating (1-5).

Naturalness needs no Japanese — pitch-accent correctness is measured
separately by the scorer. results/mos_sample.csv: file, text, rating, note.
"""

SENTENCES = [
    "箸で橋を渡る。",
    "雨が降る中、飴を舐めた。",
    "今日は良い天気ですね。",
    "神楽坂をあとにし、次の選挙のため、全国を遊説して回る。",
    "東京都に行きます。",
    "この料理はとても美味しいです。",
    "明日は会議があります。",
    "彼女の声はとても綺麗だ。",
    "新幹線は速い乗り物だ。",
    "日本語の発音は難しい。",
    "彼は毎朝ジョギングをする。",
    "この本は面白かった。",
    "映画を見に行こう。",
    "音楽を聴くのが好きだ。",
    "友達と旅行に行った。",
    "美味しい寿司が食べたい。",
    "富士山は美しい山だ。",
    "桜が満開になった。",
    "夏休みに海へ行く。",
    "おはようございます。",
]


def main():
    import csv
    from pathlib import Path

    from tone_swap import load_model

    model = load_model()
    rows = []
    for i, text in enumerate(SENTENCES):
        sr, wav = model.infer(text=text, line_split=False)
        p = Path(f"results/mos_{i:02d}.wav")
        from scipy.io import wavfile

        wavfile.write(p, sr, wav)
        rows.append({"file": str(p), "text": text, "rating": "", "note": ""})
    with open("results/mos_sample.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "rating", "note"])
        w.writeheader()
        w.writerows(rows)
    print("wrote 20 MOS samples + results/mos_sample.csv")


if __name__ == "__main__":
    main()
