# Akusento — decisions (append-only, newest first)

## 2026-10-10 — M3 tone control (base jvnv-F1-jp, NO fine-tune needed)
- infer() takes given_phone/given_tone (per-phone 0/1) — explicit accent
  control exists in the BASE model. M3 pivoted: validate control + eval,
  no JSUT fine-tune (avoids the JSUT-weight release problem entirely).
- Traps found: (1) line_split=True (default) SILENTLY IGNORES given_tone —
  first experiment showed corr~0 for this reason; always line_split=False
  with explicit tones. (2) numpy 2.x breaks pyopenjtalk ABI when torch/librosa
  load first — main venv pinned numpy<2/scipy<1.14/matplotlib 3.9.4, no torch/
  librosa/sbv2; TTS lives in .venv-tts (documented in pyproject).
  (3) DeBERTa loads fp16 while TTS net is fp32 — .float() both after load.
- Cross-swap (same phones, swapped tones): output follows TONES (雨/飴
  0.82/0.92 contour corr; 箸/橋 inconclusive 0.80-0.91 all-pairs).
  Aggregate directional: auto 92/35 agree (72%), forced 75/53 (59%) vs
  auto-baseline 66/57 (54%) — control is PARTIAL (BERT semantics compete).
- 22/43 groups usable (21 dropped: same tones or reading mismatches like
  生地=きじ — the drop logic works as designed).
- Licence: SBV2 code AGPL-3.0 (pip dependency only, not vendored);
  default jvnv models inherit CC BY-SA 4.0 (JVNV corpus). Generated eval
  audio shippable with attribution; no corpus audio redistributed.
