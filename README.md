# Akusento — open Japanese pitch-accent scorer + accent-controllable TTS

Status: scaffold (2026-10-10). See `docs/decisions.md`, `docs/licences.md`.

## Licence posture (verified 2026-10-10, re-check at release)
JSUT/JVS audio: research + personal use; **re-distribution forbidden**;
commercial use via Todai TLO. Training locally = research/personal use (fine).
**Scorer** (small model on F0 statistics + OpenJTalk labels) ships openly —
it cannot reproduce anyone's voice. **TTS weights stay local** (code + evals
public); public release needs TLO permission. No real person's voice is
cloned for any demo without consent.

## Layout
`src/akusento/{labels,align,scorer,tts,eval,cli}` · `tests/` · `docs/` ·
`examples/` · `results/` (small JSON, committed) · `data/` (gitignored)
