# Akusento — licences (verified 2026-10-10)

## JSUT (single speaker, ~10h)
Source: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- Text: CC-BY-SA 4.0 etc. (see corpus LICENCE file at build).
- Audio: research by academic institutions; **non-commercial research
  (incl. inside commercial orgs)**; personal use incl. blog posts.
- **Re-distribution forbidden** (may upload ~100 files to a blog).
- Commercial use: contact Todai TLO (nakahara@todaitlo.jp) + Takamichi.
- Our use: independent non-commercial research — training locally is covered;
  public weight release is NOT clearly covered.

## JVS (100 speakers, ~30h, 24kHz studio)
Source: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
Paper: Takamichi et al. 2019, arXiv 1908.06248 (verified).
- Text: from JSUT (licences as above). Tags: CC-BY-SA 4.0.
- Audio: same terms as JSUT (~10 files may be uploaded).
- Ships phoneme alignments (verify at download) —used for M1 if present.
- Same release posture as JSUT.

## Tooling
- pyopenjtalk / OpenJTalk: Apache-2.0 (verify at build).
- pyworld: MIT (verify at build). librosa: ISC. scikit-learn: BSD.
- TTS base (to be chosen M3): prefer Apache-2.0/MIT bases; the JSUT/JVS
  question attaches to the TRAINING DATA regardless of base licence.
- Kokoro-82M (Apache-2.0) is the comparison default.

## Release plan (from the above)
- OPEN: scorer code + weights (no voice reproduction possible), eval code +
  numbers, article, demo UI code.
- LOCAL-ONLY: TTS fine-tuned weights + generated audio samples of the corpus
  speaker. Public TTS release only with written TLO permission (draft email
  queued in docs/).
