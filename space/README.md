---
title: Akusento — Japanese pitch accent you can hear
emoji: 🗣️
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Akusento demo

Type a Japanese word: hear it with the dictionary accent and with the tones
flipped (typical learner mistake). Or record yourself and see the expected vs
heard pitch pattern.

- Scorer: logistic regression on F0 (mora accuracy 0.789) — [model](https://huggingface.co/raihan-js/akusento-scorer).
- Voice: Style-Bert-VITS2 jvnv-F1-jp (CC BY-SA 4.0, JVNV corpus) with explicit tone control.
- Accents: OpenJTalk Tokyo dialect — not human-checked (disclosed in-app).
- Code: [github.com/raihan-js/akusento](https://github.com/raihan-js/akusento).

First boot downloads models (~3 GB) and takes a few minutes; synthesis runs on CPU (~1–3 min per utterance, queued).
