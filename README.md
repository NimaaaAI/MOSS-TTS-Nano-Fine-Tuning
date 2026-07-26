<div align="center">

# 🎙️ MOSS-TTS-Nano — Persian Fine-Tune

**Bringing native Persian accent quality to a 100M-parameter TTS model**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/base%20model-MOSS--TTS--Nano--100M-blue)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)
[![Language](https://img.shields.io/badge/language-Persian%20%28Farsi%29-green)]()
[![Made with Kaggle](https://img.shields.io/badge/trained%20on-Kaggle%20T4×2-20BEFF?logo=kaggle)](https://kaggle.com)

</div>

---

## ✨ What is this?

Fine-tuning [**OpenMOSS-Team/MOSS-TTS-Nano-100M**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) —
a lightweight, realtime, CPU-friendly TTS model — to sound genuinely **Persian**, not
just "technically correct Persian pronunciation with a foreign accent."

> 🗣️ The base model supports Persian as one of 20 trained languages, but out-of-the-box
> voice-cloned output leans flat and non-native-sounding. This project closes that gap.

---

## 🎯 Motivation

| Before fine-tuning | After fine-tuning |
|---|---|
| Flat, robotic delivery | More natural pacing |
| Non-native accent on cloned voices | Closer to native Persian accent |
| Cuts off mid-sentence on multi-sentence input | Fixed via concatenated training examples |

---

## 🧩 Pipeline

```
Common Voice Persian (CC0)
        │
        ▼
 quality filter (up-votes, sentence length)
        │
        ▼
 ┌─────────────┴─────────────┐
 │                           │
single-sentence clips   concatenated 2–3 clip groups
 │                           │  (fixes mid-sentence cutoff bug)
 └─────────────┬─────────────┘
               ▼
      convert → 24kHz WAV
               ▼
   encode → audio_codes (MOSS-Audio-Tokenizer-Nano)
               ▼
        fine-tune (accelerate + DDP)
               ▼
     verify via voice-cloned synthesis
```

All of this lives in one notebook: **`finetune.ipynb`** — every knob (sample size, epochs,
learning rate, which checkpoint to resume from) is controlled from a single `CONFIG`
cell at the top.

---

## 📊 Status

- ✅ **Round 1** — 4,000 clips, 3 epochs → clear improvement over base model
- ⚠️ Found: mid-sentence cutoffs on multi-sentence input
- 🔄 **Round 2** (in progress) — 25,000 clips + concatenated multi-sentence examples, targeting the cutoff fix
- ⏳ Pending: listening validation of Round 2 before deciding whether to publish weights

---

## 🔊 Samples

| Text | Base model | Fine-tuned |
|---|---|---|
| سلام، این یک آزمایش تبدیل متن به گفتار... | [listen](samples/base.wav) | [listen](samples/finetuned.wav) |

*(GitHub can't play audio inline in a README — click through to download/play.)*

---

## ⚙️ Requirements

- Python 3.12, CUDA GPU (developed on Kaggle's free T4×2)
- `ffmpeg` on PATH
- See `MOSS-TTS-Nano/requirements.txt` (fetched automatically)

## 🚀 Usage

```bash
# On Kaggle (recommended — free GPU + fast dataset access):
# 1. Attach dataset: amirftma/common-voice-fa-v13
# 2. Edit the CONFIG cell at the top of finetune.ipynb
# 3. Save Version → Save & Run All (Commit) for unattended runs
```

## 📄 Attribution & Licensing

This repo contains **only original pipeline code** (MIT, see `LICENSE`). No third-party
weights or datasets are redistributed here.

- **Base model**: [MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) + [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) — Apache-2.0, © OpenMOSS Team. Fine-tuned weights remain under this license.
- **Training data**: [Mozilla Common Voice — Persian](https://commonvoice.mozilla.org/) — CC0 (public domain).

<div align="center">

*Not affiliated with or endorsed by the OpenMOSS team.*

</div>
