<div align="center">

# 🎙️ MOSS-TTS-Nano — Persian Fine-Tune - Version 6 is running - modify the code and readme file after the result

**Bringing native Persian accent quality to a 100M-parameter TTS model**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/base%20model-MOSS--TTS--Nano--100M-blue)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)
[![Language](https://img.shields.io/badge/language-Persian%20%28Farsi%29-green)]()
[![Training Notebook](https://img.shields.io/badge/training%20notebook-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)

</div>

---

## ✨ What is this?

Fine-tuning [**OpenMOSS-Team/MOSS-TTS-Nano-100M**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) —
a lightweight, realtime, CPU-friendly TTS model — to sound genuinely **Persian**, not
just "technically correct Persian pronunciation with a foreign accent."

> 🗣️ The base model supports Persian as one of 20 trained languages, but out-of-the-box
> voice-cloned output leans flat and non-native-sounding. This project closes that gap.

**📓 Full training notebook (public, runnable on Kaggle):**
[kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)
— every round's checkpoint, config, and logs are visible there under its "Versions" tab.

---

## 🎯 Motivation

| Before fine-tuning | After fine-tuning |
|---|---|
| Flat, robotic delivery | More natural pacing |
| Non-native accent on cloned voices | Closer to native Persian accent |
| Cuts off mid-sentence on multi-sentence input | Addressed via concatenated training examples |

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
 │                           │  (targets the mid-sentence cutoff bug)
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
cell at the top, so each round builds on the last without rewriting the pipeline.

---

## 📊 Status

- ✅ **Round 1** — 4,000 clips, 3 epochs, trained from the base model.
  Clear improvement in accent naturalness over the untouched base model.
  ⚠️ Found: cuts off mid-sentence on multi-sentence input — traced to the training
  data being entirely single-sentence Common Voice clips.

- ✅ **Round 2** — 25,000 clips (30% built as concatenated 2–3 sentence groups to
  address the cutoff), 3 epochs, trained from the base model. ~6h 41m on Kaggle T4×2.
  Verified via voice-cloned synthesis using a held-out reference speaker: no cutoff
  on the tested multi-sentence input, and a clear step up in naturalness over Round 1.

- ✅ **Round 3** (current best) — 34,000 clips, continuing training from Round 2's
  checkpoint (not restarting from the base model), 3 epochs, ~9h target on Kaggle T4×2.
  Listened and validated: a modest improvement over Round 2, and a clear improvement
  over the original base model.

- 📓 **Weights currently available via the public Kaggle notebook above** (see its
  Output tab for `checkpoint-last` under each round's output folder). A Hugging Face
  model repo is planned for later, for easier standard download/usage.

- 🔜 Possible future round: further fine-tuning on a new/additional dataset, time permitting.

---

## 🔊 Samples

> Round 2's output, using the exact test sentence and reference speaker described above.
> Round 3 is the current best-performing checkpoint — a Round 3 sample clip is planned
> to replace this once uploaded.

**[▶ Listen to Round 2 output](samples/round2.wav)**

| Text | Base model | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| سلام، این یک آزمایش تبدیل متن به گفتار... | *(not yet uploaded)* | *(not yet uploaded)* | [listen](samples/round2.wav) | *(not yet uploaded)* |

---

## ⚙️ Requirements

- Python 3.12, CUDA GPU (developed on Kaggle's free T4×2)
- `ffmpeg` on PATH
- See `MOSS-TTS-Nano/requirements.txt` (fetched automatically)

## 🚀 Usage

**Option 1 — use the weights directly:** download `checkpoint-last` from Round 3's
output folder in the [public Kaggle notebook](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)
(Output tab), and load it with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`
plus the matching `MOSS-Audio-Tokenizer-Nano` codec.

**Option 2 — reproduce or extend training:**
```bash
# On Kaggle (recommended — free GPU + fast dataset access):
# 1. Attach dataset: amirftma/common-voice-fa-v13
# 2. Edit the CONFIG cell at the top of finetune.ipynb
#    (point BASE_MODEL_PATH at a prior checkpoint to continue training it,
#    or at the base model to start fresh)
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