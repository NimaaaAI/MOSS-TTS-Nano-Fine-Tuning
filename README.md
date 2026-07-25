# Persian TTS Fine-Tuning — MOSS-TTS-Nano

Fine-tuning [OpenMOSS-Team/MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano)
(a 100M-parameter text-to-speech model) for improved Persian (Farsi) speech synthesis,
using voice-cloning-conditioned training on Mozilla Common Voice Persian recordings.

## Motivation

The base MOSS-TTS-Nano model supports Persian as one of its 20 trained languages, but
out-of-the-box output leans toward a flat, non-native-sounding accent when cloning a
Persian reference voice. This project fine-tunes the model on a filtered subset of
Persian speech data to improve accent naturalness and delivery.

## What's in this repo

| File | Purpose |
|---|---|
| `finetune.ipyb` | End-to-end pipeline: dataset sampling → wav conversion → JSONL construction → training → verification |
| `README.md` | This file |

Model weights, checkpoints, and audio data are **not** included in this repo — they're
downloaded/generated at runtime by the script (see below).

## Pipeline overview

1. Sample and quality-filter clips from Common Voice Persian (up-votes, sentence length)
2. Convert a configurable number of clips to WAV
3. Build additional multi-sentence training examples by concatenating 2–3 clips —
   this specifically addresses a bug where the model would cut off mid-sentence on
   multi-sentence input after training only on single-sentence clips
4. Preprocess into `audio_codes` via the model's own audio tokenizer
5. Fine-tune via `accelerate launch` (supports single or multi-GPU)
6. Verify via voice-cloned synthesis on held-out reference clips

All parameters (sample size, epoch count, learning rate, which checkpoint to continue
from, etc.) are controlled from a single `CONFIG` block at the top of `finetune.py`.

## Requirements

- Python 3.12, CUDA-capable GPU (developed/tested on Kaggle's T4 ×2 environment)
- `ffmpeg` available on PATH
- See `MOSS-TTS-Nano/requirements.txt` (pulled automatically) for model dependencies

## Attribution & Licensing

This repository contains **only original fine-tuning/data-pipeline code**, licensed under MIT
(see `LICENSE`). It does **not** redistribute any third-party model weights or datasets.
Those are downloaded directly from their original sources at runtime:

- **Base model**: [`OpenMOSS-Team/MOSS-TTS-Nano`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano)
  and [`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) —
  used under their original license (see the model cards on Hugging Face for current terms).
  Fine-tuned weights produced by this code remain subject to that same license.
- **Training data**: [Mozilla Common Voice — Persian (fa)](https://commonvoice.mozilla.org/),
  released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).
  No attribution is legally required, but Mozilla Common Voice is credited here as good practice.

Fine-tuned model checkpoints produced by running this code are **not included in this repo**
(too large for Git) — see the notebook's "Versions → Output" for saved checkpoints, or
host them separately (e.g. a Hugging Face model repo) if you want to share them.

## Usage

```bash
# On Kaggle (recommended — free T4 GPUs, fast dataset access):
# 1. Attach the Kaggle dataset: amirftma/common-voice-fa-v13
# 2. Adjust the CONFIG block at the top of finetune.py
# 3. Save Version → Save & Run All (Commit) for long unattended runs
```

## Status

Round 1: 4,000 clips, 3 epochs — confirmed improvement over base model, but exhibited
mid-sentence cutoffs on multi-sentence input.

Round 2 (in progress): 25,000 clips including concatenated multi-sentence examples,
targeting the cutoff issue. 