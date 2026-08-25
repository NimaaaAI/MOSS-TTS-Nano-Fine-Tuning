<div align="center">

# 🎙️ MOSS-TTS-Nano Persian Fine-Tune

**Bringing native Persian accent quality to a 140M parameter TTS model**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/base%20model-MOSS--TTS--Nano-blue)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano)
[![Language](https://img.shields.io/badge/language-Persian%20%28Farsi%29-green)]()
[![Training Notebook](https://img.shields.io/badge/training%20notebook-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)

</div>

---

## ✨ What is this?

A fine tuning pipeline for [**OpenMOSS-Team/MOSS-TTS-Nano**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano),
a lightweight TTS model of roughly 140M parameters, adapted to sound genuinely **Persian**
rather than technically correct Persian with a foreign accent.

The base model lists Persian among its supported languages, but voice cloned output leans
flat and non native sounding. This project closes that gap, and documents what was learned
along the way about where the remaining limits actually come from.

**📓 Full training notebook (public, runnable on Kaggle):**
[kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)

---

## 🎯 Result

Blind A/B against the untouched base model, same reference clip, same Persian text,
same seed: the fine tuned checkpoint produces a noticeably more native Persian accent.
The base model reads Persian text with an audible English or Chinese accent, which is
consistent with what it was trained on.

The gain is in accent and pronunciation. Long form stability is a separate problem and
is documented under Known limitations below.

---

## 🧩 Pipeline

```
Common Voice Persian (CC0)
        │
        ▼
 quality filter (up votes, sentence length)
        │
        ▼
 ┌─────────────┴─────────────┐
 │                           │
single sentence clips   concatenated 2 to 3 clip groups
 │                       (same speaker only, see Findings)
 └─────────────┬─────────────┘
               ▼
      convert to 24kHz WAV
               ▼
   encode to audio_codes (MOSS-Audio-Tokenizer-Nano)
               ▼
        fine tune (accelerate + DDP, fp32 on T4×2)
               ▼
     verify via voice cloned synthesis
```

Everything lives in one notebook, `finetune.ipynb`. Sample size, epochs, learning rate,
and which checkpoint to resume from are all controlled from a single `CONFIG` cell at the
top, so each round builds on the last without rewriting the pipeline.

---

## 📊 Training rounds

| Round | Clips | Started from | Epochs | Notes |
|---|---|---|---|---|
| 1 | 4,000 | base model | 3 | First accent improvement. Revealed the cutoff problem on long input. |
| 2 | 25,000 | round 1 checkpoint | 3 | 30% concatenated groups added. ~6h41m on T4×2. |
| 3 | 34,000 | round 2 checkpoint | 3 | Modest gain over round 2. Diminishing returns from stacking rounds. |
| 4 | 34,000 | **base model** | 3 | Current best. Concatenation rebuilt per speaker. See Findings. |

Round 4 deliberately restarts from the base model rather than continuing the chain.
Rounds 2 and 3 trained repeatedly over overlapping data, so parts of the corpus had been
seen six or nine times, which showed up as diminishing returns rather than improvement.

---

## 🔬 Findings

Three things worth writing down, because each one cost a training round to discover.

**Concatenated examples were mixing speakers.** The multiple sentence groups were built by
slicing a globally shuffled dataframe, so almost every group spliced 2 or 3 different
people into one audio file paired with one transcript. Thirty percent of the training data
was teaching the model that the voice may change partway through an utterance, which is
the opposite of what a voice cloning model should learn. Fixed in round 4 by grouping on
`client_id` before concatenating. The corpus supports this easily: 102,507 same speaker
groups were available against 4,080 requested.

**The training data is short.** Measured across all 27,880 encoded examples at 12.4 frames
per second: median 48 frames (3.9s), p90 107 frames (8.6s), max 313 frames (25s), with 66%
under 5 seconds. The model stops generating around the edge of the longest thing it has
seen. `MAX_LENGTH` was never the constraint, since nothing came close to the 1024 token cap.

**Reference audio quality dominates output quality.** Same checkpoint, same text, same
seed, only the reference clip changed: a clean studio recording produced clearly better
output than a Common Voice clip did. A 140M parameter model has limited capacity to clean
up a noisy prompt, so it reproduces the room noise and microphone character along with
the voice.

---

## ⚠️ Known limitations

- **Practical utterance limit is around 5 seconds.** Beyond that, output can stop early or
  degrade. This traces directly to the training data distribution above, not to a
  configuration setting. The base model shows the same behaviour, so it is a property of
  the autoregressive decoder and the corpus rather than something introduced here.
- **Long passages need chunking.** Split the text, generate each chunk with the same
  reference and seed, then concatenate the audio.
- **Output is emotionally flat.** Common Voice is volunteers reading sentences off a
  screen. There is no emotional variation in the data, so there is none in the model.
- **Reference clip matters more than expected.** Use the cleanest single speaker recording
  available. Noisy references degrade output noticeably.

For long form Persian synthesis today, a large non autoregressive model such as
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) handles full passages in a single
call and is the better tool. This project's value is in the small model footprint and in
the documented path from a flat base model to a native sounding Persian accent.

---

## 🔊 Samples

| Text | Base model | Round 4 |
|---|---|---|
| سلام، این یک آزمایش تبدیل متن به گفتار... | *(to be uploaded)* | *(to be uploaded)* |

---

## ⚙️ Requirements

- Python 3.12, CUDA GPU (developed on Kaggle's free T4×2)
- `ffmpeg` on PATH
- See `MOSS-TTS-Nano/requirements.txt` (fetched automatically by the notebook)

Note on precision: the T4 does not support bf16, and fp16 broke gradient scaling during
training. The pipeline runs in fp32, which is slower but stable across both GPUs.

---

## 🚀 Usage

**Option 1, use the weights directly.** Download `checkpoint-last` from round 4's output
folder in the [public Kaggle notebook](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)
(Output tab), then load it with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`
alongside the matching `MOSS-Audio-Tokenizer-Nano` codec.

A Hugging Face model repo is planned so this becomes a standard one line download.

**Option 2, reproduce or extend training.**

```bash
# On Kaggle (free GPU and fast dataset access):
# 1. Attach dataset: amirftma/common-voice-fa-v13
# 2. Edit the CONFIG cell at the top of finetune.ipynb
#    (point BASE_MODEL_PATH at the base model for a clean run,
#     or at a prior checkpoint to continue training it)
# 3. Save Version, then Save & Run All (Commit) for unattended runs
```

Kaggle wipes `/kaggle/working` when a session ends. Use Save Version so the checkpoint
survives in the notebook's Output, and attach that output as an Input in a separate
notebook to load it afterwards.

---

## 🔭 Next steps

- Raise `CONCAT_MAX_CLIPS` from 3 to 6 or 8 and `CONCAT_FRACTION` toward 0.5, so the model
  sees 20 to 25 second utterances as normal rather than as extreme outliers.
- Filter the corpus by signal to noise before adding volume. A smaller set of clean clips
  is likely to beat a larger noisy one.
- Publish weights to Hugging Face.

---

## 📄 Attribution and licensing

This repo contains **only original pipeline code** (MIT, see `LICENSE`). No third party
weights or datasets are redistributed here.

- **Base model**: [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) and [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano), Apache 2.0, © OpenMOSS Team. Fine tuned weights remain under this license.
- **Training data**: [Mozilla Common Voice, Persian](https://commonvoice.mozilla.org/), CC0 (public domain).

<div align="center">

*Not affiliated with or endorsed by the OpenMOSS team.*

</div>