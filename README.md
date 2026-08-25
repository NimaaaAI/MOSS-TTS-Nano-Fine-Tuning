<div align="center">

# 🎙️ MOSS-TTS-Nano Persian Fine-Tune

<img width="1536" height="1024" alt="ChatGPT Image Aug 25, 2026 at 11_27_08 PM" src="https://github.com/user-attachments/assets/7bbf4465-da83-426e-b8c8-5dbf0ce60535" />

**Bringing native Persian accent quality to a 117M parameter TTS model**

[![License: MIT](https://img.shields.io/badge/code%20license-MIT-yellow.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-Apache--2.0-green.svg)](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-MOSS--TTS--Nano--Persian-orange)](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian)
[![Base Model](https://img.shields.io/badge/base%20model-MOSS--TTS--Nano-blue)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano)
[![Training Notebook](https://img.shields.io/badge/training%20notebook-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)

**[🤗 Model on Hugging Face](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian)** ·
**[📓 Training notebook on Kaggle](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2)**

</div>

---

## ✨ What is this?

A fine tuning pipeline for [**OpenMOSS-Team/MOSS-TTS-Nano**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano),
a lightweight 117M parameter TTS model, adapted to sound genuinely **Persian** rather than
technically correct Persian with a foreign accent.

Persian is one of the 20 languages the MOSS-TTS family officially supports, but voice cloned
output from the Nano model leans flat and non native sounding. This project closes that gap,
and documents what was learned along the way about where the remaining limits actually come from.

The resulting weights are published at
**[nimaaaAI/MOSS-TTS-Nano-Persian](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian)**.

---

## 🎯 Result

Same reference clip, same Persian text, same seed, only the weights differ: the fine tuned
checkpoint produces a noticeably more native Persian accent. The base model reads Persian
with an audible English or Chinese accent, consistent with the distribution of its training data.

The gain is in accent and pronunciation. Long form stability is a separate problem, documented
under Known limitations.

---

## 🔊 Samples

Side by side base against fine tuned, all generated from the same English reference voice so
the accent difference is audible. GitHub does not embed audio players, so these open on
Hugging Face.

| Text | Base model | This fine tune |
|---|---|---|
| امروز هوا بسیار خوب است. | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t1_base.wav) | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t1_finetuned.wav) |
| لطفاً در را ببندید. | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t2_base.wav) | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t2_finetuned.wav) |
| خواهرم در دانشگاه تهران درس می‌خواند. | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t3_base.wav) | [▶ play](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian/resolve/main/samples/t3_finetuned.wav) |

All six play inline on the
[model card](https://huggingface.co/nimaaaAI/MOSS-TTS-Nano-Persian).

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

Everything lives in one notebook, `finetune.ipynb`. Sample size, epochs, learning rate, and
which checkpoint to resume from are all controlled from a single `CONFIG` cell at the top, so
each round builds on the last without rewriting the pipeline. The notebook clones the upstream
[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) repo automatically for its
training and inference scripts.

---

## 📊 Training rounds

Every round was trained on Kaggle's free T4×2. The
[public notebook](https://www.kaggle.com/code/nimasaghi/moss-tts-nano-fine-tuning-v2) has each
round's config and logs under its Versions tab.

| Round | Clips | Started from | Epochs | Notes |
|---|---|---|---|---|
| 1 | 4,000 | base model | 3 | First accent improvement. Revealed the cutoff problem on long input. |
| 2 | 25,000 | round 1 checkpoint | 3 | 30% concatenated groups added. ~6h41m. |
| 3 | 34,000 | round 2 checkpoint | 3 | Modest gain over round 2. Diminishing returns from stacking rounds. |
| 4 | 34,000 | **base model** | 3 | **Published.** Concatenation rebuilt per speaker. See Findings. |

Round 4 deliberately restarts from the base model rather than continuing the chain. Rounds 2
and 3 trained repeatedly over overlapping data, so parts of the corpus had been seen six or
nine times, which showed up as diminishing returns rather than improvement.

---

## 🔬 Findings

Three things worth writing down, because each one cost a training round to discover.

**Concatenated examples were mixing speakers.** The multiple sentence groups were built by
slicing a globally shuffled dataframe, so almost every group spliced 2 or 3 different people
into one audio file paired with one transcript. Thirty percent of the training data was
teaching the model that the voice may change partway through an utterance, which is the
opposite of what a voice cloning model should learn. Fixed in round 4 by grouping on
`client_id` before concatenating. The corpus supports this easily: 102,507 same speaker groups
were available against 4,080 requested.

**The training data is short.** Measured across all 27,880 encoded examples at 12.4 frames per
second, which matches the 12.5 tokens per second documented upstream: median 48 frames (3.9s),
p90 107 frames (8.6s), max 313 frames (25s), with 66% under 5 seconds. The model stops
generating around the edge of the longest thing it has seen. `MAX_LENGTH` was never the
constraint, since nothing came close to the 1024 token cap.

**Reference audio quality dominates output quality.** Same checkpoint, same text, same seed,
only the reference clip changed: a clean studio recording produced clearly better output than a
Common Voice clip did. A 117M parameter model has limited capacity to clean up a noisy prompt,
so it reproduces the room noise and microphone character along with the voice.

---

## ⚠️ Known limitations

- **Practical utterance limit is around 5 seconds.** Beyond that, output can stop early or
  degrade. This traces directly to the training data distribution above, not to a configuration
  setting. The base model shows the same behaviour, so it is a property of the autoregressive
  decoder and the corpus rather than something introduced here.
- **Long passages need chunking.** Split the text, generate each chunk with the same reference
  and seed, then concatenate the audio.
- **Output is emotionally flat.** Common Voice is volunteers reading sentences off a screen.
  There is no emotional variation in the data, so there is none in the model.
- **Reference clip matters more than expected.** Use the cleanest single speaker recording
  available. Noisy references degrade output noticeably.

For long form Persian synthesis today, a large non autoregressive model such as
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) handles full passages in a single call and
is the better tool. This project's value is in the small model footprint and in the documented
path from a flat base model to a native sounding Persian accent.

---

## 🚀 Usage

### Synthesize with the published weights

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git
cd MOSS-TTS-Nano
pip install -r requirements.txt
pip install torchvision==0.22.0   # required for the pinned torch version

python finetuning/verify.py \
  --checkpoint nimaaaAI/MOSS-TTS-Nano-Persian \
  --audio-tokenizer-pretrained-name-or-path OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano \
  --mode voice_clone \
  --text "امروز هوا بسیار خوب است." \
  --prompt-audio-path reference.wav \
  --seed 42 \
  --output-audio-path out.wav
```

Both the weights and the audio tokenizer download from the Hub on first run. `reference.wav` is
a short recording of the voice to clone.

Or load the weights directly:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "nimaaaAI/MOSS-TTS-Nano-Persian"
model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
```

**Runtime.** Runs on CPU or GPU. Add `--device cpu` or `--device cuda` to force one; the script
picks automatically otherwise. On a short utterance, end to end time was 24.3s on CPU against
21.4s on a T4. Nearly all of that is model loading, so in a long running process where the model
stays resident, per request generation is a few seconds on either.

### Reproduce or extend training

```bash
# On Kaggle (free GPU and fast dataset access):
# 1. Attach dataset: amirftma/common-voice-fa-v13
# 2. Edit the CONFIG cell at the top of finetune.ipynb
#    (point BASE_MODEL_PATH at the base model for a clean run,
#     or at a prior checkpoint to continue training it)
# 3. Save Version, then Save & Run All (Commit) for unattended runs
```

Requirements: Python 3.12, a CUDA GPU, and `ffmpeg` on PATH. The T4 does not support bf16, and
fp16 broke gradient scaling, so the pipeline runs in fp32: slower but stable across both GPUs.

Kaggle wipes `/kaggle/working` when a session ends. Use Save Version so the checkpoint survives
in the notebook's Output, then attach that output as an Input in a separate notebook to load it
afterwards.

---

## 🔭 Next steps

- Raise `CONCAT_MAX_CLIPS` from 3 to 6 or 8 and `CONCAT_FRACTION` toward 0.5, so the model sees
  20 to 25 second utterances as normal rather than as extreme outliers.
- Filter the corpus by signal to noise before adding volume. A smaller set of clean clips is
  likely to beat a larger noisy one.
- Parallelize audio conversion. 34,000 sequential ffmpeg calls take about 43 minutes; a thread
  pool cuts that to roughly 10.
- A Hugging Face Space for interactive side by side comparison.

---

## 📄 Attribution and licensing

The pipeline code in this repo is **MIT** (see `LICENSE`). The published **weights are Apache
2.0**, inherited from the base model. No third party weights or datasets are redistributed here.

- **Base model**: [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) and
  [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano),
  Apache 2.0, © OpenMOSS Team.
- **Training data**: [Mozilla Common Voice, Persian](https://commonvoice.mozilla.org/), CC0
  (public domain), via the
  [amirftma/common-voice-fa-v13](https://www.kaggle.com/datasets/amirftma/common-voice-fa-v13)
  Kaggle mirror.
- **Compute**: Kaggle free tier, T4×2.

If you use this work, please cite the upstream
[MOSS-TTS technical report](https://arxiv.org/abs/2603.18090).

<div align="center">

*Not affiliated with or endorsed by the OpenMOSS team.*

</div>
