<div align="center">

# AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes

**ICML 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-2563eb)](https://fudancvl.github.io/AVTrack/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/abs/2606.02724)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b)](https://arxiv.org/abs/2606.02724)
[![Dataset on 🤗](https://img.shields.io/badge/Dataset-FudanCVL%2FAVTrack-ffd21e)](https://huggingface.co/datasets/FudanCVL/AVTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A human-centric audio-visual instance segmentation dataset for dynamic real-world scenes.

<img src="https://raw.githubusercontent.com/FudanCVL/AVTrack/gh-pages/static/images/demo_grid_43x20.gif" width="100%" alt="AVTrack dataset montage (43x20 grid of speakers)">

</div>

---

## News

- **2026-05** &nbsp; AVTrack code, dataset, and project page released.
- **2026-05** &nbsp; AVTrack is accepted to **ICML 2026** 🎉

## Authors

[Yaoting Wang](https://yaotingwangofficial.github.io),
[Yun Zhou](https://henghuiding.com),
[Zipei Zhang](https://henghuiding.com),
[Henghui Ding](https://henghuiding.com)

*Institute of Big Data, College of Computer Science and Artificial Intelligence, Fudan University, Shanghai, China*

## What is AVTrack?

**AVTrack** is a human-centric audio-visual instance segmentation (AVIS) dataset built specifically for evaluation in *dynamic, real-world* scenes. It complements existing AVIS benchmarks — which often rely on static, single-speaker, or laboratory-style footage — with the kind of messy conditions that real applications actually see.

- **871 videos**, 100% test split, averaging **54 s** per clip.
- **Pixel-level instance masks** with cross-frame identity (tracking), plus aligned audio.
- Spans interviews, films, anime, operas, narrations, and stage performances — broad coverage of speakers, languages, and acoustic conditions.
- Per-video **challenge attributes**: camera motion, occlusion, position changes, overlapping speech, and more.
- Released with a **training-free baseline (AVTracker)** to bootstrap future research.

See the [project page](https://fudancvl.github.io/AVTrack/) for figures and a dataset comparison table.

## Pipeline Overview (AVTracker baseline)

```
Input: Video Frames + Audio
         │
    ┌────┴────┐
    ▼         ▼
 SAM3      Whisper
 (Track)   (ASR)
    │         │
    │    [Optional: Mossformer2 speech separation]
    │         │
    │    Speech Chunks (with timestamps)
    │         │
    │    [Optional: Speaker similarity compression]
    │         │
    └────┬────┘
         ▼
   Local Window Analysis
   (VLM per-chunk: who is speaking?)
         │
         ▼ IoU matching (VLM bbox ↔ SAM3 mask)
         │
         ▼
   Global Window Analysis
   (VLM cross-frame: which frames = same person?)
   [or DeepFace in face variant]
         │
         ▼
   Output: Per-person tracklets (mask sequences)
```

## Project Structure

```
AVTrack/
├── configs/                            # Hydra configuration (all hyperparams here)
│   ├── config.yaml                     #   Main config with defaults composition
│   ├── agent/                          #   Variant configs
│   │   ├── base.yaml                   #     Full method (default)
│   │   ├── no_separation.yaml          #     Ablation: disable Mossformer2 separation
│   │   ├── no_compression.yaml         #     Ablation: disable speech-chunk compression
│   │   ├── fix_window.yaml             #     Ablation: fixed 8 s window
│   │   └── face.yaml                   #     Global stage uses DeepFace instead of VLM
│   ├── model/                          #   Model selection configs (asr, vlm, tracker, speaker, separator)
│   ├── data/avtrack.yaml               #   Dataset paths (env-var driven)
│   └── eval/default.yaml               #   Evaluation metrics config
│
├── src/                                # Core source code
│   ├── agent/                          #   BaseAgent / SepAgent / FaceAgent + factory
│   ├── skills/                         #   ASR · SAM3 tracker · Qwen3-VL · ECAPA speaker · Mossformer2 · DeepFace
│   ├── data/                           #   AVTrackDataset loader
│   ├── evaluation/                     #   HOTA / DetA / AssA / IDF1 / MOTA via trackeval
│   ├── prompts/                        #   VLM prompt templates
│   └── utils/                          #   I/O, seeding, matching, text, visualization
│
├── scripts/
│   ├── run_inference.py                #   Hydra app: single-GPU & multi-GPU inference
│   ├── run_inference_one_video_example.py  # Single-video demo entry
│   └── run_eval.py                     #   Hydra app: evaluation
│
├── docs/                               # Algorithm/method notes
├── pyproject.toml                      # uv-managed project metadata
├── uv.lock                             # Pinned dependency lockfile
├── LICENSE                             # MIT
├── .env.example                        # Environment variable template
└── .gitignore
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. A pinned `uv.lock` is committed for reproducibility.

### 1. Install uv (one-time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and sync

```bash
git clone https://github.com/FudanCVL/AVTrack.git
cd AVTrack

# Core dependencies (Python 3.12 + PyTorch CUDA 12.6 by default)
uv sync

# Or include all optional groups (recommended for full pipeline)
uv sync --extra sep --extra face --extra dev
```

Run anything inside the managed environment with `uv run`, e.g.
`uv run python scripts/run_inference.py`.

### 3. Different CUDA / CPU-only

The default torch wheels target CUDA 12.6. Override with another PyTorch index when syncing:

```bash
# CUDA 12.1
uv sync --reinstall-package torch --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# CPU only
uv sync --reinstall-package torch --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
```

### 4. Configure paths and API keys

```bash
cp .env.example .env
# Edit .env to set AVTRACK_DATA_ROOT and (optionally) local model paths.
# DASHSCOPE_API_KEY is only needed for the Qwen3-VL-Plus API agent.
```

## Dataset

The dataset is hosted on Hugging Face: **[FudanCVL/AVTrack](https://huggingface.co/datasets/FudanCVL/AVTrack)**.

```bash
# Download and unpack
hf download FudanCVL/AVTrack AVTrack.zip --repo-type dataset --local-dir .
unzip AVTrack.zip -d avtrack_data
export AVTRACK_DATA_ROOT=$PWD/avtrack_data
```

After unpacking, `AVTRACK_DATA_ROOT` should contain:

```
$AVTRACK_DATA_ROOT/
├── avtrack_meta.json
├── Images/
├── Audios/
├── Instance_Masks/
└── Instance_Masks_merged/
```

Individual paths can also be overridden directly, e.g. `data.image_dir=/custom/Images`.

## Models

Configs reference HuggingFace repo IDs and weights are downloaded automatically on first use. For offline runs, point each model config to a local directory via the corresponding environment variable (see `.env.example`) or via a CLI override.

| Model | Default (HF repo) | Local-path env var |
|-------|-------------------|---------------------|
| SAM3 | `facebook/sam3` | `SAM3_PATH` |
| Whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | `WHISPER_LARGE_PATH` |
| Whisper-small | `openai/whisper-small` | `WHISPER_SMALL_PATH` |
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `QWEN3VL_4B_PATH` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `QWEN3VL_8B_PATH` |
| ECAPA-VoxCeleb | `speechbrain/spkrec-ecapa-voxceleb` | — |

## Inference

```bash
# Default: full method (compression + speech-boundary windows + VLM global
# matching + Mossformer2 speech separation), Whisper-large, Qwen3-VL-4B, single GPU
uv run python scripts/run_inference.py

# Specify GPU
uv run python scripts/run_inference.py device=cuda:2

# Default method with the larger 8B VLM
uv run python scripts/run_inference.py model/vlm=qwen3vl_8b

# Use face-based global matching
uv run python scripts/run_inference.py agent=face

# Ablation: disable speech separation
uv run python scripts/run_inference.py agent=no_separation

# Ablation: disable speech-chunk compression
uv run python scripts/run_inference.py agent=no_compression

# Ablation: fixed 8 s window instead of speech boundaries
uv run python scripts/run_inference.py agent=fix_window model/asr=whisper_small

# Use API-based VLM (requires DASHSCOPE_API_KEY in .env)
uv run python scripts/run_inference.py model/vlm=qwen3vl_plus

# Multi-GPU parallel inference (6 GPUs)
uv run python scripts/run_inference.py inference.mode=multi_gpu inference.gpu_list=[0,1,2,3,4,5]

# Custom output directory
uv run python scripts/run_inference.py output_dir=outputs/my_experiment
```

### Single-video example

For a quick smoke test on one video without going through the full dataset loader:

```bash
uv run python scripts/run_inference_one_video_example.py \
    --frames $AVTRACK_DATA_ROOT/Images/interview_Z_16 \
    --audio  $AVTRACK_DATA_ROOT/Audios/interview_Z_16.wav \
    --output_dir outputs/one_video_demo \
    --device cuda:0
```

The script also accepts a single `--video clip.mp4` and runs ffmpeg under the hood to extract frames and a 16 kHz mono wav.

## Evaluation

```bash
# Evaluate predictions
uv run python scripts/run_eval.py output_dir=outputs/base

# Evaluate with a specific variant config
uv run python scripts/run_eval.py output_dir=outputs/no_separation agent=no_separation
```

### Config override examples

All hyperparameters live in YAML configs and can be overridden via CLI:

```bash
# Change speaker similarity threshold
uv run python scripts/run_inference.py agent.compression.threshold=0.4

# Change max frames per chunk
uv run python scripts/run_inference.py inference.max_frame_count=50

# Change IoU matching threshold
uv run python scripts/run_inference.py agent.thresholds.iou_matching=0.5

# Print resolved config without running
uv run python scripts/run_inference.py --cfg job
```

## Variants

| Variant | Config | Description |
|---|---|---|
| **Base** (default) | `agent=base` | Full method: compression + speech-boundary windows + VLM global matching + Mossformer2 speech separation |
| **No Separation** | `agent=no_separation` | Ablation: disable Mossformer2 speech separation |
| **No Compression** | `agent=no_compression` | Ablation: disable speech-chunk compression |
| **Fixed Window** | `agent=fix_window` | Ablation: use fixed 8 s windows instead of speech boundaries |
| **Face** | `agent=face` | Global stage uses DeepFace instead of VLM for person grouping |
| **Plus** | `model/vlm=qwen3vl_plus` | Use the Qwen3-VL-Plus API for stronger VLM reasoning |

## Citation

If you find AVTrack useful in your research, please cite our paper:

```bibtex
@inproceedings{wang2026avtrack,
  title     = {{AVTrack}: Audio-Visual Tracking in Human-centric Complex Scenes},
  author    = {Wang, Yaoting and Zhou, Yun and Zhang, Zipei and Ding, Henghui},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE). Note that the third-party models used by this pipeline (SAM3, Whisper, Qwen3-VL, ECAPA-VoxCeleb, Mossformer2, DeepFace) are governed by their own licenses; please consult each model card before redistribution.
