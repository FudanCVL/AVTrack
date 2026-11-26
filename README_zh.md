<div align="center">

# AVTrack：以人为中心的复杂场景视听跟踪

**ICML 2026**

[![项目主页](https://img.shields.io/badge/Project-Page-2563eb)](https://fudancvl.github.io/AVTrack/)
[![论文](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/abs/2606.02724)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b)](https://arxiv.org/abs/2606.02724)
[![数据集](https://img.shields.io/badge/Dataset-FudanCVL%2FAVTrack-ffd21e)](https://huggingface.co/datasets/FudanCVL/AVTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

面向动态真实场景、以人为中心的视听实例分割数据集。

<img src="https://raw.githubusercontent.com/FudanCVL/AVTrack/gh-pages/static/images/demo_grid_43x20.gif" width="100%" alt="AVTrack 数据集示例，43×20 网格">

</div>

---

## 最新动态

- **2026-05** 发布 AVTrack 代码、数据集和项目主页。
- **2026-05** AVTrack 被 **ICML 2026** 接收。
- **2026-11** 原始高帧率视频：Hugging Face 上的 [`Videos_high_fps.zip`](https://huggingface.co/datasets/FudanCVL/AVTrack/blob/main/Videos_high_fps.zip)。标注帧与视频时间戳的对应方法见 [`docs/high-fps/`](docs/high-fps/README_zh.md)。

## 作者

[Yaoting Wang](https://yaotingwangofficial.github.io)、
[Yun Zhou](https://henghuiding.com)、
[Zipei Zhang](https://henghuiding.com)、
[Henghui Ding](https://henghuiding.com)

*复旦大学大数据研究院、计算机科学与人工智能学院，中国上海*

## AVTrack 是什么

**AVTrack** 是一个以人为中心的视听实例分割（AVIS）数据集，用于评测模型在动态真实场景中的表现。现有 AVIS benchmark 多使用静态、单人说话或实验室环境中的视频。AVTrack 补充了实际应用中常见的复杂场景。

- 共 **871 段视频**，全部用于测试，每段平均 **54 秒**。
- 提供**像素级实例 mask**、跨帧身份标注和对齐的音频。
- 涵盖访谈、电影、动画、戏曲、解说和舞台表演，覆盖不同说话人、语言和声学条件。
- 为每段视频标注挑战因素，包括镜头运动、遮挡、位置变化和多人同时说话等。
- 同时发布无需训练的 baseline **AVTracker**。

图示和数据集对比见[项目主页](https://fudancvl.github.io/AVTrack/)。

## AVTracker 流程

```
输入：视频帧 + 音频
         │
    ┌────┴────┐
    ▼         ▼
 SAM3      Whisper
（跟踪）    （ASR）
    │         │
    │    [可选：使用 Mossformer2 分离语音]
    │         │
    │    带时间戳的语音片段
    │         │
    │    [可选：按说话人相似度合并语音片段]
    │         │
    └────┬────┘
         ▼
   局部窗口分析
   （VLM 判断每个语音片段里谁在说话）
         │
         ▼ IoU 匹配（VLM bbox ↔ SAM3 mask）
         │
         ▼
   全局窗口分析
   （VLM 判断不同帧中的人物是否为同一人）
   [Face 配置使用 DeepFace]
         │
         ▼
   输出：每个人的跟踪轨迹（mask 序列）
```

## 项目结构

```
AVTrack/
├── configs/                            # Hydra 配置，包含所有超参数
│   ├── config.yaml                     #   主配置，定义默认配置组合
│   ├── agent/                          #   各方法配置
│   │   ├── base.yaml                   #     完整方法（默认）
│   │   ├── no_separation.yaml          #     Ablation：关闭 Mossformer2 语音分离
│   │   ├── no_compression.yaml         #     Ablation：关闭语音片段合并
│   │   ├── fix_window.yaml             #     Ablation：采用固定 8 秒窗口
│   │   └── face.yaml                   #     全局阶段使用 DeepFace 匹配人物
│   ├── model/                          #   模型配置（asr、vlm、tracker、speaker、separator）
│   ├── data/avtrack.yaml                #   数据集路径，通过环境变量设置
│   └── eval/default.yaml               #   评测指标配置
│
├── src/                                # 核心代码
│   ├── agent/                          #   BaseAgent、SepAgent、FaceAgent 及其 Factory
│   ├── skills/                         #   ASR、SAM3、Qwen3-VL、ECAPA、Mossformer2、DeepFace
│   ├── data/                           #   AVTrackDataset 数据加载
│   ├── evaluation/                     #   通过 trackeval 计算 HOTA、DetA、AssA、IDF1、MOTA
│   ├── prompts/                        #   VLM prompt 模板
│   └── utils/                          #   I/O、随机种子、匹配、文本处理和可视化
│
├── scripts/
│   ├── run_inference.py                #   单 GPU 和多 GPU 推理入口
│   ├── run_inference_one_video_example.py  # 单视频示例
│   └── run_eval.py                     #   评测入口
│
├── docs/                               # 项目文档
├── pyproject.toml                      # uv 管理的项目配置
├── uv.lock                             # 依赖锁文件
├── LICENSE                             # MIT
├── .env.example                        # 环境变量模板
└── .gitignore
```

## 安装与配置

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。仓库提供 `uv.lock`，可按锁定版本安装。

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆仓库并安装依赖

```bash
git clone https://github.com/FudanCVL/AVTrack.git
cd AVTrack

# 安装核心依赖，默认使用 Python 3.12 和 PyTorch CUDA 12.6
uv sync

# 完整流程可同时安装所有可选依赖
uv sync --extra sep --extra face --extra dev
```

用 `uv run` 在项目环境中运行命令，例如 `uv run python scripts/run_inference.py`。

### 3. 使用其他 CUDA 版本或 CPU

默认的 PyTorch wheel 对应 CUDA 12.6。用以下命令切换 PyTorch 下载源。

```bash
# CUDA 12.1
uv sync --reinstall-package torch --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# 仅使用 CPU
uv sync --reinstall-package torch --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
```

### 4. 配置路径和 API key

```bash
cp .env.example .env
# 在 .env 中设置 AVTRACK_DATA_ROOT，可按需指定本地模型路径。
# 只有使用 Qwen3-VL-Plus API 时才需要设置 DASHSCOPE_API_KEY。
```

## 数据集

数据集托管在 Hugging Face（[FudanCVL/AVTrack](https://huggingface.co/datasets/FudanCVL/AVTrack)）。

```bash
# 下载并解压
hf download FudanCVL/AVTrack AVTrack.zip --repo-type dataset --local-dir .
unzip AVTrack.zip -d avtrack_data
export AVTRACK_DATA_ROOT=$PWD/avtrack_data
```

解压后，`AVTRACK_DATA_ROOT` 应包含以下内容。

```
$AVTRACK_DATA_ROOT/
├── avtrack_meta.json
├── Images/
├── Audios/
├── Instance_Masks/
└── Instance_Masks_merged/
```

也可以单独指定某项路径，例如 `data.image_dir=/custom/Images`。

原始高帧率视频单独打包为同一仓库下的 [`Videos_high_fps.zip`](https://huggingface.co/datasets/FudanCVL/AVTrack/blob/main/Videos_high_fps.zip)。下载后解压到数据集目录：

```bash
hf download FudanCVL/AVTrack Videos_high_fps.zip --repo-type dataset --revision main --local-dir .
unzip Videos_high_fps.zip -d "$AVTRACK_DATA_ROOT"
```

解压后，视频位于 `AVTRACK_DATA_ROOT` 下的 `Videos/<category>/<clip>_merged.mp4`。该 ZIP 仅包含视频；标注仍在 `AVTrack.zip` 中。标注图与 MP4 时间戳的对应关系见 [`docs/high-fps/`](docs/high-fps/README_zh.md)。

## 模型

配置文件通过 Hugging Face 仓库 ID 指定模型，首次运行时会自动下载权重。离线运行时，可通过对应的环境变量（见 `.env.example`）或命令行参数指定本地模型目录。

| 模型 | 默认 Hugging Face 仓库 | 本地路径环境变量 |
| --- | --- | --- |
| SAM3 | `facebook/sam3` | `SAM3_PATH` |
| Whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | `WHISPER_LARGE_PATH` |
| Whisper-small | `openai/whisper-small` | `WHISPER_SMALL_PATH` |
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `QWEN3VL_4B_PATH` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `QWEN3VL_8B_PATH` |
| ECAPA-VoxCeleb | `speechbrain/spkrec-ecapa-voxceleb` | — |

## 推理

```bash
# 默认使用完整方法：合并语音片段、按语音边界划分窗口、
# 用 VLM 做全局匹配，并使用 Mossformer2 分离语音。
# 模型为 Whisper-large 和 Qwen3-VL-4B，使用单 GPU。
uv run python scripts/run_inference.py

# 指定 GPU
uv run python scripts/run_inference.py device=cuda:2

# 使用 8B VLM
uv run python scripts/run_inference.py model/vlm=qwen3vl_8b

# 在全局阶段按人脸匹配人物
uv run python scripts/run_inference.py agent=face

# Ablation：关闭语音分离
uv run python scripts/run_inference.py agent=no_separation

# Ablation：关闭语音片段合并
uv run python scripts/run_inference.py agent=no_compression

# Ablation：使用固定的 8 秒窗口
uv run python scripts/run_inference.py agent=fix_window model/asr=whisper_small

# 使用 VLM API，需要在 .env 中设置 DASHSCOPE_API_KEY
uv run python scripts/run_inference.py model/vlm=qwen3vl_plus

# 使用 6 块 GPU 并行推理
uv run python scripts/run_inference.py inference.mode=multi_gpu inference.gpu_list=[0,1,2,3,4,5]

# 自定义输出目录
uv run python scripts/run_inference.py output_dir=outputs/my_experiment
```

### 单视频示例

用以下命令在一段视频上做快速验证，可以直接指定图片目录和 WAV 文件。

```bash
uv run python scripts/run_inference_one_video_example.py \
    --frames $AVTRACK_DATA_ROOT/Images/interview_Z_16 \
    --audio  $AVTRACK_DATA_ROOT/Audios/interview_Z_16.wav \
    --output_dir outputs/one_video_demo \
    --device cuda:0
```

也可以用 `--video clip.mp4` 直接传入视频。脚本会调用 ffmpeg 提取视频帧和 16 kHz 单声道 WAV。

## 评测

```bash
# 评测预测结果
uv run python scripts/run_eval.py output_dir=outputs/base

# 按指定的方法配置评测
uv run python scripts/run_eval.py output_dir=outputs/no_separation agent=no_separation
```

### 修改配置的示例

超参数存放在 YAML 配置文件中，也可以从命令行覆盖。

```bash
# 修改说话人相似度阈值
uv run python scripts/run_inference.py agent.compression.threshold=0.4

# 修改每个语音片段最多使用的帧数
uv run python scripts/run_inference.py inference.max_frame_count=50

# 修改 IoU 匹配阈值
uv run python scripts/run_inference.py agent.thresholds.iou_matching=0.5

# 查看最终配置，不启动推理
uv run python scripts/run_inference.py --cfg job
```

## 方法配置

| 方法 | 配置 | 说明 |
| --- | --- | --- |
| **Base**（默认） | `agent=base` | 完整方法，包含语音片段合并、按语音边界划分窗口、VLM 全局匹配和 Mossformer2 语音分离。 |
| **No Separation** | `agent=no_separation` | 关闭 Mossformer2 语音分离。 |
| **No Compression** | `agent=no_compression` | 关闭语音片段合并。 |
| **Fixed Window** | `agent=fix_window` | 使用固定的 8 秒窗口。 |
| **Face** | `agent=face` | 在全局阶段用 DeepFace 匹配同一人。 |
| **Plus** | `model/vlm=qwen3vl_plus` | 使用 Qwen3-VL-Plus API 进行 VLM 推理。 |

## 引用

如果你的研究使用了 AVTrack，请引用以下论文。

```bibtex
@inproceedings{wang2026avtrack,
  title     = {{AVTrack}: Audio-Visual Tracking in Human-centric Complex Scenes},
  author    = {Wang, Yaoting and Zhou, Yun and Zhang, Zipei and Ding, Henghui},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 许可证

本项目采用 [MIT License](LICENSE)。流程使用的第三方模型（SAM3、Whisper、Qwen3-VL、ECAPA-VoxCeleb、Mossformer2、DeepFace）各自适用对应的许可证。再分发前请查阅各模型的 model card。
