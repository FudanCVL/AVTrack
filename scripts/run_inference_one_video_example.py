"""Run AVTrack inference on a single video.

Two input modes are supported:

1. Provide an mp4/mkv/... video file and let ffmpeg extract frames + audio:

    python scripts/run_inference_one_video_example.py \
        --video /path/to/clip.mp4 \
        --output_dir outputs/demo

2. Provide a directory of frames (sorted by filename) and a wav file:

    python scripts/run_inference_one_video_example.py \
        --frames /path/to/frames_dir \
        --audio /path/to/audio.wav \
        --output_dir outputs/demo

Hydra overrides may be appended after `--` to switch agent / model variants:

    python scripts/run_inference_one_video_example.py --video clip.mp4 -- \
        agent=sep model/vlm=qwen3vl_8b
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Make `src` importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import get_agent  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-video AVTrack inference example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file (mp4/mkv/...). Requires ffmpeg.")
    parser.add_argument("--frames", type=str, default=None,
                        help="Directory of sorted frames (alternative to --video).")
    parser.add_argument("--audio", type=str, default=None,
                        help="WAV audio path (used with --frames).")
    parser.add_argument("--output_dir", type=str, default="outputs/one_video_demo",
                        help="Directory to write tracklets / visualizations.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frame extraction rate when --video is given (default: 5).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER,
                        help="Hydra overrides forwarded to compose (after `--`).")
    args = parser.parse_args()

    if not args.video and not (args.frames and args.audio):
        parser.error("Provide either --video, or both --frames and --audio.")
    return args


def _extract_frames_and_audio(
    video_path: Path,
    work_dir: Path,
    fps: float,
) -> tuple[Path, Path]:
    """Use ffmpeg to extract frames and a 16 kHz mono wav from a video file."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --video mode. Install it or pass --frames/--audio.")

    frames_dir = work_dir / "frames"
    audio_path = work_dir / "audio.wav"
    frames_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting frames at %.2f fps -> %s", fps, frames_dir)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            str(frames_dir / "%06d.jpg"),
        ],
        check=True,
    )

    logger.info("Extracting audio -> %s", audio_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(audio_path),
        ],
        check=True,
    )
    return frames_dir, audio_path


def _list_frames(frames_dir: Path) -> List[str]:
    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in FRAME_EXTS
    )
    if not frames:
        raise RuntimeError(f"No image frames found in {frames_dir}")
    return [str(p) for p in frames]


def _load_cfg(overrides: List[str], device: str, output_dir: str):
    """Compose the same Hydra config used by run_inference.py."""
    overrides = [o for o in overrides if o != "--"]
    overrides = overrides + [f"device={device}", f"output_dir={output_dir}"]
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        frames_dir, audio_path = _extract_frames_and_audio(
            video_path=Path(args.video).resolve(),
            work_dir=output_dir / "_extracted",
            fps=args.fps,
        )
    else:
        frames_dir = Path(args.frames).resolve()
        audio_path = Path(args.audio).resolve()

    frame_paths = _list_frames(frames_dir)
    logger.info("Loaded %d frames; audio=%s", len(frame_paths), audio_path)

    cfg = _load_cfg(args.overrides or [], device=args.device, output_dir=str(output_dir))
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    agent = get_agent(cfg)

    tracklets = agent.inference(frame_paths, str(audio_path))
    logger.info("Done. Produced %d tracklet(s). Outputs in %s",
                len(tracklets) if tracklets is not None else 0, output_dir)


if __name__ == "__main__":
    main()
