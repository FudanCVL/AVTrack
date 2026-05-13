import logging
import os
import random
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agent import get_agent
from src.data import AVTrackDataset
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """AVTrack inference entry point.

    Single GPU:
        python scripts/run_inference.py device=cuda:0

    Multi GPU:
        python scripts/run_inference.py inference.mode=multi_gpu

    Override agent:
        python scripts/run_inference.py agent=sep model/vlm=qwen3vl_8b
    """
    set_seed(cfg.seed)

    if cfg.inference.get("mode") == "multi_gpu":
        _launch_multi_gpu(cfg)
    else:
        _run_single_gpu(cfg)


def _run_single_gpu(cfg: DictConfig) -> None:
    """Run inference on a single GPU."""
    dataset = AVTrackDataset(cfg.data)
    output_dir = cfg.get("output_dir", f"outputs/{cfg.agent.name}")
    os.makedirs(output_dir, exist_ok=True)

    samples = dataset.get_samples(
        output_dir=output_dir,
        patch_counts=cfg.inference.get("patch_counts", 1),
        patch_idx=cfg.inference.get("patch_idx", 0),
        subset_file=cfg.inference.get("subset_file", None),
    )
    logger.info("Unprocessed videos: %d", len(samples))

    random.shuffle(samples)
    agent = get_agent(cfg)

    for sample in samples:
        try:
            logger.info("Processing video %s: %s", sample.video_id, sample.video_name)
            agent.inference(sample.frame_paths, sample.audio_path)
        except Exception:
            logger.exception("Error processing video %s", sample.video_id)
            continue


def _launch_multi_gpu(cfg: DictConfig) -> None:
    """Launch parallel processes across multiple GPUs."""
    gpu_list = list(cfg.inference.gpu_list)
    patch_counts = len(gpu_list)
    output_dir = cfg.get("output_dir", f"outputs/{cfg.agent.name}")

    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    processes = []
    log_files = []

    logger.info("Starting %d parallel processes on GPUs: %s", len(gpu_list), gpu_list)

    for patch_idx, gpu_id in enumerate(gpu_list):
        device = f"cuda:{gpu_id}"
        log_file = f"logs/log_device{gpu_id}_patch{patch_idx}_{timestamp}.txt"

        # Build command with Hydra overrides
        cmd = [
            sys.executable,
            "scripts/run_inference.py",
            f"device={device}",
            f"output_dir={output_dir}",
            f"inference.mode=single_gpu",
            f"inference.patch_counts={patch_counts}",
            f"inference.patch_idx={patch_idx}",
            f"agent={cfg.agent.name}",
        ]

        logger.info("Launching process %d on %s: %s", patch_idx, device, " ".join(cmd))

        log_f = open(log_file, "w")
        log_files.append(log_f)

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        processes.append({
            "process": proc,
            "gpu_id": gpu_id,
            "patch_idx": patch_idx,
            "log_file": log_file,
        })
        time.sleep(0.5)

    logger.info("All %d processes started. Waiting for completion...", len(processes))

    # Wait for all
    failed = []
    for proc_info in processes:
        proc = proc_info["process"]
        return_code = proc.wait()
        log_files[proc_info["patch_idx"]].close()

        if return_code != 0:
            logger.error("Process %d (GPU %d) failed with code %d",
                        proc_info["patch_idx"], proc_info["gpu_id"], return_code)
            failed.append(proc_info)
        else:
            logger.info("Process %d (GPU %d) completed", proc_info["patch_idx"], proc_info["gpu_id"])

    if failed:
        logger.error("%d process(es) failed", len(failed))
        sys.exit(1)
    else:
        logger.info("All %d tasks completed successfully!", len(processes))


if __name__ == "__main__":
    main()
