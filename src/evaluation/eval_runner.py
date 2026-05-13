"""Evaluation runner for AVTrack predictions.

Provides a config-driven evaluation loop that merges per-instance masks,
computes tracking metrics per video, and returns averaged results.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from tqdm import tqdm

from .metrics import get_track_metrics, merge_per_instance_mask

logger = logging.getLogger(__name__)

EMPTY_RESULT: Dict[str, float] = {
    "HOTA": 0.0,
    "DetA": 0.0,
    "AssA": 0.0,
    "IDF1": 0.0,
    "MOTA": 0.0,
}


def evaluate(
    cfg: DictConfig,
    pred_mask_dir: str,
    pred_mask_merge_dir: str = "",
) -> Dict[str, float]:
    """Run evaluation on predictions.

    Args:
        cfg: Full Hydra config (needs cfg.data.meta_file, cfg.data.gt_mask_merged_dir, cfg.eval).
        pred_mask_dir: Directory containing prediction masks.
        pred_mask_merge_dir: Directory to store merged prediction masks.
            If empty, defaults to pred_mask_dir + "_merged".

    Returns:
        Dict of averaged metric scores.
    """
    if not pred_mask_merge_dir:
        pred_mask_merge_dir = pred_mask_dir + "_merged"

    gt_mask_dir = cfg.data.gt_mask_merged_dir

    with open(cfg.data.meta_file, "r") as f:
        data = json.load(f)
    videos = data["videos"]

    res_list, miss_list = _evaluate_videos(
        videos=videos,
        pred_mask_dir=pred_mask_dir,
        gt_mask_dir=gt_mask_dir,
        pred_mask_merge_dir=pred_mask_merge_dir,
        with_jf=cfg.eval.get("with_jf", False),
    )

    if not res_list:
        logger.warning("No results computed")
        return EMPTY_RESULT

    # Compute averages
    avg_res: Dict[str, float] = {}
    for key in res_list[0]:
        avg_res[key] = sum(r[key] for r in res_list) / len(res_list)

    logger.info("Missed videos: %d", len(miss_list))
    logger.info("Average results: %s", avg_res)
    return avg_res


def _evaluate_videos(
    videos: List[dict],
    pred_mask_dir: str,
    gt_mask_dir: str,
    pred_mask_merge_dir: str,
    with_jf: bool = False,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Evaluate all videos.

    Iterates over each video entry, merges per-instance masks if needed,
    and computes tracking metrics. Videos without GT are skipped; videos
    without predictions receive zero scores.

    Args:
        videos: List of video metadata dicts (must contain "file_names").
        pred_mask_dir: Root directory of per-instance prediction masks.
        gt_mask_dir: Root directory of ground-truth merged masks.
        pred_mask_merge_dir: Directory for storing merged prediction masks.
        with_jf: Whether to compute J&F metrics.

    Returns:
        Tuple of (results list, missed video names list).
    """
    res_list: List[Dict[str, float]] = []
    miss_list: List[str] = []

    for video in tqdm(videos, desc="Evaluating videos"):
        video_name = video["file_names"][0].split("/")[0]
        pred_path = os.path.join(pred_mask_dir, video_name)
        gt_path = os.path.join(gt_mask_dir, video_name)

        if not os.path.exists(gt_path):
            miss_list.append(video_name)
            continue

        if not os.path.exists(pred_path):
            res = EMPTY_RESULT.copy()
        else:
            try:
                merge_tgt = os.path.join(pred_mask_merge_dir, video_name)
                if not os.path.exists(merge_tgt):
                    merge_per_instance_mask(pred_path, gt_path, merge_tgt)
                res = get_track_metrics(merge_tgt, gt_path, with_jf=with_jf)
            except Exception as e:
                res = EMPTY_RESULT.copy()
                logger.error("Error evaluating %s: %s", video_name, e)

        # Clamp to non-negative
        res = {k: max(v, 0.0) for k, v in res.items()}
        res_list.append(res)
        logger.debug("Video %s: %s", video_name, res)

    return res_list, miss_list
