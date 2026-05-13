"""Tracking evaluation metrics.

Provides utilities for merging per-instance masks and computing
tracking metrics (HOTA, DetA, AssA, IDF1, MOTA) via trackeval.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import trackeval
from trackeval import Evaluator, datasets, metrics

logger = logging.getLogger(__name__)


def merge_per_instance_mask(
    pred_png_mask_dir: str,
    gt_png_mask_dir: str,
    pred_png_mask_merge_dir: str,
) -> Path:
    """Merge per-instance mask PNGs into a single merged mask per frame.

    GT frame naming: frame_00000.png (5-digit).
    Prediction instance frame naming: frame_0000.png (4-digit).
    Each instance is assigned a unique color ID (starting from 1); background is 0.

    Args:
        pred_png_mask_dir: Directory with per-instance subdirectories of predicted masks.
        gt_png_mask_dir: Directory with ground-truth merged masks.
        pred_png_mask_merge_dir: Output directory for merged prediction masks.

    Returns:
        Path to the merged output directory.
    """
    pred_root = Path(pred_png_mask_dir)
    gt_root = Path(gt_png_mask_dir)

    # Only get subdirectories (one per instance)
    instance_pred_dirs: List[Path] = sorted(
        [d for d in pred_root.iterdir() if d.is_dir()]
    )
    logger.info("Found %d instance directories.", len(instance_pred_dirs))

    # Get all GT frames sorted by frame index
    gt_frame_paths = sorted(gt_root.glob("frame_*.png"))

    # Extract frame indices and sort
    gt_frames: List[Tuple[int, Path]] = []
    for p in gt_frame_paths:
        name = p.name
        if name.startswith("frame_") and name.endswith(".png"):
            try:
                idx = int(name[6:11])  # "frame_00000.png" -> "00000"
                gt_frames.append((idx, p))
            except ValueError:
                continue
    gt_frames.sort(key=lambda x: x[0])

    # Create output directory
    tgt_dir = Path(pred_png_mask_merge_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, gt_path in gt_frames:
        # Read GT to get dimensions
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            logger.warning("Cannot read GT frame %s", gt_path)
            continue
        height, width = gt_mask.shape

        # Initialize merged mask
        merged_mask = np.zeros((height, width), dtype=np.uint8)

        # Set color interval
        color_interval = 255 // len(instance_pred_dirs)

        # Iterate over each instance directory
        for instance_id, instance_dir in enumerate(instance_pred_dirs, start=1):
            # Prediction frames use 4-digit format
            pred_frame_name = f"frame_{frame_idx:04d}.png"
            pred_path = instance_dir / pred_frame_name

            if not pred_path.exists():
                continue

            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                continue

            # Binarize: >0 treated as foreground
            binary_mask = (pred_mask > 0).astype(np.uint8)
            # Assign current instance color (overlapping regions: last writer wins)
            instance_color = instance_id * color_interval
            merged_mask[binary_mask == 1] = instance_color

        # Save with 5-digit naming (consistent with GT)
        output_path = tgt_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(output_path), merged_mask)

    logger.info("Merged masks saved to: %s", tgt_dir)
    return tgt_dir


def get_track_metrics(
    pred_png_mask_dir: str,
    gt_png_mask_dir: str,
    with_jf: bool = False,
) -> Dict[str, float]:
    """Compute tracking metrics for a single video using trackeval.

    Sets up a temporary DAVIS-format directory structure for trackeval,
    computes HOTA/DetA/AssA/IDF1/MOTA (and optionally J&F), then cleans up.

    Args:
        pred_png_mask_dir: Directory containing merged prediction mask PNGs.
        gt_png_mask_dir: Directory containing ground-truth mask PNGs.
        with_jf: Whether to also compute J&F metrics.

    Returns:
        Dict with metric names as keys and scores (0-100 scale) as values.

    Raises:
        ValueError: When no common frames are found or no results are produced.
    """
    pred_dir = Path(pred_png_mask_dir)
    gt_dir = Path(gt_png_mask_dir)

    # Get intersection of frames
    gt_frames: Set[str] = {f.stem for f in gt_dir.glob("*.png")}
    pred_frames: Set[str] = {f.stem for f in pred_dir.glob("*.png")}
    common_frames: List[str] = sorted(gt_frames & pred_frames)

    if not common_frames:
        raise ValueError("No common frames found between pred and gt.")

    tracker_name = "my_tracker"
    seq_name = "sample_video"

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)

        # 1. GT directory
        gt_base = tmp_root / "gt_data"
        gt_seq_dir = gt_base / seq_name
        gt_seq_dir.mkdir(parents=True)

        # 2. Tracker directory (requires a "data" folder)
        trackers_base = tmp_root / "trackers"
        pred_seq_dir = trackers_base / tracker_name / "data" / seq_name
        pred_seq_dir.mkdir(parents=True)

        # Create symlinks (fall back to copy on OSError)
        for frame in common_frames:
            try:
                (gt_seq_dir / f"{frame}.png").symlink_to(gt_dir / f"{frame}.png")
                (pred_seq_dir / f"{frame}.png").symlink_to(pred_dir / f"{frame}.png")
            except OSError:
                shutil.copy2(gt_dir / f"{frame}.png", gt_seq_dir / f"{frame}.png")
                shutil.copy2(
                    pred_dir / f"{frame}.png", pred_seq_dir / f"{frame}.png"
                )

        # 3. Create seqmap file
        seqmap_dir = tmp_root / "seqmaps"
        seqmap_dir.mkdir()
        seqmap_path = seqmap_dir / "test.txt"
        with open(seqmap_path, "w") as f:
            f.write(seq_name)

        # 4. Dataset configuration
        dataset_config: Dict[str, object] = {
            "GT_FOLDER": str(gt_base),
            "TRACKERS_FOLDER": str(trackers_base),
            "SEQMAP_FILE": str(seqmap_path),
            "TRACKERS_TO_EVAL": [tracker_name],
            "SPLIT_TO_EVAL": "test",
            "PRINT_CONFIG": False,
            "ANNOTATION_ID": "",
            "SUBSET": "",
        }

        eval_config: Dict[str, object] = {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 8,
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_EMPTY_CLASSES": False,
            "PRINT_CONFIG": False,
        }

        clear_config: Dict[str, object] = {"PRINT_CONFIG": False}
        identity_config: Dict[str, object] = {"PRINT_CONFIG": False}

        # 5. Run evaluation
        dataset = datasets.DAVIS(dataset_config)
        metrics_list = [
            metrics.HOTA(),
            metrics.CLEAR(clear_config),
            metrics.Identity(identity_config),
        ]
        if with_jf:
            metrics_list.append(metrics.JAndF())

        evaluator = Evaluator(eval_config)
        output_res, error_msg = evaluator.evaluate([dataset], metrics_list)

        # 6. Extract results
        res_data = output_res["DAVIS"][tracker_name][seq_name]

        if not res_data:
            raise ValueError(f"No results found for sequence {seq_name}")

        # Dynamically get the class name (usually 'void')
        cls = list(res_data.keys())[0]
        actual_res = res_data[cls]

        result: Dict[str, float] = {
            "HOTA": float(actual_res["HOTA"]["HOTA"].mean() * 100),
            "DetA": float(actual_res["HOTA"]["DetA"].mean() * 100),
            "AssA": float(actual_res["HOTA"]["AssA"].mean() * 100),
            "IDF1": float(actual_res["Identity"]["IDF1"].mean() * 100),
            "MOTA": float(actual_res["CLEAR"]["MOTA"].mean() * 100),
        }
        if with_jf:
            jf_res = actual_res["JAndF"]
            result["J-Mean"] = float(jf_res.get("J-Mean", 0) * 100)
            result["F-Mean"] = float(jf_res.get("F-Mean", 0) * 100)
            result["J&F-Mean"] = (result["J-Mean"] + result["F-Mean"]) / 2

    return result
