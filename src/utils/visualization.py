"""Visualization utilities for masks, bounding boxes, and tracklets."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def overlay_masks(
    image: Image.Image,
    masks: torch.Tensor,
) -> Image.Image:
    """Overlay colored semi-transparent masks on an image.

    Args:
        image: PIL Image to overlay masks on.
        masks: Binary mask tensor of shape [n_masks, H, W].

    Returns:
        PIL Image with masks overlaid.
    """
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def add_bbox(
    image: Image.Image,
    boxes: List[List[float]],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image to draw on.
        boxes: List of bounding boxes as [x1, y1, x2, y2].
        color: RGB color tuple for the box outline.
        width: Line width of the box outline.

    Returns:
        PIL Image with bounding boxes drawn.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=width,
        )

    return image


def vis_sam3_masks(
    frames: List[Image.Image],
    masks: List[torch.Tensor],
    save_dir: str,
) -> None:
    """Visualize SAM3 masks overlaid on video frames.

    Args:
        frames: List of PIL Images (video frames).
        masks: List of mask tensors corresponding to each frame.
        save_dir: Directory to save visualization images.
    """
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx, (frame, mask) in enumerate(zip(frames, masks)):
        overlay_image = overlay_masks(frame, mask)
        overlay_image.save(os.path.join(save_dir, f"frame_{frame_idx:04d}.png"))
    logger.info("Saved SAM3 mask visualizations to %s", save_dir)


def vis_vlm_bboxes(
    frames: List[Image.Image],
    bboxes: List[List[List[float]]],
    save_dir: str,
) -> None:
    """Visualize VLM bounding boxes on video frames.

    Note:
        frame_idx may not start from 0 in the original context (FIXME).

    Args:
        frames: List of PIL Images (video frames).
        bboxes: List of bounding box lists for each frame.
        save_dir: Directory to save visualization images.
    """
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
        overlay_image = add_bbox(frame, bbox)
        overlay_image.save(os.path.join(save_dir, f"frame_{frame_idx:04d}.png"))
    logger.info("Saved VLM bbox visualizations to %s", save_dir)


def vis_instance_tracklet(
    tracklet: Dict,
    frames: List[Image.Image],
    save_dir: str,
    overlay: bool = True,
    skip_blank_mask: bool = True,
) -> None:
    """Visualize a single instance tracklet across video frames.

    Args:
        tracklet: Dict with 'track' (frame_idx -> mask mapping) and 'person_id'.
        frames: List of PIL Images (video frames).
        save_dir: Directory to save visualization images.
        overlay: If True, overlay mask on frame; otherwise show mask alone.
        skip_blank_mask: If True, skip frames with None or all-zero masks.
    """
    os.makedirs(save_dir, exist_ok=True)

    track = tracklet["track"]  # dict: {frame_idx: mask, ...}
    person_id = tracklet["person_id"]

    for frame_idx, mask in track.items():
        frame_idx = int(frame_idx)
        frame = frames[frame_idx]

        if skip_blank_mask and (mask is None or mask.sum() == 0):
            continue

        if overlay:
            if mask is None:
                vis_image = frame.copy()
            else:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)  # [1, H, W]
                vis_image = overlay_masks(frame, mask)
        else:
            if mask is None:
                vis_image = Image.new("RGB", frame.size, (0, 0, 0))
            else:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                if mask.dtype == bool:
                    mask_uint8 = (255 * mask.astype(np.float32)).astype(
                        np.uint8
                    )
                elif mask.max() <= 1.0:
                    mask_uint8 = (255 * mask).astype(np.uint8)
                else:
                    mask_uint8 = mask.astype(np.uint8)
                vis_image = Image.fromarray(mask_uint8, mode="L").convert(
                    "RGB"
                )

        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        vis_image.save(save_path)

    logger.info(
        "Saved tracklet visualization for person %s to %s",
        person_id,
        save_dir,
    )
