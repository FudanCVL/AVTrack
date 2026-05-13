"""Bounding box and mask matching utilities."""

import logging
import math
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def round_half_up(x: float, n: int = 0) -> float:
    """Round a number using half-up strategy.

    Args:
        x: Number to round.
        n: Number of decimal places.

    Returns:
        Rounded number.
    """
    factor = 10 ** n
    return math.floor(x * factor + 0.5) / factor


def calculate_iou(
    box1: List[float],
    box2: List[float],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box as [x1, y1, x2, y2].
        box2: Second bounding box as [x1, y1, x2, y2].

    Returns:
        IoU value in [0, 1].
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(
        0, min(y2, y4) - max(y1, y3)
    )
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return intersection_area / float(box1_area + box2_area - intersection_area)


def get_mask_area(
    mask: Optional[Union[torch.Tensor, np.ndarray]],
) -> float:
    """Compute the area of a binary mask (number of positive pixels).

    Args:
        mask: Binary mask tensor or array. If None, returns 0.

    Returns:
        Mask area (sum of positive values).
    """
    if mask is None:
        return 0
    # Convert to numpy 0-1 mask, count ones
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask_area = mask.sum()
    return float(mask_area)
