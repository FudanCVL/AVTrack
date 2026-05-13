"""Image segmentation skill using SAM3."""

import logging
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)


def load_sam3_image(
    model_id: str,
    device: Optional[str] = None,
) -> tuple[Sam3Model, Sam3Processor, str]:
    """Load SAM3 image segmentation model and processor (call once, reuse).

    Args:
        model_id: Path or HuggingFace model ID for SAM3.
        device: Inference device. Auto-detected if None.

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAM3 image model from %s on %s", model_id, device)

    model = Sam3Model.from_pretrained(model_id).to(device)
    processor = Sam3Processor.from_pretrained(model_id)

    logger.info("SAM3 image model loaded successfully")
    return model, processor, device


def segment_image_with_text(
    image: Image.Image,
    text_prompt: str,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Segment objects in an image using SAM3 with text prompt.

    Args:
        image: PIL Image (RGB).
        text_prompt: Text prompt (e.g. "ear", "person", "car").
        model: Loaded SAM3Model.
        processor: Sam3Processor.
        device: Inference device.
        threshold: Object confidence threshold.
        mask_threshold: Mask binarization threshold.

    Returns:
        Results dict with keys:
            - masks: Tensor [N, H, W]
            - boxes: Tensor [N, 4] (xyxy)
            - scores: Tensor [N]
    """
    logger.debug("Segmenting image with text prompt: '%s'", text_prompt)

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    logger.info("Segmentation complete: %d objects found", len(results.get("masks", [])))
    return results
