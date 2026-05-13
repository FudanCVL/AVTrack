"""Video object tracking skill using SAM3 Video."""

import logging
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor

logger = logging.getLogger(__name__)


def load_sam3_video(
    model_id: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Sam3VideoModel, Sam3VideoProcessor, str]:
    """Load SAM3 Video model and processor (call once, reuse).

    Args:
        model_id: Path or HuggingFace model ID for SAM3.
        device: Inference device. Auto-detected if None.
        dtype: Model dtype for inference.

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAM3 Video model from %s on %s", model_id, device)

    model = Sam3VideoModel.from_pretrained(model_id).to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(model_id)

    logger.info("SAM3 Video model loaded successfully")
    return model, processor, device


def track_objects_in_video(
    frames: List[Image.Image],
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: str,
    text_prompt: str = "human",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, dict]:
    """Track objects in video frames using SAM3 Video with text prompt.

    Args:
        frames: List of PIL Image frames.
        model: Loaded SAM3 Video model.
        processor: SAM3 Video Processor.
        device: Inference device.
        text_prompt: Text prompt (e.g. "human", "person", "car").
        dtype: Inference dtype.

    Returns:
        Dict mapping frame_idx to detection results:
            {
                frame_idx: {
                    'masks': Tensor[N, H, W],
                    'boxes': Tensor[N, 4],
                    'scores': Tensor[N],
                    'object_ids': Tensor[N],
                }
            }
    """
    logger.info(
        "Tracking '%s' across %d frames", text_prompt, len(frames)
    )

    # Initialize video session
    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    # Add text prompt
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )

    outputs_per_frame: Dict[int, dict] = {}

    with torch.no_grad():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(frames),
        ):
            processed = processor.postprocess_outputs(
                inference_session, model_outputs
            )
            outputs_per_frame[model_outputs.frame_idx] = processed

    logger.info("Tracking complete: %d frames processed", len(outputs_per_frame))
    return outputs_per_frame
