"""Automatic Speech Recognition skill using Whisper."""

import logging
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)


def load_asr_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> Any:
    """Load Whisper ASR pipeline (call once, reuse).

    Args:
        model_id: Path or HuggingFace model ID for Whisper.
        device: Inference device (e.g. "cuda:0"). Auto-detected if None.

    Returns:
        HuggingFace ASR pipeline instance.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    logger.info("Loading ASR model from %s on %s", model_id, device)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    logger.info("ASR pipeline loaded successfully")
    return pipe


def speech_recognition(
    wav_path: str,
    pipe: Any,
    with_time: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Run ASR on a single wav file.

    Args:
        wav_path: Path to the input wav file.
        pipe: Loaded ASR pipeline from load_asr_pipeline().
        with_time: If True, return timestamps with the transcription.

    Returns:
        ASR result (str or dict with timestamps).
    """
    logger.debug("Running ASR on %s (with_time=%s)", wav_path, with_time)
    result = pipe(wav_path, return_timestamps=with_time)
    return result
