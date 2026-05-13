"""Audio file I/O utilities."""

import logging
from typing import Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_wav_tensor(
    audio_file: str,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """Load a WAV file as a tensor, converting to mono and resampling.

    Args:
        audio_file: Path to the WAV file.
        target_sr: Target sample rate for resampling.

    Returns:
        Tuple of (waveform [1, T], sample_rate).
    """
    waveform, sr = torchaudio.load(audio_file)  # [C, T]

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    logger.debug(
        "Loaded audio %s: shape=%s, sr=%d", audio_file, waveform.shape, sr
    )
    return waveform, sr


def slice_wav_by_time(
    waveform: torch.Tensor,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> torch.Tensor:
    """Slice a waveform tensor by time range [start_time, end_time).

    Args:
        waveform: Audio tensor of shape [1, T].
        sample_rate: Sample rate of the waveform.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Sliced waveform tensor.
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    return waveform[:, start_sample:end_sample]
