"""Speech separation skill using MossFormer2."""

import logging
from typing import List, Optional

import numpy as np
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logger = logging.getLogger(__name__)


class SpeechSeparationMossformer:
    """MossFormer2-based speech separation model."""

    def __init__(
        self,
        model: str = "iic/speech_mossformer2_separation_temporal_8k",
        device: str = "cuda:0",
    ) -> None:
        """Initialize MossFormer speech separation model (call once, reuse).

        Args:
            model: ModelScope model identifier.
            device: Inference device.
        """
        logger.info("Loading MossFormer separation model: %s on %s", model, device)

        self.separation = pipeline(
            Tasks.speech_separation,
            model=model,
            device=device,
        )

        logger.info("MossFormer separation model loaded successfully")

    def separate(
        self,
        input_path: str,
        output_prefix: str = "output_spk",
        sample_rate: int = 8000,
    ) -> List[str]:
        """Separate speech from an audio file and save results to disk.

        Args:
            input_path: Input audio path (URL or local file).
            output_prefix: Output filename prefix (e.g. "output_spk").
            sample_rate: Output sample rate (default 8000).

        Returns:
            List of saved output file paths.
        """
        logger.debug("Separating audio: %s", input_path)

        result = self.separation(input_path)
        output_files: List[str] = []

        for i, signal in enumerate(result["output_pcm_list"]):
            save_file = f"{output_prefix}{i}.wav"
            sf.write(
                save_file,
                np.frombuffer(signal, dtype=np.int16),
                sample_rate,
            )
            output_files.append(save_file)

        logger.info("Separation complete: %d speakers -> %s", len(output_files), output_files)
        return output_files

    def separate_to_arrays(
        self,
        input_path: str,
        sample_rate: int = 8000,
        return_float32: bool = True,
    ) -> List[np.ndarray]:
        """Separate speech and return audio data as numpy arrays (no file I/O).

        Args:
            input_path: Input audio path (URL or local file).
            sample_rate: Sample rate (default 8000, MossFormer native).
            return_float32: If True, return float32 in [-1.0, 1.0];
                otherwise return int16 in [-32768, 32767].

        Returns:
            List of numpy arrays, one per separated speaker.
        """
        logger.debug("Separating audio to arrays: %s", input_path)

        result = self.separation(input_path)
        separated_arrays: List[np.ndarray] = []

        for signal in result["output_pcm_list"]:
            # signal is bytes-format PCM data (int16)
            audio_int16 = np.frombuffer(signal, dtype=np.int16)

            if return_float32:
                # Convert to float32, normalize to [-1.0, 1.0]
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                separated_arrays.append(audio_float32)
            else:
                # Keep int16 format
                separated_arrays.append(audio_int16)

        logger.info("Separation to arrays complete: %d speakers", len(separated_arrays))
        return separated_arrays
