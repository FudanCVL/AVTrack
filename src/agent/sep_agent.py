"""Speech-separation agent that handles overlapping speech via MossFormer."""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm

from src.skills.asr import speech_recognition
from src.skills.separator import SpeechSeparationMossformer
from src.utils.io import slice_wav_by_time
from src.utils.text import truncate_repetitive_text

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SepAgent(BaseAgent):
    """Agent that adds speech separation before ASR for overlapping speech.

    Overrides ``__init__`` (loads MossFormer) and ``_preprocess_speech``
    (separation -> per-source ASR -> merge/dedup -> compression).
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize SepAgent with additional MossFormer separator.

        Args:
            cfg: Full Hydra config (must also contain model.separator.*).
        """
        super().__init__(cfg)
        self.mossformer_separator = SpeechSeparationMossformer(
            model=cfg.model.separator.model_id,
            device=cfg.device,
        )

    # ------------------------------------------------------------------
    # Stage 1 override: speech separation + ASR
    # ------------------------------------------------------------------

    def _preprocess_speech(
        self,
        audio_file: str,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> List[dict]:
        """Run Whisper -> speech separation -> per-source ASR -> merge/dedup.

        Args:
            audio_file: Path to audio file.
            waveform: Audio waveform tensor.
            sample_rate: Audio sample rate.

        Returns:
            Cleaned list of speech chunks with timestamps and text.
        """
        # Initial Whisper pass on full audio
        initial_transcript = speech_recognition(audio_file, self.asr_pipeline, with_time=True)
        initial_chunks: List[dict] = initial_transcript["chunks"]
        logger.info("Initial speech chunks from Whisper [%d]: %s", len(initial_chunks), initial_chunks)

        # Speech separation and re-recognition
        speech_chunks = self._process_speech_separation(
            initial_chunks=initial_chunks,
            waveform=waveform,
            sample_rate=sample_rate,
            audio_file=audio_file,
        )
        logger.info(
            "Speech chunks after separation and merge [%d]: %s",
            len(speech_chunks), speech_chunks,
        )

        # Speaker similarity compression (inherited)
        if self.cfg.agent.compression.enabled:
            threshold: float = self.cfg.agent.compression.threshold
            speech_chunks = self._compress_speech_chunks(speech_chunks, waveform, sample_rate, threshold)

        # Additional cleaning passes specific to sep agent
        speech_chunks = self._filter_meaningless_chunks(speech_chunks)
        speech_chunks = self._merge_text_containment_chunks(speech_chunks)
        speech_chunks = self._merge_same_timestamp_chunks(speech_chunks)
        logger.info(
            "Compressed speech chunks after filtering, text containment merge, "
            "and same timestamp merge [%d]: %s",
            len(speech_chunks), speech_chunks,
        )

        return speech_chunks

    # ------------------------------------------------------------------
    # Speech separation pipeline
    # ------------------------------------------------------------------

    def _process_speech_separation(
        self,
        initial_chunks: List[dict],
        waveform: torch.Tensor,
        sample_rate: int,
        audio_file: str,
    ) -> List[dict]:
        """Process speech separation for each chunk and merge results.

        Args:
            initial_chunks: Initial chunks from Whisper on full audio.
            waveform: Full audio waveform [1, T].
            sample_rate: Audio sample rate (should be 16000).
            audio_file: Path to original audio file.

        Returns:
            Merged and deduplicated list of chunks from all separated sources.
        """
        all_separated_chunks: List[dict] = []
        audio_duration_seconds: float = waveform.shape[1] / sample_rate
        sep_sample_rate: int = self.cfg.model.separator.get("sample_rate", 8000)

        for chunk_idx, chunk in enumerate(
            tqdm(initial_chunks, desc="Processing chunks with speech separation")
        ):
            timestamp = self._validate_and_normalize_timestamp(
                chunk.get("timestamp"), audio_duration_seconds
            )
            if timestamp is None:
                continue

            start_time, end_time = timestamp

            # Extract chunk audio segment
            chunk_waveform = slice_wav_by_time(waveform, sample_rate, start_time, end_time)

            # Resample to separator native rate
            chunk_waveform_sep = torchaudio.functional.resample(
                chunk_waveform, sample_rate, sep_sample_rate
            ).cpu()

            # Create temporary file for this chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_chunk_path = tmp_file.name
                torchaudio.save(tmp_chunk_path, chunk_waveform_sep, sep_sample_rate)

            try:
                # Perform speech separation
                separated_arrays = self.mossformer_separator.separate_to_arrays(
                    tmp_chunk_path,
                    sample_rate=sep_sample_rate,
                    return_float32=True,
                )

                # Process each separated source
                for source_array in separated_arrays:
                    source_waveform_sep = torch.from_numpy(source_array).unsqueeze(0)  # [1, T]

                    # Resample back to ASR rate (16 kHz)
                    source_waveform_16k = torchaudio.functional.resample(
                        source_waveform_sep, sep_sample_rate, 16000
                    ).cpu()

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_source_file:
                        tmp_source_path = tmp_source_file.name
                        torchaudio.save(tmp_source_path, source_waveform_16k, 16000)

                    try:
                        source_transcript = speech_recognition(
                            tmp_source_path, self.asr_pipeline, with_time=True
                        )

                        if "chunks" in source_transcript and source_transcript["chunks"]:
                            for source_chunk in source_transcript["chunks"]:
                                rel_start = source_chunk.get("timestamp", (0, 0))[0] or 0.0
                                rel_end = source_chunk.get("timestamp", (0, 0))[1] or 0.0

                                global_start = start_time + rel_start
                                global_end = start_time + rel_end

                                adjusted_timestamp = self._validate_and_normalize_timestamp(
                                    (global_start, global_end), audio_duration_seconds
                                )

                                if adjusted_timestamp:
                                    adjusted_chunk = {
                                        "text": source_chunk.get("text", "").strip(),
                                        "timestamp": adjusted_timestamp,
                                    }
                                    if adjusted_chunk["text"]:
                                        all_separated_chunks.append(adjusted_chunk)
                    finally:
                        if os.path.exists(tmp_source_path):
                            os.remove(tmp_source_path)
            finally:
                if os.path.exists(tmp_chunk_path):
                    os.remove(tmp_chunk_path)

        # Merge and deduplicate chunks
        merged_chunks = self._merge_and_deduplicate_chunks(all_separated_chunks)
        merged_chunks.sort(key=lambda x: x["timestamp"][0])

        return merged_chunks

    # ------------------------------------------------------------------
    # Timestamp validation
    # ------------------------------------------------------------------

    def _validate_and_normalize_timestamp(
        self,
        timestamp: Optional[Tuple[float, float]],
        audio_duration_seconds: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Validate and normalize a timestamp tuple.

        Args:
            timestamp: Tuple of (start_time, end_time).
            audio_duration_seconds: Optional audio duration to clamp timestamps.

        Returns:
            Validated (start_time, end_time) with start < end, or None.
        """
        if timestamp is None:
            return None

        start_time, end_time = timestamp[0], timestamp[1]

        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = start_time + 0.1

        start_time = max(0.0, float(start_time))
        end_time = max(0.0, float(end_time))

        if audio_duration_seconds is not None:
            start_time = min(start_time, audio_duration_seconds)
            end_time = min(end_time, audio_duration_seconds)

        if start_time >= end_time:
            end_time = start_time + 0.01  # Minimum 10 ms duration

        return (start_time, end_time)

    # ------------------------------------------------------------------
    # Merge and deduplicate
    # ------------------------------------------------------------------

    def _merge_and_deduplicate_chunks(
        self,
        chunks: List[dict],
        overlap_threshold: float = 0.5,
        text_similarity_threshold: float = 0.8,
    ) -> List[dict]:
        """Merge and deduplicate chunks from different separated sources.

        Args:
            chunks: List of chunks from all separated sources.
            overlap_threshold: Time overlap threshold for considering chunks as duplicates.
            text_similarity_threshold: Text similarity threshold (character-based).

        Returns:
            Deduplicated list of chunks.
        """
        if not chunks:
            return []

        def _chunks_overlap(chunk1: dict, chunk2: dict) -> Tuple[bool, float]:
            """Check if two chunks overlap in time."""
            t1 = chunk1.get("timestamp")
            t2 = chunk2.get("timestamp")
            if not t1 or not t2:
                return False, 0.0

            s1, e1 = t1[0], t1[1]
            s2, e2 = t2[0], t2[1]
            if s1 >= e1 or s2 >= e2:
                return False, 0.0

            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start >= overlap_end:
                return False, 0.0

            overlap_dur = overlap_end - overlap_start
            min_dur = min(e1 - s1, e2 - s2)
            ratio = overlap_dur / min_dur if min_dur > 0 else 0.0
            return ratio > overlap_threshold, ratio

        def _text_similar(text1: str, text2: str) -> bool:
            """Simple character overlap similarity check."""
            t1 = "".join(text1.split())
            t2 = "".join(text2.split())
            if not t1 or not t2:
                return False
            s1, s2 = set(t1), set(t2)
            similarity = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0.0
            return similarity > text_similarity_threshold

        # Filter invalid timestamps
        valid_chunks: List[dict] = []
        for chunk in chunks:
            ts = chunk.get("timestamp")
            if ts and len(ts) >= 2 and ts[0] is not None and ts[1] is not None and ts[0] < ts[1]:
                valid_chunks.append(chunk)

        sorted_chunks = sorted(valid_chunks, key=lambda x: x["timestamp"][0])
        merged: List[dict] = []

        for chunk in sorted_chunks:
            if not chunk.get("text", "").strip():
                continue

            is_duplicate = False
            for existing_chunk in merged:
                overlaps, _ = _chunks_overlap(chunk, existing_chunk)
                if overlaps and _text_similar(chunk["text"], existing_chunk["text"]):
                    is_duplicate = True
                    ct = chunk.get("timestamp")
                    et = existing_chunk.get("timestamp")
                    if ct and et:
                        chunk_dur = ct[1] - ct[0]
                        existing_dur = et[1] - et[0]
                        if chunk_dur > existing_dur or len(chunk["text"]) > len(existing_chunk["text"]):
                            merged.remove(existing_chunk)
                            merged.append(chunk)
                            is_duplicate = False
                    break

            if not is_duplicate:
                merged.append(chunk)

        return merged

    # ------------------------------------------------------------------
    # Same-timestamp merge
    # ------------------------------------------------------------------

    def _merge_same_timestamp_chunks(self, chunks: List[dict]) -> List[dict]:
        """Merge chunks that have the same timestamp by concatenating their texts.

        Args:
            chunks: List of chunks to process.

        Returns:
            Merged list of chunks where chunks with identical timestamps are combined.
        """
        if not chunks:
            return []

        timestamp_groups: Dict[Tuple[float, float], List[dict]] = {}
        for chunk in chunks:
            ts = chunk.get("timestamp")
            if not ts or len(ts) < 2:
                continue
            if ts[0] is None or ts[1] is None or ts[0] >= ts[1]:
                continue

            ts_key = (float(ts[0]), float(ts[1]))
            if ts_key not in timestamp_groups:
                timestamp_groups[ts_key] = []
            timestamp_groups[ts_key].append(chunk)

        merged: List[dict] = []
        for ts_key, group_chunks in timestamp_groups.items():
            if len(group_chunks) == 1:
                merged.append(group_chunks[0])
            else:
                texts = [c.get("text", "").strip() for c in group_chunks if c.get("text", "").strip()]
                if texts:
                    merged.append({"text": " ".join(texts), "timestamp": ts_key})

        merged.sort(key=lambda x: x["timestamp"][0])
        return merged

    # ------------------------------------------------------------------
    # Filter meaningless chunks
    # ------------------------------------------------------------------

    def _filter_meaningless_chunks(self, chunks: List[dict]) -> List[dict]:
        """Filter out chunks with meaningless text.

        Removes chunks that are only punctuation, special chars, single common
        interjections, or too short after stripping punctuation.

        Args:
            chunks: List of chunks to filter.

        Returns:
            Filtered list of chunks with meaningful text.
        """
        if not chunks:
            return []

        common_interjections: Set[str] = {
            "oh", "okay", "ok", "hey", "yeah", "yes", "no", "ah", "um", "uh",
            "hmm", "huh", "wow", "ooh", "aah", "ha", "ho", "hi", "hello",
            "thanks", "thank", "please", "sorry", "bye", "goodbye",
        }
        punctuation_only: Set[str] = set(".,!?;:()[]{}\"'-_=+*/\\|@#$%^&*~`")

        filtered: List[dict] = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            text_clean = text.strip()

            # Filter 1: Only punctuation marks
            if all(c in punctuation_only or c.isspace() for c in text_clean):
                continue

            # Filter 2: Mostly non-meaningful characters
            meaningful_chars = sum(
                1 for c in text_clean if c.isalnum() or c in ".,!?;:()[]{}\"'-_ "
            )
            if meaningful_chars < len(text_clean) * 0.3:
                continue

            # Filter 3: Single common interjection
            words = text_clean.split()
            if len(words) == 1:
                word_lower = words[0].lower().rstrip(".,!?;:")
                if word_lower in common_interjections:
                    continue

            # Filter 4: Too short after removing punctuation
            text_no_punct = "".join(c for c in text_clean if c.isalnum())
            if len(text_no_punct) < 2:
                continue

            filtered.append(chunk)

        return filtered

    # ------------------------------------------------------------------
    # Text containment merge
    # ------------------------------------------------------------------

    def _merge_text_containment_chunks(self, chunks: List[dict]) -> List[dict]:
        """Merge adjacent chunks where one text is contained in the other.

        For example, ``"There's still pie."`` (26-29s) followed by ``"Still pie."``
        (29-31s) becomes ``"There's still pie."`` (26-31s).

        Args:
            chunks: List of chunks to process.

        Returns:
            Merged list of chunks.
        """
        if not chunks:
            return []

        def _normalize(text: str) -> str:
            return text.strip().lower()

        merged: List[dict] = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i].copy()
            current_text = current_chunk.get("text", "").strip()
            current_normalized = _normalize(current_text)

            if not current_text:
                i += 1
                continue

            merged_this_round = False
            j = i + 1

            while j < len(chunks):
                next_chunk = chunks[j]
                next_text = next_chunk.get("text", "").strip()
                next_normalized = _normalize(next_text)
                next_ts = next_chunk.get("timestamp")

                if not next_text or not next_ts or len(next_ts) < 2:
                    j += 1
                    continue
                if next_ts[0] is None or next_ts[1] is None or next_ts[0] >= next_ts[1]:
                    j += 1
                    continue

                if next_normalized in current_normalized:
                    current_start = current_chunk["timestamp"][0]
                    current_end = max(current_chunk["timestamp"][1], next_ts[1])
                    if current_start < current_end:
                        current_chunk["timestamp"] = (current_start, current_end)
                        merged_this_round = True
                        j += 1
                    else:
                        break
                elif current_normalized in next_normalized:
                    current_start = current_chunk["timestamp"][0]
                    current_end = max(current_chunk["timestamp"][1], next_ts[1])
                    if current_start < current_end:
                        current_chunk["timestamp"] = (current_start, current_end)
                        current_chunk["text"] = next_text
                        current_normalized = next_normalized
                        merged_this_round = True
                        j += 1
                    else:
                        break
                else:
                    break

            merged.append(current_chunk)

            if merged_this_round:
                i = j
            else:
                i += 1

        return merged
