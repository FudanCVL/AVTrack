"""Base agent for audio-visual person tracking."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.prompts import GLOBAL_WINDOW_PROMPT, LOCAL_WINDOW_PROMPT
from src.skills.asr import load_asr_pipeline, speech_recognition
from src.skills.speaker import load_similarity_model, speaker_cosine_similarity
from src.skills.tracker import load_sam3_video, track_objects_in_video
from src.skills.vlm import Qwen3VLChat, Qwen3VLPlusChat
from src.utils.io import load_wav_tensor, slice_wav_by_time
from src.utils.matching import calculate_iou, get_mask_area, round_half_up
from src.utils.text import parse_vlm_response, truncate_repetitive_text
from src.utils.visualization import vis_instance_tracklet

logger = logging.getLogger(__name__)


class BaseAgent:
    """Agent for audio-visual person tracking using speech recognition and vision models."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the agent by loading all required models.

        Args:
            cfg: Full Hydra config (must contain model.*, agent.*, device, output_dir).
        """
        self.cfg = cfg
        device: str = cfg.device

        self.vision_tracker, self.vision_processor, self.vision_device = load_sam3_video(
            cfg.model.tracker.path, device=device,
        )
        self.asr_pipeline = load_asr_pipeline(cfg.model.asr.path, device=device)

        vlm_type: str = cfg.model.vlm.get("type", "local")
        if vlm_type == "api":
            self.vlm_model = Qwen3VLPlusChat()
        else:
            self.vlm_model = Qwen3VLChat(model_path=cfg.model.vlm.path, device_map=device)

        self.similarity_model = load_similarity_model(cfg.model.speaker.path, device=device)
        self.output_vis_dir: str = cfg.output_dir

    def inference(self, frame_paths: List[str], audio_file: str) -> List[dict]:
        """Run the 4-stage audio-visual person tracking pipeline.

        Args:
            frame_paths: List of paths to video frames.
            audio_file: Path to audio file corresponding to the video.

        Returns:
            List of tracklets with person_id and track (dict of frame_idx -> mask).
        """
        self.video_name: str = audio_file.split("/")[-1].split(".")[0]
        waveform, sample_rate = load_wav_tensor(audio_file)

        # Stage 1: Speech preprocessing
        speech_chunks = self._preprocess_speech(audio_file, waveform, sample_rate)

        # Stage 2: Vision preprocessing
        frames, all_frame_boxes, all_frame_masks, video_fps = self._preprocess_vision(
            frame_paths, waveform, sample_rate,
        )

        # Stage 3: Local window analysis
        local_results = self._process_local_windows(
            speech_chunks, frames, all_frame_boxes, all_frame_masks, video_fps,
        )

        # Stage 4: Global window analysis
        return self._process_global_window(
            frames=frames,
            all_chunk_masks=local_results["chunk_masks"],
            key_frame_indices=local_results["key_frame_indices"],
            all_chunk_frame_indices=local_results["chunk_frame_indices"],
            video_name=self.video_name,
        )

    def _preprocess_speech(
        self, audio_file: str, waveform: torch.Tensor, sample_rate: int,
    ) -> List[dict]:
        """Run ASR and optional compression on the audio."""
        transcript = speech_recognition(audio_file, self.asr_pipeline, with_time=True)
        speech_chunks: List[dict] = transcript["chunks"]

        if self.cfg.agent.compression.enabled:
            threshold: float = self.cfg.agent.compression.threshold
            speech_chunks = self._compress_speech_chunks(speech_chunks, waveform, sample_rate, threshold)
            logger.info("Compressed speech chunks [%d]: %s", len(speech_chunks), speech_chunks)

        return speech_chunks

    def _preprocess_vision(
        self, frame_paths: List[str], waveform: torch.Tensor, sample_rate: int,
    ) -> Tuple[List[Image.Image], List[Any], List[Any], float]:
        """Load frames, compute FPS, and run SAM3 tracking."""
        frames = [Image.open(fp) for fp in frame_paths]
        audio_dur = waveform.shape[1] / sample_rate
        video_fps = len(frames) / audio_dur if audio_dur > 0 else 1.0
        logger.info("Video FPS: %.2f, Total frames: %d, Audio duration: %.2fs",
                     video_fps, len(frames), audio_dur)

        text_prompt: str = self.cfg.model.tracker.get("text_prompt", "person")
        outputs_per_frame = track_objects_in_video(
            frames=frames, model=self.vision_tracker,
            processor=self.vision_processor, device=self.vision_device,
            text_prompt=text_prompt,
        )
        sorted_idxs = sorted(outputs_per_frame.keys())
        all_frame_boxes = [outputs_per_frame[i]["boxes"] for i in sorted_idxs]
        all_frame_masks = [outputs_per_frame[i]["masks"] for i in sorted_idxs]
        return frames, all_frame_boxes, all_frame_masks, video_fps

    def _process_local_windows(
        self,
        speech_chunks: List[dict],
        frames: List[Image.Image],
        all_frame_boxes: List[Any],
        all_frame_masks: List[Any],
        video_fps: float,
    ) -> Dict[str, list]:
        """Process each speech chunk to identify the speaking person in corresponding frames.

        Returns:
            Dict with keys: chunk_bboxes, chunk_masks, key_frame_indices, chunk_frame_indices.
        """
        max_frame_count: int = self.cfg.inference.get("max_frame_count", 30)
        bbox_scale: int = self.cfg.agent.thresholds.vlm_bbox_scale

        all_chunk_bboxes: List[list] = []
        all_chunk_masks: List[list] = []
        key_frame_indices: List[int] = []
        all_chunk_frame_indices: List[List[int]] = []

        for chunk_idx, chunk in enumerate(
            tqdm(speech_chunks, total=len(speech_chunks), desc="Processing speech chunks")
        ):
            start_time_sec = chunk["timestamp"][0]
            end_time_sec = chunk["timestamp"][1]
            chunk_text = chunk["text"]

            start_frame_idx = int(round_half_up(start_time_sec * video_fps))
            end_frame_idx = int(round_half_up(end_time_sec * video_fps))
            if start_frame_idx == end_frame_idx:
                start_frame_idx -= 1

            end_frame_idx = min(end_frame_idx, start_frame_idx + max_frame_count)
            chunk_frames = frames[start_frame_idx:end_frame_idx]
            chunk_frame_indices = list(range(start_frame_idx, end_frame_idx))

            # Prepare VLM input: prompt + speech text + frames
            vlm_input: List[dict] = [
                {"type": "text", "text": LOCAL_WINDOW_PROMPT},
                {"type": "text", "text": f"Speech: {chunk_text}"},
            ]
            for frame in chunk_frames:
                vlm_input.append({"type": "image", "image": frame})

            vlm_response = self.vlm_model.chat(
                [{"role": "user", "content": vlm_input}], max_new_tokens=2048,
            )
            vlm_response = parse_vlm_response(vlm_response)
            logger.info("VLM response [local window]:\n%s", vlm_response)

            vlm_bboxes: list = vlm_response["boxes"]

            # Fill missing frames with [0,0,0,0] bboxes
            existing_ids = {b["frame_id"] for b in vlm_bboxes}
            for fi in range(len(chunk_frames)):
                if fi not in existing_ids:
                    vlm_bboxes.append({"frame_id": fi, "bbox": [0, 0, 0, 0]})
            vlm_bboxes = sorted(
                [b for b in vlm_bboxes if 0 <= b["frame_id"] < len(chunk_frames)],
                key=lambda x: x["frame_id"],
            )
            assert len(vlm_bboxes) == len(chunk_frames), (
                f"VLM bboxes length {len(vlm_bboxes)} != chunk frames length {len(chunk_frames)}"
            )

            # Convert bboxes from 0-bbox_scale to pixel coordinates
            ref = chunk_frames[0]
            vlm_bboxes_px = [
                [int(b[0] / bbox_scale * ref.width), int(b[1] / bbox_scale * ref.height),
                 int(b[2] / bbox_scale * ref.width), int(b[3] / bbox_scale * ref.height)]
                for b in (r["bbox"] for r in vlm_bboxes)
            ]

            chunk_sam3_bboxes = all_frame_boxes[start_frame_idx:end_frame_idx]
            chunk_sam3_masks = all_frame_masks[start_frame_idx:end_frame_idx]

            # Match VLM bboxes with SAM3 detections using IoU
            matched_bboxes: list = []
            matched_masks: list = []
            for fi, vlm_bbox in enumerate(vlm_bboxes_px):
                if vlm_bbox == [0, 0, 0, 0]:
                    matched_bboxes.append([0, 0, 0, 0])
                    matched_masks.append(None)
                else:
                    sam_bbs = chunk_sam3_bboxes[fi]
                    ious = [calculate_iou(vlm_bbox, sb) for sb in sam_bbs]
                    if not ious:
                        matched_bboxes.append([0, 0, 0, 0])
                        matched_masks.append(None)
                        continue
                    ious = [s.item() if isinstance(s, torch.Tensor) else s for s in ious]
                    best = int(np.argmax(ious))
                    matched_bboxes.append(sam_bbs[best])
                    if fi < len(chunk_sam3_masks) and best < len(chunk_sam3_masks[fi]):
                        matched_masks.append(chunk_sam3_masks[fi][best])
                    else:
                        matched_masks.append(None)

            all_chunk_bboxes.append(matched_bboxes)
            all_chunk_masks.append(matched_masks)
            all_chunk_frame_indices.append(chunk_frame_indices)

            # Key frame: frame with largest mask area
            areas = [get_mask_area(m) for m in matched_masks]
            if areas:
                kf_local = int(np.argmax(areas))
                key_frame_indices.append(kf_local + start_frame_idx)
            else:
                key_frame_indices.append(start_frame_idx)

        return {
            "chunk_bboxes": all_chunk_bboxes,
            "chunk_masks": all_chunk_masks,
            "key_frame_indices": key_frame_indices,
            "chunk_frame_indices": all_chunk_frame_indices,
        }

    def _process_global_window(
        self,
        frames: List[Image.Image],
        all_chunk_masks: List[list],
        key_frame_indices: List[int],
        all_chunk_frame_indices: List[List[int]],
        video_name: str,
    ) -> List[dict]:
        """Analyze relationships between tracklets to group key frames by person identity."""
        local_tracklets = all_chunk_masks
        logger.info("Number of local window tracklets: %d", len(local_tracklets))

        valid_kf = [i for i in key_frame_indices if 0 <= i < len(frames)]
        if not valid_kf:
            logger.warning("No valid key frames found for global analysis")
            return []

        key_frames = [frames[i] for i in valid_kf]

        # Prepare VLM input for global analysis
        vlm_input: List[dict] = [{"type": "text", "text": GLOBAL_WINDOW_PROMPT}]
        for frame in key_frames:
            vlm_input.append({"type": "image", "image": frame})

        raw_resp = self.vlm_model.chat(
            [{"role": "user", "content": vlm_input}], max_new_tokens=2048,
        )
        logger.info("Raw VLM response [global window]:\n%s", raw_resp)
        parsed = parse_vlm_response(raw_resp)
        logger.info("Parsed VLM response [global window]:\n%s", parsed)

        person_mapping = parsed["persons"]
        logger.info("Chunk frame indices: %s", all_chunk_frame_indices)

        global_tracklets: List[dict] = []
        for person_id, kf_indices in person_mapping.items():
            logger.info("Processing %s: key frame indices %s", person_id, kf_indices)
            person_track: Dict[int, Any] = {}

            for kf_idx in kf_indices:
                if kf_idx < 0 or kf_idx >= len(local_tracklets):
                    logger.warning("key_frame_idx %d out of bounds, skipping", kf_idx)
                    continue
                if kf_idx >= len(all_chunk_frame_indices):
                    logger.warning("key_frame_idx %d out of bounds for chunk frame indices, skipping", kf_idx)
                    continue

                local_track = local_tracklets[kf_idx]
                vid_indices = all_chunk_frame_indices[kf_idx]
                logger.info("Frame indices in video: %s", vid_indices)
                logger.info("Local track length: %d", len(local_track))

                for ti in range(min(len(local_track), len(vid_indices))):
                    fiv = vid_indices[ti]
                    if 0 <= fiv < len(frames):
                        person_track[fiv] = local_track[ti]
                    else:
                        logger.warning("frame_idx_in_video %d out of bounds [0, %d), skipping", fiv, len(frames))

            # Fill missing frames with blank masks
            for fi in range(len(frames)):
                if fi not in person_track:
                    w, h = frames[fi].size
                    person_track[fi] = np.zeros((h, w), dtype=np.float32)

            person_track = dict(sorted(person_track.items(), key=lambda x: int(x[0])))
            logger.info("Person track sorted frame indices: %s", list(person_track.keys()))

            tracklet = {"person_id": person_id, "track": person_track}
            save_dir = f"{self.output_vis_dir}/{video_name}/{person_id}"
            vis_instance_tracklet(tracklet, frames, save_dir, overlay=False, skip_blank_mask=True)
            global_tracklets.append(tracklet)

        return global_tracklets

    def _compress_speech_chunks(
        self,
        speech_chunks: List[dict],
        waveform: torch.Tensor,
        sample_rate: int,
        threshold: float = 0.35,
    ) -> List[dict]:
        """Merge consecutive speech chunks by text dedup and speaker similarity.

        Args:
            speech_chunks: List of speech chunks from Whisper.
            waveform: Audio waveform tensor.
            sample_rate: Audio sample rate.
            threshold: Cosine similarity threshold for merging chunks.

        Returns:
            Compressed list of speech chunks.
        """
        audio_dur: float = waveform.shape[1] / sample_rate

        # (1) Text dedup: merge adjacent chunks with identical text
        i = 1
        while i < len(speech_chunks):
            if speech_chunks[i]["text"] == speech_chunks[i - 1]["text"]:
                end_time = min(speech_chunks[i]["timestamp"][1], audio_dur)
                speech_chunks[i]["timestamp"] = (speech_chunks[i - 1]["timestamp"][0], end_time)
                speech_chunks.pop(i - 1)
            else:
                i += 1

        # (2) Speaker similarity compression
        def _speaker_sim(c1: dict, c2: dict) -> float:
            s1 = slice_wav_by_time(waveform, sample_rate, c1["timestamp"][0], c1["timestamp"][1])
            s2 = slice_wav_by_time(waveform, sample_rate, c2["timestamp"][0], c2["timestamp"][1])
            return speaker_cosine_similarity(s1, s2, self.similarity_model)

        if not speech_chunks:
            return []

        compressed: List[dict] = []
        cur = speech_chunks[0].copy()

        for i in range(1, len(speech_chunks)):
            nxt = speech_chunks[i]
            if not nxt.get("text", "").strip():
                continue
            try:
                cos_sim = _speaker_sim(cur, nxt)
            except Exception:
                cos_sim = -1.0

            if cos_sim > threshold:
                end_time = min(nxt["timestamp"][1], audio_dur)
                cur["timestamp"] = (cur["timestamp"][0], end_time)
                cur["text"] = cur["text"].rstrip() + " " + nxt["text"].lstrip()
            else:
                compressed.append(cur)
                cur = nxt.copy()

        if cur.get("text", "").strip():
            compressed.append(cur)

        # Clamp all timestamps to audio duration
        prev_end = 0.0
        for chunk in compressed:
            st = chunk["timestamp"][0]
            if st is None:
                st = prev_end
            st = max(0.0, min(st, audio_dur))
            et = max(st, min(chunk["timestamp"][1] or audio_dur, audio_dur))
            chunk["timestamp"] = (st, et)
            prev_end = et
            chunk["text"] = truncate_repetitive_text(chunk["text"])

        return compressed
