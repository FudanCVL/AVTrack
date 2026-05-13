"""Face-detection-based agent for global window analysis."""

import logging
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from src.skills.face import face_frame_pairing
from src.utils.visualization import vis_instance_tracklet

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FaceAgent(BaseAgent):
    """Agent that uses face detection instead of VLM for global person grouping.

    Overrides only ``_process_global_window`` -- all other stages are inherited
    from :class:`BaseAgent`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize FaceAgent (identical model loading to BaseAgent).

        Args:
            cfg: Full Hydra config.
        """
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Stage 4 override: face-detection global window
    # ------------------------------------------------------------------

    def _process_global_window(
        self,
        frames: List[Image.Image],
        all_chunk_masks: List[list],
        key_frame_indices: List[int],
        all_chunk_frame_indices: List[List[int]],
        video_name: str,
    ) -> List[dict]:
        """Group key frames by face identity using DeepFace.

        Args:
            frames: List of video frames (PIL Images).
            all_chunk_masks: List of lists of masks for each chunk.
            key_frame_indices: List of key frame indices (one per chunk).
            all_chunk_frame_indices: List of lists of frame indices for each chunk.
            video_name: Name of the video for saving outputs.

        Returns:
            List of tracklets, each containing person_id and track.
        """
        local_window_tracklets = all_chunk_masks
        logger.info("Number of local window tracklets: %d", len(local_window_tracklets))

        valid_key_frame_indices = [idx for idx in key_frame_indices if 0 <= idx < len(frames)]
        if len(valid_key_frame_indices) == 0:
            logger.warning("No valid key frames found for global analysis")
            return []

        key_frames = [frames[idx] for idx in valid_key_frame_indices]

        # face_frame_pairing expects file paths, so save PIL Images to temp files
        tmp_paths: List[str] = []
        try:
            for pil_img in key_frames:
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                pil_img.save(tmp_path)
                tmp_paths.append(tmp_path)

            person_key_frame_mapping: Dict[int, List[int]] = face_frame_pairing(tmp_paths)
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

        logger.info("Person key frame mapping: %s", person_key_frame_mapping)
        logger.info("Chunk frame indices: %s", all_chunk_frame_indices)

        global_tracklets: List[dict] = []
        for person_id, key_frame_indices_for_person in person_key_frame_mapping.items():
            logger.info("Processing person_%d: key frame indices %s", person_id, key_frame_indices_for_person)

            person_track: Dict[int, Any] = {}
            for key_frame_idx in key_frame_indices_for_person:
                if key_frame_idx < 0 or key_frame_idx >= len(local_window_tracklets):
                    logger.warning("key_frame_idx %d out of bounds, skipping", key_frame_idx)
                    continue
                if key_frame_idx >= len(all_chunk_frame_indices):
                    logger.warning(
                        "key_frame_idx %d out of bounds for chunk frame indices, skipping",
                        key_frame_idx,
                    )
                    continue

                local_track = local_window_tracklets[key_frame_idx]
                frame_indices_in_video = all_chunk_frame_indices[key_frame_idx]

                logger.info("Frame indices in video: %s", frame_indices_in_video)
                logger.info("Local track length: %d", len(local_track))

                min_length = min(len(local_track), len(frame_indices_in_video))
                for track_idx in range(min_length):
                    frame_idx_in_video = frame_indices_in_video[track_idx]
                    if 0 <= frame_idx_in_video < len(frames):
                        person_track[frame_idx_in_video] = local_track[track_idx]
                    else:
                        logger.warning(
                            "frame_idx_in_video %d out of bounds [0, %d), skipping",
                            frame_idx_in_video, len(frames),
                        )

            # Fill missing frames with blank masks
            for frame_idx in range(len(frames)):
                if frame_idx not in person_track:
                    frame_width, frame_height = frames[frame_idx].size
                    blank_mask = np.zeros((frame_height, frame_width), dtype=np.float32)
                    person_track[frame_idx] = blank_mask

            person_track = dict(sorted(person_track.items(), key=lambda x: int(x[0])))
            logger.info("Person track sorted frame indices: %s", list(person_track.keys()))

            tracklet = {"person_id": person_id, "track": person_track}

            save_dir = f"{self.output_vis_dir}/{video_name}/person_{person_id}"
            vis_instance_tracklet(tracklet, frames, save_dir, overlay=False, skip_blank_mask=True)
            global_tracklets.append(tracklet)

        return global_tracklets
