import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSample:
    """A single video sample with metadata."""
    video_id: str
    video_name: str
    frame_paths: List[str]
    audio_path: str
    width: int
    height: int


class AVTrackDataset:
    """AVTrack dataset loader.

    Args:
        cfg: Data configuration with paths (meta_file, image_dir, audio_dir).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        with open(cfg.meta_file, "r") as f:
            data = json.load(f)
        self.videos = data["videos"]
        self.annotations = data["annotations"]
        logger.info("Loaded %d videos, %d annotations", len(self.videos), len(self.annotations))

    def find_video_by_id(self, video_id: str) -> dict:
        """Find a video by its ID.

        Args:
            video_id: The video ID to search for.

        Returns:
            Video dict from metadata.

        Raises:
            ValueError: If video not found.
        """
        for video in self.videos:
            if video["id"] == video_id:
                return video
        raise ValueError(f"Video not found: {video_id}")

    def find_annotations_by_video_id(self, video_id: str) -> Optional[List[dict]]:
        """Find annotations for a video.

        Args:
            video_id: The video ID to search for.

        Returns:
            List of annotation dicts, or None if not found.
        """
        annotations_list = [a for a in self.annotations if a["video_id"] == video_id]
        if not annotations_list:
            logger.warning("No annotations for video %s", video_id)
            return None
        return annotations_list

    def get_samples(
        self,
        output_dir: str,
        patch_counts: int = 1,
        patch_idx: int = 0,
        subset_file: Optional[str] = None,
    ) -> List[VideoSample]:
        """Get video samples, filtering already processed and applying patch splitting.

        Args:
            output_dir: Output directory to check for already processed videos.
            patch_counts: Total number of patches for distributed processing.
            patch_idx: Index of current patch.
            subset_file: Optional path to a text file with one video name per line.
                If provided, only videos listed in this file will be processed.

        Returns:
            List of VideoSample objects to process.
        """
        # Load subset filter if provided
        subset_names = None
        if subset_file:
            with open(subset_file) as f:
                subset_names = {line.strip() for line in f if line.strip()}
            logger.info("Loaded subset filter: %d video names from %s", len(subset_names), subset_file)

        samples = []
        for video in self.videos:
            video_name = video["file_names"][0].split("/")[0]
            if subset_names is not None and video_name not in subset_names:
                continue
            out_path = Path(output_dir) / video_name
            if out_path.exists():
                continue
            # Check annotations exist
            annots = self.find_annotations_by_video_id(video["id"])
            if annots is None:
                continue
            audio_path = str(Path(self.cfg.audio_dir) / f"{video_name}.wav")
            frame_paths = [str(Path(self.cfg.image_dir) / fn) for fn in video["file_names"]]
            samples.append(VideoSample(
                video_id=video["id"],
                video_name=video_name,
                frame_paths=frame_paths,
                audio_path=audio_path,
                width=video["width"],
                height=video["height"],
            ))

        # Patch splitting for distributed processing
        if patch_counts > 1:
            n = len(samples)
            start = (n * patch_idx) // patch_counts
            end = (n * (patch_idx + 1)) // patch_counts
            samples = samples[start:end]

        logger.info("Returning %d samples (patch %d/%d)", len(samples), patch_idx, patch_counts)
        return samples
