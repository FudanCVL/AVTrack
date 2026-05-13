"""Face detection and clustering skill using DeepFace."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _compute_cosine_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.6,
) -> Dict[int, List[int]]:
    """Cluster face embeddings by cosine similarity with greedy assignment.

    Args:
        embeddings: Face embedding matrix of shape (N, M).
        threshold: Similarity threshold for grouping faces.

    Returns:
        Dict mapping person_id to list of face indices.
    """
    similarity_matrix = cosine_similarity(embeddings)

    person_frames: Dict[int, List[int]] = defaultdict(list)
    person_id: int = 0
    assigned_person: List[int] = [-1] * len(embeddings)

    for i in range(len(embeddings)):
        if assigned_person[i] != -1:
            continue

        assigned_person[i] = person_id
        person_frames[person_id].append(i)

        for j in range(i + 1, len(embeddings)):
            if assigned_person[j] == -1:
                similarity: float = similarity_matrix[i, j]
                if similarity >= threshold:
                    assigned_person[j] = person_id
                    person_frames[person_id].append(j)

        person_id += 1

    return dict(person_frames)


def face_frame_pairing(
    image_paths: List[str],
    model_name: str = "Facenet512",
    detector_backend: str = "retinaface",
    distance_threshold: float = 0.6,
) -> Dict[int, List[int]]:
    """Detect faces across frames and group by identity.

    Args:
        image_paths: List of image file paths (index = frame_id).
        model_name: DeepFace recognition model name.
        detector_backend: DeepFace face detector backend.
        distance_threshold: Cosine similarity threshold for clustering.

    Returns:
        Dict mapping person_id to list of frame indices.
    """
    logger.info(
        "Running face-frame pairing on %d images (model=%s, detector=%s)",
        len(image_paths),
        model_name,
        detector_backend,
    )

    faces: List[Dict] = []

    # 1. Extract all face embeddings
    for frame_idx, img_path in enumerate(image_paths):
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
        )

        for r in reps:
            faces.append({
                "embedding": np.array(r["embedding"]),
                "frame_idx": frame_idx,
            })

    if len(faces) == 0:
        logger.warning("No faces detected in any frame")
        return {}

    embeddings = np.vstack([f["embedding"] for f in faces])

    person_frames = _compute_cosine_similarity(embeddings, distance_threshold)

    logger.info("Face pairing complete: %d identities found", len(person_frames))
    return person_frames
