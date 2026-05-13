"""Speaker similarity skill using SpeechBrain ECAPA-TDNN."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)


def load_similarity_model(
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
    device: Optional[str] = None,
) -> EncoderClassifier:
    """Load SpeechBrain speaker encoder (call once, reuse).

    Args:
        model_id: SpeechBrain model source identifier.
        device: Inference device. Auto-detected if None.

    Returns:
        Loaded EncoderClassifier on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading speaker similarity model from %s on %s", model_id, device)
    classifier = EncoderClassifier.from_hparams(source=model_id).to(device)
    logger.info("Speaker similarity model loaded successfully")
    return classifier


def speaker_embedding_from_waveform(
    waveform: torch.Tensor,
    classifier: EncoderClassifier,
) -> torch.Tensor:
    """Extract speaker embedding from a waveform.

    Args:
        waveform: Audio waveform tensor of shape [1, T].
        classifier: Loaded EncoderClassifier.

    Returns:
        Speaker embedding tensor of shape [192].
    """
    with torch.no_grad():
        emb = classifier.encode_batch(waveform)  # [1, 1, 192]
    return emb.squeeze(0).squeeze(0)


def speaker_cosine_similarity(
    seg1: torch.Tensor,
    seg2: torch.Tensor,
    classifier: EncoderClassifier,
) -> torch.Tensor:
    """Compute cosine similarity between two speech segments.

    Args:
        seg1: First audio waveform tensor.
        seg2: Second audio waveform tensor.
        classifier: Loaded EncoderClassifier.

    Returns:
        Cosine similarity scalar tensor.
    """
    emb1 = speaker_embedding_from_waveform(seg1, classifier)
    emb2 = speaker_embedding_from_waveform(seg2, classifier)

    sim = F.cosine_similarity(emb1, emb2, dim=0)
    return sim
