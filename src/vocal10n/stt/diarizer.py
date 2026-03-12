"""Lightweight real-time speaker diarizer using pyannote speaker embeddings.

Extracts speaker embeddings from audio segments and clusters them into
speaker identities using cosine similarity.  Designed for streaming use
— each segment is tagged independently against a growing speaker bank.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — pyannote is optional
_Model = None


def _get_embedding_model():
    """Lazy-load the pyannote embedding model."""
    global _Model
    if _Model is not None:
        return _Model

    try:
        from pyannote.audio import Model
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding,
        )
    except ImportError:
        logger.error(
            "pyannote-audio not installed. "
            "Install with: pip install pyannote-audio"
        )
        return None

    return PretrainedSpeakerEmbedding, Model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


class SpeakerDiarizer:
    """Real-time speaker tagger using pyannote embeddings.

    Call :meth:`load` to initialise the model, then
    :meth:`identify_speaker` for each audio segment.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        sample_rate: int = 16000,
        max_speakers: int = 10,
    ) -> None:
        self._threshold = similarity_threshold
        self._sample_rate = sample_rate
        self._max_speakers = max_speakers

        self._embedding_model = None
        self._loaded = False
        self._lock = threading.Lock()

        # Speaker bank: list of (label, embedding_vector)
        self._speakers: list[tuple[str, np.ndarray]] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load the speaker embedding model. Returns True on success."""
        if self._loaded:
            return True

        try:
            from pyannote.audio import Inference

            self._embedding_model = Inference(
                "pyannote/embedding",
                window="whole",
            )
            self._loaded = True
            logger.info("Speaker diarizer loaded (pyannote/embedding)")
            return True
        except ImportError:
            logger.error(
                "pyannote-audio not installed — speaker diarization unavailable"
            )
            return False
        except Exception as e:
            logger.exception("Failed to load speaker embedding model: %s", e)
            return False

    def unload(self) -> None:
        """Unload the model and clear speaker bank."""
        with self._lock:
            self._embedding_model = None
            self._speakers.clear()
            self._next_id = 1
            self._loaded = False
        logger.info("Speaker diarizer unloaded")

    def reset_speakers(self) -> None:
        """Clear the speaker bank (new session)."""
        with self._lock:
            self._speakers.clear()
            self._next_id = 1
        logger.info("Speaker bank reset")

    # ------------------------------------------------------------------
    # Speaker identification
    # ------------------------------------------------------------------

    def identify_speaker(self, audio: np.ndarray) -> str:
        """Identify the speaker in *audio* (float32 mono).

        Returns a speaker label like "Speaker 1", "Speaker 2", etc.
        If the model is not loaded, returns an empty string.
        """
        if not self._loaded or self._embedding_model is None:
            return ""

        if len(audio) < self._sample_rate * 0.3:
            # Too short for reliable embedding
            return self._last_speaker_label()

        try:
            embedding = self._extract_embedding(audio)
            if embedding is None:
                return self._last_speaker_label()

            with self._lock:
                return self._match_or_create(embedding)
        except Exception:
            logger.debug("Speaker identification failed", exc_info=True)
            return self._last_speaker_label()

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        """Extract speaker embedding from an audio segment."""
        try:
            # pyannote expects (channel, samples) or {"waveform": ..., "sample_rate": ...}
            waveform = audio.reshape(1, -1)
            import torch

            input_data = {
                "waveform": torch.from_numpy(waveform).float(),
                "sample_rate": self._sample_rate,
            }
            embedding = self._embedding_model(input_data)

            if isinstance(embedding, np.ndarray):
                return embedding.flatten()

            # Handle torch tensor
            return embedding.detach().cpu().numpy().flatten()
        except Exception:
            logger.debug("Embedding extraction failed", exc_info=True)
            return None

    def _match_or_create(self, embedding: np.ndarray) -> str:
        """Match embedding to existing speaker or create new one.

        Must be called with self._lock held.
        """
        best_sim = -1.0
        best_label = ""

        for label, ref_emb in self._speakers:
            sim = _cosine_similarity(embedding, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim >= self._threshold and best_label:
            # Update the speaker embedding with exponential moving average
            for i, (label, ref_emb) in enumerate(self._speakers):
                if label == best_label:
                    self._speakers[i] = (label, 0.8 * ref_emb + 0.2 * embedding)
                    break
            return best_label

        # New speaker
        if len(self._speakers) >= self._max_speakers:
            # Replace oldest speaker beyond limit
            label = f"Speaker {self._next_id}"
            self._speakers.pop(0)
            self._speakers.append((label, embedding))
        else:
            label = f"Speaker {self._next_id}"
            self._speakers.append((label, embedding))

        self._next_id += 1
        logger.info("New speaker detected: %s (total: %d)", label, len(self._speakers))
        return label

    def _last_speaker_label(self) -> str:
        """Return the most recent speaker label, or empty string."""
        if self._speakers:
            return self._speakers[-1][0]
        return ""

    @property
    def speaker_count(self) -> int:
        """Number of distinct speakers detected so far."""
        return len(self._speakers)
