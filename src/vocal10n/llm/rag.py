"""Vector-based term retrieval for large glossaries (Phase 4.4).

Uses ``sentence-transformers`` (CPU-only, ~80 MB) to embed glossary
terms and ``faiss-cpu`` for fast nearest-neighbour search.  This
replaces the O(n) pinyin scan in :class:`Corrector` when the glossary
exceeds a configurable threshold (default 100 terms).

The index is built once at startup and cached to disk alongside the
glossary files (``knowledge_base/*.index``, ``*.npy`` — gitignored).

Usage::

    rag = RAGIndex()
    rag.add_terms([("科技小院", "Science and Technology Backyard"), ...])
    rag.build()
    results = rag.search("科技巧院遍地开花", top_k=5)
    # → [("科技小院", "Science and Technology Backyard", 0.92), ...]
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and optional
_faiss = None
_SentenceTransformer = None

_MODEL_NAME = "all-MiniLM-L6-v2"  # ~80 MB, CPU-only, multilingual-ish
_CACHE_DIR = None  # Set by caller or auto-detected


def _ensure_imports() -> bool:
    """Lazy-import faiss and sentence-transformers.  Returns True if available."""
    global _faiss, _SentenceTransformer
    if _faiss is not None and _SentenceTransformer is not None:
        return True
    try:
        import faiss as _f
        _faiss = _f
    except ImportError:
        logger.warning("faiss-cpu not installed — RAG disabled")
        return False
    try:
        from sentence_transformers import SentenceTransformer as _ST
        _SentenceTransformer = _ST
    except ImportError:
        logger.warning("sentence-transformers not installed — RAG disabled")
        return False
    return True


class RAGIndex:
    """Embeds glossary terms and provides fast vector search."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._terms: list[str] = []           # source terms in insertion order
        self._translations: list[str] = []    # parallel translations list
        self._embeddings: np.ndarray | None = None
        self._index = None                    # faiss.IndexFlatIP
        self._model = None                    # SentenceTransformer
        self._available = False
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._content_hash: str = ""          # hash of all terms for cache invalidation

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available and self._index is not None

    @property
    def term_count(self) -> int:
        return len(self._terms)

    def add_terms(self, terms: list[tuple[str, str]]) -> None:
        """Add ``(source_term, translation)`` pairs.  Call ``build()`` after."""
        for src, tgt in terms:
            if src and src not in self._terms:
                self._terms.append(src)
                self._translations.append(tgt)

    def build(self) -> bool:
        """Embed all terms and build the FAISS index.

        Returns True on success, False if dependencies missing or no terms.
        """
        if not self._terms:
            logger.info("RAG: no terms to index")
            return False

        if not _ensure_imports():
            return False

        # Check cache
        self._content_hash = self._compute_hash()
        if self._try_load_cache():
            self._available = True
            logger.info("RAG: loaded cached index (%d terms)", len(self._terms))
            return True

        # Load embedding model (CPU only)
        logger.info("RAG: embedding %d terms with %s (CPU)...", len(self._terms), _MODEL_NAME)
        try:
            self._model = _SentenceTransformer(_MODEL_NAME, device="cpu")
        except Exception:
            logger.exception("RAG: failed to load embedding model")
            return False

        # Embed all terms
        try:
            embeddings = self._model.encode(
                self._terms,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=64,
            )
            self._embeddings = np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception("RAG: embedding failed")
            return False

        # Build FAISS index (inner product on normalised vectors = cosine similarity)
        dim = self._embeddings.shape[1]
        self._index = _faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

        self._available = True
        self._save_cache()
        logger.info("RAG: index built — %d terms, dim=%d", len(self._terms), dim)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 8,
               min_score: float = 0.3) -> list[tuple[str, str, float]]:
        """Return the top-K most similar glossary entries to *query*.

        Returns list of ``(source_term, translation, score)`` sorted by
        descending similarity.
        """
        if not self._available or self._index is None or self._model is None:
            return []

        try:
            q_emb = self._model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            q_emb = np.array(q_emb, dtype=np.float32)
        except Exception:
            logger.exception("RAG: query encoding failed")
            return []

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(q_emb, k)
        results: list[tuple[str, str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            results.append((
                self._terms[idx],
                self._translations[idx],
                float(score),
            ))
        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _compute_hash(self) -> str:
        """Hash all term content for cache invalidation."""
        content = "\n".join(f"{t}|{tr}" for t, tr in zip(self._terms, self._translations))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _cache_paths(self) -> tuple[Path, Path, Path] | None:
        """Return (index_path, embeddings_path, meta_path) or None."""
        if not self._cache_dir:
            return None
        return (
            self._cache_dir / "rag_index.faiss",
            self._cache_dir / "rag_embeddings.npy",
            self._cache_dir / "rag_meta.txt",
        )

    def _try_load_cache(self) -> bool:
        """Try to load index from cache.  Returns True if successful."""
        paths = self._cache_paths()
        if not paths:
            return False
        idx_path, emb_path, meta_path = paths

        if not all(p.exists() for p in paths):
            return False

        # Check hash
        try:
            cached_hash = meta_path.read_text(encoding="utf-8").strip()
            if cached_hash != self._content_hash:
                logger.info("RAG: cache invalidated (terms changed)")
                return False
        except Exception:
            return False

        try:
            self._embeddings = np.load(str(emb_path))
            self._index = _faiss.read_index(str(idx_path))
            # Load model for query encoding
            self._model = _SentenceTransformer(_MODEL_NAME, device="cpu")
            return True
        except Exception:
            logger.warning("RAG: cache load failed, will rebuild")
            return False

    def _save_cache(self) -> None:
        """Persist index to disk for fast reload."""
        paths = self._cache_paths()
        if not paths or self._index is None or self._embeddings is None:
            return

        idx_path, emb_path, meta_path = paths
        try:
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            _faiss.write_index(self._index, str(idx_path))
            np.save(str(emb_path), self._embeddings)
            meta_path.write_text(self._content_hash, encoding="utf-8")
            logger.info("RAG: cache saved to %s", idx_path.parent)
        except Exception:
            logger.warning("RAG: failed to save cache", exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release all resources."""
        self._terms.clear()
        self._translations.clear()
        self._embeddings = None
        self._index = None
        self._model = None
        self._available = False
