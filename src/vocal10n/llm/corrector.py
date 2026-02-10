"""Post-STT correction via glossary matching and LLM prompt augmentation.

Phase 4.3 — provides domain-specific correction hints that are injected
into the translation prompt.  No separate LLM call is needed; the
corrector builds a compact glossary snippet that the translator embeds
in the prompt so the LLM can fix STT errors in-context.

Glossary file format (one entry per line):
    source_term|preferred_translation
    科技小院|Science and Technology Courtyard
    # comment lines are ignored
    plain term without translation

If no ``|`` separator, the term is added as a correction hint without
a preferred translation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pypinyin import Style, lazy_pinyin
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False


class Corrector:
    """Glossary-based STT correction and prompt augmentation."""

    def __init__(self) -> None:
        # term → preferred translation (or empty string)
        self._glossary: dict[str, str] = {}
        # pinyin index for fuzzy matching
        self._term_pinyin: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_glossary(self, path: str | Path) -> int:
        """Load glossary entries from *path*.  Returns count loaded."""
        p = Path(path)
        if not p.exists():
            logger.warning("Glossary file not found: %s", p)
            return 0

        count = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                parts = line.split("|", 1)
                term = parts[0].strip()
                translation = parts[1].strip()
            else:
                term = line
                translation = ""
            if term:
                self._glossary[term] = translation
                if _HAS_PYPINYIN:
                    py = "".join(lazy_pinyin(term, style=Style.NORMAL))
                    self._term_pinyin[term] = py
                count += 1

        logger.info("Loaded %d glossary entries from %s", count, p.name)
        return count

    def load_glossary_dir(self, directory: str | Path) -> int:
        """Load all ``.txt`` glossary files from *directory*."""
        d = Path(directory)
        if not d.is_dir():
            return 0
        total = 0
        for f in sorted(d.glob("*.txt")):
            total += self.load_glossary(f)
        return total

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_relevant_terms(self, text: str, max_terms: int = 8) -> list[tuple[str, str]]:
        """Return glossary entries relevant to *text*.

        Matches by substring (exact) and pinyin similarity (fuzzy).
        Returns list of (source_term, preferred_translation) tuples.
        """
        if not self._glossary:
            return []

        matches: list[tuple[str, str, float]] = []  # (term, translation, score)

        for term, translation in self._glossary.items():
            # Exact substring match
            if term in text:
                matches.append((term, translation, 1.0))
                continue

            # Pinyin fuzzy match (for STT errors like 科技巧院 → 科技小院)
            if _HAS_PYPINYIN and len(term) >= 2:
                score = self._fuzzy_match_score(text, term)
                if score >= 0.80:
                    matches.append((term, translation, score))

        # Sort by score descending, take top N
        matches.sort(key=lambda x: x[2], reverse=True)
        return [(t, tr) for t, tr, _ in matches[:max_terms]]

    def build_glossary_hint(self, text: str) -> str:
        """Build a compact glossary hint string for the LLM prompt.

        Returns empty string if no relevant terms found.
        """
        terms = self.find_relevant_terms(text)
        if not terms:
            return ""

        lines = []
        for src, tgt in terms:
            if tgt:
                lines.append(f"  {src} → {tgt}")
            else:
                lines.append(f"  {src}")

        return "Glossary (use these correct terms):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fuzzy_match_score(self, text: str, term: str) -> float:
        """Check if any substring of *text* matches *term* by pinyin."""
        if not _HAS_PYPINYIN:
            return 0.0

        term_py = self._term_pinyin.get(term, "")
        if not term_py:
            return 0.0

        # Clean text (remove punctuation)
        clean = re.sub(r"[，。,.\s!?！？；;：:、]", "", text)
        term_len = len(re.sub(r"[，。,.\s]", "", term))

        best = 0.0
        for i in range(len(clean) - term_len + 1):
            window = clean[i:i + term_len]
            window_py = "".join(lazy_pinyin(window, style=Style.NORMAL))
            if not window_py:
                continue
            # Simple ratio
            shorter, longer = (term_py, window_py) if len(term_py) <= len(window_py) else (window_py, term_py)
            if not longer:
                continue
            # Count matching chars
            match_count = sum(1 for a, b in zip(shorter, longer) if a == b)
            score = match_count / max(len(shorter), len(longer))
            if score > best:
                best = score

        return best

    @property
    def term_count(self) -> int:
        return len(self._glossary)
