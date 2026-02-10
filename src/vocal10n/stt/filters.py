"""Hallucination / repetition / duplicate filters for STT output.

Ported from the prebuild's ``TranscriptManager`` helper methods
(``_is_repetitive``, ``_contains_filtered_phrases``, ``_is_duplicate``,
phonetic correction, Traditional→Simplified conversion, punctuation
normalisation).

The filter-file format (``filters.txt``) is:

    # comment
    PHRASE:literal text to block
    REGEX:^pattern$
    plain line treated as PHRASE
"""

import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Optional Chinese helpers ──────────────────────────────────────────
try:
    from opencc import OpenCC
    _cc = OpenCC("t2s")
    _HAS_OPENCC = True
except ImportError:
    _cc = None
    _HAS_OPENCC = False

try:
    from pypinyin import Style, lazy_pinyin
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False


# ── Acoustic confusion table (Whisper common errors) ──────────────────
_ACOUSTIC_CONFUSIONS: dict[str, list[str]] = {
    "an": ["ang", "en"], "ang": ["an", "eng"],
    "en": ["eng", "an"], "eng": ["en", "ang"],
    "in": ["ing"], "ing": ["in"],
    "un": ["ong"], "ong": ["un"],
    "za": ["zha", "ja"], "zha": ["za", "ja"],
    "ze": ["zhe", "je"], "zhe": ["ze", "je"],
    "zi": ["zhi", "ji"], "zhi": ["zi", "ji"],
    "zu": ["zhu", "ju"], "zhu": ["zu", "ju"],
    "ca": ["cha"], "cha": ["ca"],
    "ci": ["chi", "qi"], "chi": ["ci", "qi"],
    "cu": ["chu"], "chu": ["cu"],
    "sa": ["sha"], "sha": ["sa"],
    "si": ["shi"], "shi": ["si"],
    "su": ["shu"], "shu": ["su"],
    "la": ["na", "ra"], "na": ["la", "ra"],
    "li": ["ni", "ri"], "ni": ["li", "ri"],
    "ji": ["zhi", "qi", "xi"], "qi": ["ji", "chi", "xi"],
    "xi": ["shi", "qi", "ji"],
    "jian": ["qian", "xian"], "qian": ["jian", "xian"],
    "xian": ["jian", "qian"],
    "lei": ["lie", "le"], "lie": ["lei", "le"],
    "mei": ["mi", "mu", "mo"], "bei": ["bie", "bai"],
}


class STTFilters:
    """Filters applied to STT segments before they are accepted.

    Includes:
    - Repetition detection
    - Phrase / regex hallucination filters (loaded from file)
    - Duplicate detection (timestamp overlap)
    - Traditional → Simplified Chinese conversion
    - Phonetic correction (pinyin similarity against a term list)
    - Punctuation normalisation
    """

    def __init__(self, max_repeated_chars: int = 10) -> None:
        self._max_repeated = max_repeated_chars

        # Filter lists (loaded from file)
        self._phrase_filters: list[str] = []
        self._regex_filters: list[re.Pattern] = []

        # Phonetic index
        self._gaming_terms: list[str] = []
        self._term_pinyin: dict[str, str] = {}
        self._term_syllables: dict[str, list[str]] = {}
        self._term_clean: dict[str, str] = {}
        self._pinyin_cache: dict[str, str] = {}
        self._syllable_cache: dict[str, list[str]] = {}

        # Phonetic thresholds (overridden by caller or config)
        self.similarity_threshold: float = 0.85
        self.confidence_threshold: float = 0.70
        self.override_threshold: float = 0.95

        # Common words that must never be corrected to gaming terms
        self._protected: set[str] = {
            "还在", "还在啊", "还在吗", "还在呢",
            "简单", "不简单", "希望", "失望",
            "实际", "星期", "心里", "开心", "可以", "事情", "时间",
        }

    # ══════════════════════════════════════════════════════════════════
    #  Setup / Loading
    # ══════════════════════════════════════════════════════════════════

    def load_filter_file(self, path: str | Path) -> None:
        """Load ``PHRASE:`` / ``REGEX:`` filter patterns from *path*."""
        p = Path(path)
        if not p.exists():
            logger.warning("Filter file not found: %s", p)
            return
        self._phrase_filters.clear()
        self._regex_filters.clear()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("REGEX:"):
                try:
                    self._regex_filters.append(re.compile(line[6:].strip()))
                except re.error as e:
                    logger.warning("Bad regex filter %r: %s", line, e)
            elif line.startswith("PHRASE:"):
                self._phrase_filters.append(line[7:].strip())
            else:
                self._phrase_filters.append(line)
        logger.info(
            "Loaded %d phrase + %d regex filters",
            len(self._phrase_filters),
            len(self._regex_filters),
        )

    def build_phonetic_index(self, context_path: str | Path) -> None:
        """Build the pinyin index from a context/gaming-terms file."""
        if not _HAS_PYPINYIN:
            logger.warning("pypinyin not installed — phonetic correction disabled")
            return
        p = Path(context_path)
        if not p.exists():
            logger.warning("Context file not found: %s", p)
            return

        self._gaming_terms.clear()
        self._term_pinyin.clear()
        self._term_syllables.clear()
        self._term_clean.clear()

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self._gaming_terms.append(line)
            clean = line.replace(" ", "")
            py = "".join(lazy_pinyin(line, style=Style.NORMAL))
            sy = lazy_pinyin(line, style=Style.NORMAL)
            self._term_pinyin[line] = py
            self._term_pinyin[clean] = py
            self._term_syllables[line] = sy
            self._term_syllables[clean] = sy
            self._term_clean[line] = clean
        logger.info("Phonetic index: %d terms", len(self._gaming_terms))

    # ══════════════════════════════════════════════════════════════════
    #  Public filter API
    # ══════════════════════════════════════════════════════════════════

    def is_repetitive(self, text: str) -> bool:
        """``True`` if *text* looks like a Whisper hallucination loop."""
        if not text:
            return True

        # Consecutive repeated characters
        count = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                count += 1
                if count > self._max_repeated:
                    return True
            else:
                count = 1

        words = text.split()
        if len(words) <= 1:
            return False
        if len(set(words)) == 1 and len(words) > 3:
            return True
        mc = max(Counter(words).values())
        if mc > 3:
            return True
        if len(words) >= 5 and any(c / len(words) > 0.3 for c in Counter(words).values()):
            return True
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i + 1] == words[i + 2]:
                    return True
            for i in range(len(words) - 3):
                if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
                    return True
        if len(words) >= 6:
            mid = len(words) // 2
            if words[:mid] == words[mid : 2 * mid]:
                return True
        return False

    def contains_filtered(
        self, text: str, confidence: float = 0.0, no_speech_prob: float = 0.0
    ) -> bool:
        """``True`` if *text* matches any loaded phrase/regex filter or a
        known context-aware hallucination pattern."""
        if not text:
            return False

        stripped = text.strip().rstrip("。，,. ")
        hallucination_chars = "好对吧啊嗯"
        if len(stripped) == 1 and stripped in hallucination_chars:
            if confidence < -1.0 or no_speech_prob > 0.5:
                return True

        context_filters: dict[str, bool] = {
            "谢谢": len(text.strip()) <= 3 or confidence < -1.5 or no_speech_prob > 0.7,
            "以下是普通话的转录": True,
            "字幕由": True,
        }
        for phrase, cond in context_filters.items():
            if phrase in text and cond:
                return True

        for pf in self._phrase_filters:
            if pf in text:
                return True
        for rx in self._regex_filters:
            if rx.search(text):
                return True
        return False

    @staticmethod
    def is_duplicate(
        text: str,
        start: float,
        end: float,
        history: list[dict],
        look_back: int = 3,
    ) -> bool:
        """``True`` if *text* is a near-duplicate of a recent segment
        (by timestamp overlap + text similarity)."""
        if not history:
            return False
        clean = _strip_punct(text).lower().strip()
        if not clean:
            return True
        for seg in history[-look_back:]:
            seg_clean = _strip_punct(seg["text"]).lower().strip()
            o_start = max(start, seg["start"])
            o_end = min(end, seg["end"])
            if o_end <= o_start:
                continue
            overlap = o_end - o_start
            dur = end - start
            if dur > 0 and overlap / dur > 0.7:
                if clean == seg_clean:
                    return True
                if clean in seg_clean or seg_clean in clean:
                    shorter = min(len(clean), len(seg_clean))
                    longer = max(len(clean), len(seg_clean))
                    if shorter / longer > 0.7:
                        return True
        return False

    # ══════════════════════════════════════════════════════════════════
    #  Chinese text helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def to_simplified(text: str) -> str:
        if _HAS_OPENCC and _cc:
            return _cc.convert(text)
        return text

    @staticmethod
    def is_chinese(text: str) -> bool:
        if not text:
            return False
        cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return cn > len(text.replace(" ", "")) * 0.3

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalise punctuation to match the dominant language of *text*."""
        if not text:
            return text
        text = re.sub(r"(\d)。(\d)", r"\1.\2", text)
        if STTFilters.is_chinese(text):
            for e, c in [(",", "，"), (".", "。"), ("!", "！"), ("?", "？"), (":", "："), (";", "；")]:
                text = re.sub(r"(\d)\." , r"\1<<<D>>>", text)
                text = text.replace(e, c)
                text = text.replace("<<<D>>>", ".")
            return text
        else:
            for c, e in [("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("：", ":"), ("；", ";")]:
                text = text.replace(c, e)
            return text

    @staticmethod
    def ensure_punctuation(text: str, is_final: bool = False, long_pause: bool = False) -> str:
        """Make sure *text* ends with punctuation."""
        text = text.strip()
        if not text:
            return text
        text = STTFilters.normalize_punctuation(text)
        endings = "。！？.!?"
        clauses = "，；：、,;:"
        last = text[-1]
        if last in endings or last in clauses:
            if (is_final or long_pause) and last in clauses:
                return text[:-1] + ("。" if STTFilters.is_chinese(text) else ".")
            return text
        if is_final or long_pause:
            return text + ("。" if STTFilters.is_chinese(text) else ".")
        return text + ("，" if STTFilters.is_chinese(text) else ",")

    @staticmethod
    def strip_punctuation(text: str) -> str:
        """Remove punctuation, preserving decimal points."""
        text = re.sub(r"(\d)\.(\d)", r"\1__D__\2", text)
        for p in ".,!?;:。，！？；：、…—\"\"''「」『』【】（）《》〈〉":
            text = text.replace(p, " ")
        text = text.replace("__D__", ".")
        return re.sub(r" {2,}", " ", text).strip()

    # ══════════════════════════════════════════════════════════════════
    #  Phonetic correction
    # ══════════════════════════════════════════════════════════════════

    def phonetic_correct(self, text: str, word_confidences: dict[str, float] | None = None) -> str:
        """Apply phonetic correction to *text* using the loaded term index.

        Three passes (same as prebuild):
        1. Confidence-based — uncertain words only.
        2. Override — very strong match even if Whisper is confident.
        3. Phrase-match — sliding window across cleaned text.
        """
        if not _HAS_PYPINYIN or not self._gaming_terms:
            return text

        wc = word_confidences or {}
        original = text

        # ── Pass 3 (phrase-match, applied first to whole string) ──────
        text = self._phrase_match_pass(text)

        # ── Pass 1 + 2 (per-word) ────────────────────────────────────
        parts = re.split(r"([ ，。,.\n]+)", text)
        corrected: list[str] = []
        for part in parts:
            if not part or part.strip() in " ，。,.\n":
                corrected.append(part)
                continue
            ps = part.strip()
            if len(ps) < 2 or ps in self._protected:
                corrected.append(part)
                continue
            conf = wc.get(ps, 0.5)
            best_term, best_sim = self._best_match(ps)
            should = False
            if conf < self.confidence_threshold and best_sim >= self.similarity_threshold:
                should = True
            if best_sim >= self.override_threshold:
                should = True
            if should and best_term and best_term != ps:
                logger.debug("Phonetic: '%s' → '%s' (sim=%.2f)", ps, best_term, best_sim)
                corrected.append(best_term)
            else:
                corrected.append(part)
        result = "".join(corrected)

        if result != original:
            logger.info("Phonetic corrected: '%s' → '%s'", original, result)
        return result

    # internal helpers -------------------------------------------------

    def _phrase_match_pass(self, text: str) -> str:
        """Sliding-window phrase match across the entire text."""
        clean = re.sub(r"[ ,，。.]", "", text)
        best: dict[tuple[int, int], tuple[str, float]] = {}
        for term in self._gaming_terms:
            if len(term) < 2:
                continue
            tc = self._term_clean.get(term, term.replace(" ", ""))
            tl = len(tc)
            min_sim = max(self.similarity_threshold, 0.90) if tl == 2 else self.similarity_threshold
            for i in range(len(clean) - tl + 1):
                window = clean[i : i + tl]
                sim = self._pinyin_similarity(window, tc)
                if sim >= min_sim:
                    key = (i, tl)
                    if key not in best or sim > best[key][1]:
                        best[key] = (term, sim)
        if not best:
            return text
        # Pick single best
        (i, tl), (term, sim) = max(best.items(), key=lambda x: (x[1][1], len(x[1][0])))
        # Map cleaned index → original index
        c2o = [j for j, ch in enumerate(text) if ch not in " ,，。."]
        if i < len(c2o) and i + tl - 1 < len(c2o):
            s = c2o[i]
            e = c2o[i + tl - 1] + 1
            orig = text[s:e]
            if orig in self._protected or orig == term:
                return text
            logger.debug("Phonetic [phrase]: '%s' → '%s' (sim=%.2f)", orig, term, sim)
            return text[:s] + term + text[e:]
        return text

    def _best_match(self, word: str) -> tuple[str | None, float]:
        best_t: str | None = None
        best_s = 0.0
        for term in self._gaming_terms:
            if len(term) < 2:
                continue
            lr = len(word) / len(term) if len(term) else 0
            if lr < 0.5 or lr > 2.0:
                continue
            s = self._pinyin_similarity(word, term)
            if s > best_s:
                best_s = s
                best_t = term
        return best_t, best_s

    def _pinyin_similarity(self, a: str, b: str) -> float:
        """Levenshtein-on-pinyin with acoustic-confusion boost."""
        if not _HAS_PYPINYIN:
            return 0.0
        pa = self._get_pinyin(a)
        pb = self._get_pinyin(b)
        if not pa and not pb:
            return 1.0
        if not pa or not pb:
            return 0.0

        # Acoustic boost
        sa = self._get_syllables(a)
        sb = self._get_syllables(b)
        boost = 0.0
        if len(sa) == len(sb) and sa:
            sim_cnt = 0.0
            for s1, s2 in zip(sa, sb):
                if s1 == s2:
                    sim_cnt += 1
                elif s2 in _ACOUSTIC_CONFUSIONS.get(s1, []) or s1 in _ACOUSTIC_CONFUSIONS.get(s2, []):
                    sim_cnt += 0.8
            if sim_cnt / len(sa) > 0.7:
                boost = 0.30

        # Levenshtein
        p1, p2 = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
        dists: list[int] = list(range(len(p1) + 1))
        for c2 in p2:
            nd = [dists[0] + 1]
            for i1, c1 in enumerate(p1):
                nd.append(dists[i1] if c1 == c2 else 1 + min(dists[i1], dists[i1 + 1], nd[-1]))
            dists = nd
        mx = max(len(pa), len(pb))
        return min(1.0, 1.0 - dists[-1] / mx + boost)

    def _get_pinyin(self, text: str) -> str:
        if text in self._term_pinyin:
            return self._term_pinyin[text]
        if text in self._pinyin_cache:
            return self._pinyin_cache[text]
        r = "".join(lazy_pinyin(text, style=Style.NORMAL))
        self._pinyin_cache[text] = r
        if len(self._pinyin_cache) > 5000:
            for k in list(self._pinyin_cache)[:2500]:
                del self._pinyin_cache[k]
        return r

    def _get_syllables(self, text: str) -> list[str]:
        if text in self._term_syllables:
            return self._term_syllables[text]
        if text in self._syllable_cache:
            return self._syllable_cache[text]
        r = lazy_pinyin(text, style=Style.NORMAL)
        self._syllable_cache[text] = r
        if len(self._syllable_cache) > 5000:
            for k in list(self._syllable_cache)[:2500]:
                del self._syllable_cache[k]
        return r


# ── Module-level helpers ──────────────────────────────────────────────

def _strip_punct(text: str) -> str:
    return STTFilters.strip_punctuation(text)
