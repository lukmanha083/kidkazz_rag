"""Extractive summarization and keyword extraction (no LLM required).

Uses TextRank (via sumy) for extractive summarization and YAKE for
unsupervised keyword extraction. Falls back to naive first-N-sentences
when dependencies are unavailable.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Optional import for sumy (TextRank)
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer

    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

# Optional import for YAKE
try:
    import yake

    YAKE_AVAILABLE = True
except ImportError:
    yake = None  # type: ignore
    YAKE_AVAILABLE = False

# Ensure NLTK punkt_tab data is available
if SUMY_AVAILABLE:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            import nltk

            nltk.download("punkt_tab", quiet=True)
        except Exception as e:
            logger.debug(f"Failed to download nltk punkt_tab: {e}")


def _fallback_summarize(text: str, sentence_count: int = 3) -> str:
    """Naive first-N-sentences fallback when sumy is unavailable.

    Args:
        text: Input text to summarize.
        sentence_count: Number of sentences to extract.

    Returns:
        First N sentences joined as a string.
    """
    if not text or not text.strip():
        return ""

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""

    selected = sentences[:sentence_count]
    return " ".join(selected)


def extractive_summarize(text: str, sentence_count: int = 3) -> str:
    """Extract key sentences using TextRank algorithm.

    Args:
        text: Input text to summarize.
        sentence_count: Number of sentences to extract.

    Returns:
        Extracted key sentences joined as a string.
    """
    if not text or not text.strip():
        return ""

    if not SUMY_AVAILABLE:
        logger.debug("sumy not available, using fallback summarization")
        return _fallback_summarize(text, sentence_count)

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)

        sentences = summarizer(parser.document, sentence_count)

        if not sentences:
            return _fallback_summarize(text, sentence_count)

        return " ".join(str(s) for s in sentences)

    except Exception as e:
        logger.debug(f"TextRank failed, using fallback: {e}")
        return _fallback_summarize(text, sentence_count)


def _normalize_keyword(kw: str) -> str:
    """Normalize a keyword to a canonical form for deduplication.

    Strips trailing 's'/'es'/'ies' to merge singular/plural variants.
    """
    k = kw.lower().strip()
    if k.endswith("ies") and len(k) > 4:
        return k[:-3] + "y"
    if k.endswith("es") and len(k) > 3:
        return k[:-2]
    if k.endswith("s") and not k.endswith("ss") and len(k) > 2:
        return k[:-1]
    return k


def _deduplicate_keywords(keywords: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Merge keywords that are singular/plural variants of each other.

    Keeps the variant with the better (lower) YAKE score.
    """
    seen: dict[str, tuple[str, float]] = {}  # normalized -> (original_name, score)

    for kw, score in keywords:
        norm = _normalize_keyword(kw)
        if norm in seen:
            # Keep the one with the better (lower) score
            if score < seen[norm][1]:
                seen[norm] = (kw, score)
        else:
            seen[norm] = (kw, score)

    return [(name, score) for name, score in seen.values()]


def extract_keywords(text: str, max_keywords: int = 5) -> list[dict]:
    """Extract keywords using YAKE unsupervised algorithm.

    Includes normalization to merge singular/plural variants
    (e.g., "cost" and "costs" become one keyword).

    Args:
        text: Input text to extract keywords from.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of dicts with 'name' and 'score' keys, sorted by relevance
        (lower YAKE score = more relevant).
    """
    if not text or not text.strip():
        return []

    if not YAKE_AVAILABLE or yake is None:
        logger.debug("yake not available, returning empty keywords")
        return []

    try:
        # Request extra keywords to have headroom after dedup
        extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # max ngram size
            dedupLim=0.9,
            dedupFunc="seqm",
            top=max_keywords * 2,
        )

        raw_keywords = extractor.extract_keywords(text)
        deduped = _deduplicate_keywords(raw_keywords)

        # Sort by score (lower = more relevant) and limit
        deduped.sort(key=lambda x: x[1])
        deduped = deduped[:max_keywords]

        return [
            {"name": kw, "score": round(score, 4)}
            for kw, score in deduped
        ]

    except Exception as e:
        logger.debug(f"YAKE extraction failed: {e}")
        return []
