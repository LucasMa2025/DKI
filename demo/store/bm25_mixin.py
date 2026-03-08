"""
BM25 Search Mixin - Shared BM25 tokenization and scoring

Extracted from SQLiteChatStore / PostgresChatStore to eliminate code duplication.
Aligned with ConfigDrivenAdapter._bm25_score / _tokenize.

Author: AGI Demo Project
Version: 3.0.0
"""

import math
import re
from typing import List, Tuple

from loguru import logger

from demo.store.models import DemoMessage


class BM25Mixin:
    """
    BM25 search mixin.

    Provides:
    - _tokenize(): Chinese/English mixed tokenization (jieba + fallback)
    - _bm25_score(): BM25 scoring aligned with ConfigDrivenAdapter
    - _extract_keywords(): keyword extraction for DB pre-filtering
    """

    # Chinese stopwords (high-frequency, low-information words)
    _CN_STOPWORDS = frozenset({
        '\u7684', '\u4e86', '\u5728', '\u662f', '\u6211', '\u6709', '\u548c',
        '\u5c31', '\u4e0d', '\u4eba', '\u90fd', '\u4e00',
        '\u4e00\u4e2a', '\u4e0a', '\u4e5f', '\u5f88', '\u5230', '\u8bf4',
        '\u8981', '\u53bb', '\u4f60', '\u4f1a', '\u7740',
        '\u6ca1\u6709', '\u770b', '\u597d', '\u81ea\u5df1', '\u8fd9',
        '\u4ed6', '\u5979', '\u5b83', '\u4eec', '\u90a3', '\u4e9b',
        '\u4ec0\u4e48', '\u5417', '\u5462', '\u5427', '\u554a', '\u54e6',
        '\u55ef', '\u5440', '\u54c8', '\u54ea', '\u561b',
        '\u53ef\u4ee5', '\u6ca1', '\u8fd8', '\u5bf9', '\u628a', '\u8ba9',
        '\u88ab', '\u4ece', '\u7ed9', '\u7528', '\u4f46',
        '\u800c', '\u53c8', '\u6240\u4ee5', '\u56e0\u4e3a', '\u5982\u679c',
        '\u8fd9\u4e2a', '\u90a3\u4e2a', '\u600e\u4e48', '\u4e3a\u4ec0\u4e48',
        '\u54ea\u4e2a', '\u591a\u5c11', '\u51e0', '\u8c01',
        '\u600e\u6837', '\u8fd9\u6837', '\u90a3\u6837',
    })

    def __init_bm25__(self):
        """Initialize BM25 capabilities. Call in subclass __init__."""
        self._jieba_available = False
        try:
            import jieba  # noqa: F401
            self._jieba_available = True
        except ImportError:
            logger.info("jieba not available, BM25 will use char+bigram tokenizer")

    def _tokenize(self, text_str: str) -> List[str]:
        """
        Chinese/English mixed tokenization.

        Strategy:
        - With jieba: jieba segmentation + English words + stopword filter
        - Without jieba: char + bigram + English words + stopword filter
        """
        tokens = []
        text_lower = text_str.lower()

        # English words
        en_tokens = re.findall(r'[a-zA-Z0-9]+', text_lower)
        tokens.extend(en_tokens)

        if self._jieba_available:
            import jieba
            cn_text = re.sub(r'[a-zA-Z0-9]+', ' ', text_lower)
            words = jieba.lcut(cn_text)
            for w in words:
                w = w.strip()
                if len(w) >= 1 and any('\u4e00' <= c <= '\u9fff' for c in w):
                    if w not in self._CN_STOPWORDS:
                        tokens.append(w)
        else:
            # Fallback: char + bigram
            cn_chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
            for i in range(len(cn_chars)):
                if cn_chars[i] not in self._CN_STOPWORDS:
                    tokens.append(cn_chars[i])
                if i + 1 < len(cn_chars):
                    bigram = cn_chars[i] + cn_chars[i + 1]
                    if bigram not in self._CN_STOPWORDS:
                        tokens.append(bigram)

        return tokens

    def _bm25_score(
        self, query: str, messages: List[DemoMessage],
        k1: float = 1.5, b: float = 0.75,
    ) -> List[Tuple[DemoMessage, float]]:
        """
        BM25 scoring (aligned with ConfigDrivenAdapter._bm25_score).

        Args:
            query: search query
            messages: candidate messages
            k1: term frequency saturation parameter
            b: document length normalization parameter

        Returns:
            List of (message, score) tuples
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(msg, 0.0) for msg in messages]

        doc_tokens_list = [self._tokenize(msg.content) for msg in messages]
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)

        N = len(messages)
        idf = {}
        for qt in set(query_tokens):
            df = sum(1 for dt in doc_tokens_list if qt in dt)
            idf[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        results = []
        for msg, doc_tokens in zip(messages, doc_tokens_list):
            score = 0.0
            dl = len(doc_tokens)
            tf_map: dict = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                score += (
                    idf.get(qt, 0)
                    * (tf * (k1 + 1))
                    / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                )

            results.append((msg, score))

        return results

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords for DB ILIKE pre-filtering (PostgreSQL)."""
        tokens = self._tokenize(query)
        # Filter short tokens (single Chinese chars may be too broad)
        keywords = [t for t in tokens if len(t) >= 2]
        if not keywords:
            keywords = tokens[:5]
        return keywords[:10]
