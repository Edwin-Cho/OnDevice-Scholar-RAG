from __future__ import annotations

import re
from typing import List, Tuple

from app.config import settings
from app.pipeline.embedder import Embedder
from app.pipeline.store import VectorStore


_FALLBACK_SCORE_THRESHOLD = 0.20  # 2nd-pass threshold when primary (0.30) yields nothing


def retrieve(query: str, top_k: int | None = None) -> List[Tuple[dict, float]]:
    """
    쿼리를 임베딩 후 FAISS Top-K 검색 + 임계값 필터링.

    1st pass: retrieval_score_threshold (0.30)
    2nd pass: _FALLBACK_SCORE_THRESHOLD (0.20) — vocabulary mismatch 쿼리 구제
    최종 판단(topic 관련성)은 생성기 Rule 3에 위임.

    Returns:
        List of (metadata_dict, cosine_score) sorted by score desc.
        관련 청크가 없으면 빈 리스트 반환.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k

    embedder = Embedder.get()
    query_vector = embedder.embed_one(query)

    store = VectorStore.get()
    results = store.search(query_vector, top_k=k)

    if not results:
        results = store.search(query_vector, top_k=k, score_threshold=_FALLBACK_SCORE_THRESHOLD)

    if results:
        top_score = results[0][1]
        cutoff = top_score - settings.retrieval_score_gap
        results = [(meta, score) for meta, score in results if score >= cutoff]

    return results


# ── P15: Comparison Query Sub-Retrieval ───────────────────────────────────────

_COMPARISON_KW_RE = re.compile(
    r'\b(?:vs\.?|versus|compared?\s+(?:to|with)|difference\s+between'
    r'|how\s+does\s+\w+\s+(?:differ|compare))\b',
    re.IGNORECASE,
)

_COMPARISON_SPLIT_RE = re.compile(
    r'\s+(?:vs\.?|versus|compared?\s+(?:to|with))\s+',
    re.IGNORECASE,
)

_BETWEEN_AND_RE = re.compile(
    r'\bbetween\s+(.+?)\s+and\s+(.+)',
    re.IGNORECASE,
)


def is_comparison_query(query: str) -> bool:
    """Return True if the query compares two entities."""
    return bool(_COMPARISON_KW_RE.search(query))


def _extract_comparison_sides(query: str) -> tuple[str, str] | None:
    """
    비교 쿼리에서 두 피비교 대상 추출.
    Returns (side_a, side_b) or None if extraction fails.
    """
    m = _BETWEEN_AND_RE.search(query)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip('?.')

    parts = _COMPARISON_SPLIT_RE.split(query, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip().rstrip('?.')

    return None


def retrieve_comparison(query: str, top_k: int | None = None) -> List[Tuple[dict, float]]:
    """
    P15: Comparison query sub-retrieval.

    'A vs B' 또는 'difference between A and B' 형태의 쿼리에서
    양측을 분리하여 각각 검색한 뒤 결과를 병합.
    양측 문서가 모두 컨텍스트에 포함되도록 보장.

    비교 분해에 실패하면 기본 retrieve()로 fallback.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    sides = _extract_comparison_sides(query)

    if not sides:
        return retrieve(query, top_k=k)

    side_a, side_b = sides
    k_each = max(2, k // 2)

    results_main = retrieve(query, top_k=k_each)
    results_a = retrieve(side_a, top_k=k_each)
    results_b = retrieve(side_b, top_k=k_each)

    seen: dict[tuple, Tuple[dict, float]] = {}
    for meta, score in results_main + results_a + results_b:
        key = (meta.get("source_filename", ""), meta.get("chunk_index", 0))
        if key not in seen or score > seen[key][1]:
            seen[key] = (meta, score)

    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return merged[:k]
