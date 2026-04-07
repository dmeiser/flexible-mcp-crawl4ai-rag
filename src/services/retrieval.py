from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import text
from sqlmodel import Session, select


async def search_documents(
    *,
    settings: Any,
    session: Session,
    create_embedding_fn: Any,
    crawled_page_cls: Any,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: Optional[bool] = None,
    python_side_vector_search_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    hybrid = _resolve_hybrid_mode(settings, use_hybrid)
    query_embedding = await _safe_query_embedding(create_embedding_fn, query)
    if query_embedding is None:
        return []

    filter_json = json.dumps(filter_metadata or {})
    return _search_documents_with_embedding(
        session=session,
        query=query,
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        filter_json=filter_json,
        hybrid=hybrid,
        crawled_page_cls=crawled_page_cls,
        python_side_vector_search_fn=python_side_vector_search_fn,
    )


def _search_documents_with_embedding(
    *,
    session: Session,
    query: str,
    query_embedding: List[float],
    match_count: int,
    filter_metadata: Optional[Dict[str, Any]],
    filter_json: str,
    hybrid: bool,
    crawled_page_cls: Any,
    python_side_vector_search_fn: Optional[Any],
) -> List[Dict[str, Any]]:
    vector_results = _vector_search_rows_to_results(
        _run_vector_search(
            session=session,
            query_embedding=query_embedding,
            filter_json=filter_json,
            match_count=match_count,
            hybrid=hybrid,
            filter_metadata=filter_metadata,
            crawled_page_cls=crawled_page_cls,
            python_side_vector_search_fn=python_side_vector_search_fn,
        )
    )

    if not hybrid:
        return vector_results[:match_count]

    fts_raw = _run_fts_search(
        session=session,
        query=query,
        filter_json=filter_json,
        match_count=match_count,
    )
    return _merge_hybrid_results(vector_results=vector_results, fts_raw=fts_raw, match_count=match_count)


def search_documents_with_embedding(
    *,
    session: Session,
    query: str,
    query_embedding: List[float],
    match_count: int,
    filter_metadata: Optional[Dict[str, Any]],
    filter_json: str,
    hybrid: bool,
    crawled_page_cls: Any,
    python_side_vector_search_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    return _search_documents_with_embedding(
        session=session,
        query=query,
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        filter_json=filter_json,
        hybrid=hybrid,
        crawled_page_cls=crawled_page_cls,
        python_side_vector_search_fn=python_side_vector_search_fn,
    )


def _resolve_hybrid_mode(settings: Any, use_hybrid: Optional[bool]) -> bool:
    if use_hybrid is None:
        return bool(settings.USE_HYBRID_SEARCH)
    return use_hybrid


async def _safe_query_embedding(create_embedding_fn: Any, query: str) -> Optional[List[float]]:
    try:
        return await create_embedding_fn(query)
    except Exception:
        return None


def _run_vector_search(
    *,
    session: Session,
    query_embedding: List[float],
    filter_json: str,
    match_count: int,
    hybrid: bool,
    filter_metadata: Optional[Dict[str, Any]],
    crawled_page_cls: Any,
    python_side_vector_search_fn: Optional[Any],
) -> Sequence[Any]:
    try:
        return session.exec(  # type: ignore[call-overload]
            text(
                "SELECT id, url, chunk_number, content, metadata, similarity "
                "FROM match_crawled_pages(CAST(:emb AS vector), :cnt, CAST(:filt AS jsonb))"
            ).bindparams(
                emb=str(query_embedding),
                cnt=match_count * 2 if hybrid else match_count,
                filt=filter_json,
            )
        ).all()
    except Exception:
        fallback_fn = python_side_vector_search_fn or python_side_vector_search
        return fallback_fn(
            session=session,
            query_embedding=query_embedding,
            limit=match_count * 2,
            filter_metadata=filter_metadata,
            crawled_page_cls=crawled_page_cls,
        )


def _vector_search_rows_to_results(raw_rows: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": row[0],
            "url": row[1],
            "chunk_number": row[2],
            "content": row[3],
            "page_metadata": row[4] if isinstance(row[4], dict) else {},
            "similarity_score": float(row[5]),
            "fts_score": 0.0,
        }
        for row in raw_rows
    ]


def _run_fts_search(
    *,
    session: Session,
    query: str,
    filter_json: str,
    match_count: int,
) -> Sequence[Any]:
    try:
        return session.exec(  # type: ignore[call-overload]
            text(
                "SELECT id, url, chunk_number, content, metadata, "
                "ts_rank(fts, plainto_tsquery('english', :q)) AS rank "
                "FROM crawled_pages "
                "WHERE fts @@ plainto_tsquery('english', :q) "
                "AND is_active = TRUE "
                "AND tombstoned_at IS NULL "
                "AND CAST(:filt AS jsonb) <@ metadata "
                "ORDER BY rank DESC LIMIT :cnt"
            ).bindparams(q=query, filt=filter_json, cnt=match_count * 2)
        ).all()
    except Exception:
        return []


def _merge_hybrid_results(
    *,
    vector_results: List[Dict[str, Any]],
    fts_raw: Sequence[Any],
    match_count: int,
) -> List[Dict[str, Any]]:
    fts_map = _fts_score_map(fts_raw)
    vector_rank = _rank_map_from_vector_results(vector_results)
    fts_rank = _rank_map_from_fts_rows(fts_raw)
    rrf_scores = _rrf_scores(vector_rank, fts_rank)
    id_to_result = _merge_vector_and_fts_rows(vector_results, fts_raw, fts_map)
    merged = sorted(id_to_result.values(), key=lambda result: rrf_scores.get(result["id"], 0), reverse=True)
    return merged[:match_count]


def _fts_score_map(fts_raw: Sequence[Any]) -> Dict[int, float]:
    return {int(row[0]): float(row[5]) for row in fts_raw}


def _rank_map_from_vector_results(vector_results: List[Dict[str, Any]]) -> Dict[int, int]:
    return {int(result["id"]): index + 1 for index, result in enumerate(vector_results)}


def _rank_map_from_fts_rows(fts_raw: Sequence[Any]) -> Dict[int, int]:
    return {int(row[0]): index + 1 for index, row in enumerate(fts_raw)}


def _rrf_scores(vector_rank: Dict[int, int], fts_rank: Dict[int, int]) -> Dict[int, float]:
    k = 60
    all_ids = set(vector_rank) | set(fts_rank)
    return {
        record_id: 1 / (k + vector_rank.get(record_id, len(all_ids) + k))
        + 1 / (k + fts_rank.get(record_id, len(all_ids) + k))
        for record_id in all_ids
    }


def _merge_vector_and_fts_rows(
    vector_results: List[Dict[str, Any]],
    fts_raw: Sequence[Any],
    fts_map: Dict[int, float],
) -> Dict[int, Dict[str, Any]]:
    id_to_result: Dict[int, Dict[str, Any]] = {int(result["id"]): result for result in vector_results}
    for row in fts_raw:
        row_id = int(row[0])
        if row_id in id_to_result:
            continue
        id_to_result[row_id] = {
            "id": row_id,
            "url": row[1],
            "chunk_number": row[2],
            "content": row[3],
            "page_metadata": row[4] if isinstance(row[4], dict) else {},
            "similarity_score": 0.0,
            "fts_score": fts_map.get(row_id, 0.0),
        }
    return id_to_result


def merge_vector_and_fts_rows(
    vector_results: List[Dict[str, Any]],
    fts_raw: Sequence[Any],
    fts_map: Dict[int, float],
) -> Dict[int, Dict[str, Any]]:
    return _merge_vector_and_fts_rows(vector_results, fts_raw, fts_map)


def _python_side_vector_search(
    *,
    session: Session,
    query_embedding: List[float],
    limit: int,
    filter_metadata: Optional[Dict[str, Any]],
    crawled_page_cls: Any,
) -> List[Any]:
    q = np.array(query_embedding, dtype=np.float32)
    pages = _active_pages(session.exec(select(crawled_page_cls)).all())
    pages = _metadata_filtered_pages(pages, filter_metadata)

    scored: List[Any] = []
    for p in pages:
        scored_row = _page_similarity_row(p, q)
        if scored_row is not None:
            scored.append(scored_row)

    scored.sort(key=lambda x: x[5], reverse=True)
    return scored[:limit]


def python_side_vector_search(
    *,
    session: Session,
    query_embedding: List[float],
    limit: int,
    filter_metadata: Optional[Dict[str, Any]],
    crawled_page_cls: Any,
) -> List[Any]:
    return _python_side_vector_search(
        session=session,
        query_embedding=query_embedding,
        limit=limit,
        filter_metadata=filter_metadata,
        crawled_page_cls=crawled_page_cls,
    )


def _active_pages(pages: Sequence[Any]) -> List[Any]:
    return [page for page in pages if page.is_active and page.tombstoned_at is None]


def _metadata_filtered_pages(
    pages: List[Any],
    filter_metadata: Optional[Dict[str, Any]],
) -> List[Any]:
    if not filter_metadata:
        return pages
    return [page for page in pages if _page_matches_metadata(page, filter_metadata)]


def _page_matches_metadata(page: Any, filter_metadata: Dict[str, Any]) -> bool:
    metadata = page.page_metadata if isinstance(page.page_metadata, dict) else {}
    return all(metadata.get(key) == value for key, value in filter_metadata.items())


def _page_similarity_row(page: Any, query_vec: np.ndarray[Any, Any]) -> Optional[tuple[Any, ...]]:
    if not page.embedding:
        return None
    similarity = _cosine_similarity(query_vec, np.array(page.embedding, dtype=np.float32))
    return (page.id, page.url, page.chunk_number, page.content, page.page_metadata, similarity)


def _cosine_similarity(vec_a: np.ndarray[Any, Any], vec_b: np.ndarray[Any, Any]) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


async def search_code_examples(
    *,
    session: Session,
    query: str,
    create_embedding_fn: Any,
    match_count: int = 5,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    try:
        query_embedding = await create_embedding_fn(query)
    except Exception:
        return []

    filter_json = _code_examples_filter_json(language)
    raw = _query_code_example_rows(session, query_embedding, match_count, filter_json)
    return [_code_example_row_to_result(row) for row in raw]


def _code_examples_filter_json(language: Optional[str]) -> str:
    if not language:
        return json.dumps({})
    return json.dumps({"language": language})


def _query_code_example_rows(
    session: Session,
    query_embedding: List[float],
    match_count: int,
    filter_json: str,
) -> Sequence[Any]:
    try:
        return session.execute(
            text(
                "SELECT id, url, chunk_number, language, content, summary, metadata, similarity "
                "FROM match_code_examples(CAST(:emb AS vector), :cnt, CAST(:filt AS jsonb))"
            ).bindparams(emb=str(query_embedding), cnt=match_count, filt=filter_json)
        ).all()
    except Exception:
        return []


def query_code_example_rows(
    session: Session,
    query_embedding: List[float],
    match_count: int,
    filter_json: str,
) -> Sequence[Any]:
    return _query_code_example_rows(session, query_embedding, match_count, filter_json)


def _code_example_row_to_result(row: Any) -> Dict[str, Any]:
    return {
        "id": row[0],
        "url": row[1],
        "chunk_number": row[2],
        "language": row[3],
        "content": row[4],
        "summary": row[5],
        "metadata": row[6] if isinstance(row[6], dict) else {},
        "similarity_score": float(row[7]),
    }


def code_example_row_to_result(row: Any) -> Dict[str, Any]:
    return _code_example_row_to_result(row)
