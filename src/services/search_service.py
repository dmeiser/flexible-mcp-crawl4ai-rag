from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session

import src.utils as _utils
from src.models import CrawledPage
from src.services.retrieval import python_side_vector_search as _provider_python_side_vector_search
from src.services.retrieval import search_code_examples as _provider_search_code_examples
from src.services.retrieval import search_documents as _provider_search_documents


async def search_documents(
    session: Session,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    return await _provider_search_documents(
        settings=_utils.settings,
        session=session,
        create_embedding_fn=_utils.create_embedding,
        crawled_page_cls=CrawledPage,
        query=query,
        match_count=match_count,
        filter_metadata=filter_metadata,
        use_hybrid=use_hybrid,
        python_side_vector_search_fn=_python_side_vector_search,
    )


def _python_side_vector_search(
    session: Session,
    query_embedding: List[float],
    limit: int,
    filter_metadata: Optional[Dict[str, Any]],
) -> List[Any]:
    return _provider_python_side_vector_search(
        session=session,
        query_embedding=query_embedding,
        limit=limit,
        filter_metadata=filter_metadata,
        crawled_page_cls=CrawledPage,
    )


async def search_code_examples(
    session: Session,
    query: str,
    match_count: int = 5,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return await _provider_search_code_examples(
        session=session,
        query=query,
        create_embedding_fn=_utils.create_embedding,
        match_count=match_count,
        language=language,
    )
