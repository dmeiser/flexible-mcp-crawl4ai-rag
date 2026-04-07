from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, cast


class RerankingService:
    """Service facade for reranking workflows and helpers."""

    @staticmethod
    def rerank_results(
        *,
        settings: Any,
        endpoint_factory: Callable[[Optional[str], Optional[str]], Any],
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        logger: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank results with either OpenAI-compatible API or local cross-encoder."""
        if not settings.USE_RERANKING or not results:
            return results[:top_k]

        try:
            if settings.effective_rerank_base_url:
                return RerankingService.rerank_with_openai_compatible_api(
                    settings=settings,
                    endpoint_factory=endpoint_factory,
                    query=query,
                    results=results,
                    top_k=top_k,
                )
            return RerankingService.rerank_with_cross_encoder(
                settings=settings,
                query=query,
                results=results,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning(f"Reranking failed, returning original order: {exc}")
            return results[:top_k]

    @staticmethod
    def rerank_with_openai_compatible_api(
        *,
        settings: Any,
        endpoint_factory: Callable[[Optional[str], Optional[str]], Any],
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        endpoint = endpoint_factory(
            settings.effective_rerank_api_key,
            settings.effective_rerank_base_url,
        )
        response = endpoint.chat_completion_sync(
            request_kwargs={
                "model": settings.effective_rerank_model_name,
                "messages": cast(Any, RerankingService.rerank_messages(query, results)),
                "temperature": 0,
            },
            max_retries=settings.effective_rerank_max_retries,
            retry_delay_seconds=settings.effective_rerank_retry_delay_seconds,
            call_name="rerank LLM",
        )

        scores = RerankingService.parse_rerank_scores(response, expected_count=len(results))
        if scores is None:
            raise ValueError("Invalid rerank API response format")
        return RerankingService.apply_rerank_scores(results, scores, top_k)

    @staticmethod
    def rerank_messages(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        candidates = [
            {
                "index": idx,
                "content": str(result.get("content", ""))[:3000],
            }
            for idx, result in enumerate(results)
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You are a relevance ranker. Return strict JSON only in the format "
                    '{"scores": [float, ...]} with one score per candidate index in order. '
                    "Higher score means more relevant to the query."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"query": query, "candidates": candidates}, ensure_ascii=False),
            },
        ]

    @staticmethod
    def parse_rerank_scores(response: Any, expected_count: int) -> Optional[List[float]]:
        content = RerankingService.rerank_response_content(response)
        if content is None:
            return None
        payload = RerankingService.rerank_json_payload(content)
        scores = RerankingService.rerank_score_values(payload, expected_count)
        if scores is None:
            return None
        return [float(score) for score in scores]

    @staticmethod
    def rerank_response_content(response: Any) -> Optional[str]:
        content = response.choices[0].message.content
        return content if isinstance(content, str) else None

    @staticmethod
    def rerank_json_payload(content: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(content)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def rerank_score_values(payload: Optional[Dict[str, Any]], expected_count: int) -> Optional[List[Any]]:
        if payload is None:
            return None
        scores = payload.get("scores")
        if not isinstance(scores, list) or len(scores) != expected_count:
            return None
        return scores

    @staticmethod
    def apply_rerank_scores(results: List[Dict[str, Any]], scores: List[float], top_k: int) -> List[Dict[str, Any]]:
        for index, result in enumerate(results):
            result["rerank_score"] = float(scores[index])
        return sorted(results, key=lambda item: item.get("rerank_score", 0), reverse=True)[:top_k]

    @staticmethod
    def rerank_with_cross_encoder(
        *,
        settings: Any,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        model = CrossEncoder(settings.effective_rerank_model_name)
        pairs = [(query, result["content"]) for result in results]
        scores = model.predict(pairs)
        for index, result in enumerate(results):
            result["rerank_score"] = float(scores[index])
        return sorted(results, key=lambda item: item.get("rerank_score", 0), reverse=True)[:top_k]


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
    *,
    settings: Any | None = None,
    endpoint_factory: Callable[[Optional[str], Optional[str]], Any] | None = None,
    logger: Any | None = None,
) -> List[Dict[str, Any]]:
    """Convenience rerank facade with optional explicit dependency injection."""
    settings, endpoint_factory, logger = _resolve_rerank_dependencies(settings, endpoint_factory, logger)

    return RerankingService.rerank_results(
        settings=settings,
        endpoint_factory=endpoint_factory,
        query=query,
        results=results,
        top_k=top_k,
        logger=logger,
    )


def _resolve_rerank_dependencies(
    settings: Any | None,
    endpoint_factory: Callable[[Optional[str], Optional[str]], Any] | None,
    logger: Any | None,
) -> tuple[Any, Callable[[Optional[str], Optional[str]], Any], Any]:
    import logging as _logging  # noqa: PLC0415

    import src.utils as _utils  # noqa: PLC0415

    return (
        settings or _utils.settings,
        endpoint_factory or _utils._openai_compatible_endpoint,
        logger or _logging.getLogger(__name__),
    )


# Backward-compatible function exports
rerank_with_openai_compatible_api = RerankingService.rerank_with_openai_compatible_api
rerank_messages = RerankingService.rerank_messages
parse_rerank_scores = RerankingService.parse_rerank_scores
rerank_response_content = RerankingService.rerank_response_content
rerank_json_payload = RerankingService.rerank_json_payload
rerank_score_values = RerankingService.rerank_score_values
apply_rerank_scores = RerankingService.apply_rerank_scores
rerank_with_cross_encoder = RerankingService.rerank_with_cross_encoder
