"""
OpenRouter web search provider built on the OpenAI-compatible base.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from src.providers.openai_stack import OpenAICompatibleEndpoint, OpenAIConfiguration

logger = logging.getLogger(__name__)


class WebSearchModel(ABC):
    @abstractmethod
    async def search(
        self,
        query: str,
        engine: str,
        max_results: int,
        allowed_domains: Optional[List[str]],
        excluded_domains: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Execute live web search and return normalized response payload."""


class OpenRouterWebSearchAdapter(WebSearchModel):
    """Web search adapter that extends the OpenAI configuration base.

    Wraps OpenRouter's web-search-augmented chat endpoint using the same
    ``OpenAIConfiguration`` / ``OpenAICompatibleEndpoint`` primitives used by
    all other LLM and embedding providers.
    """

    def __init__(
        self,
        *,
        configuration: OpenAIConfiguration,
        model_name: str,
        endpoint_factory: Callable[[Optional[str], Optional[str]], OpenAICompatibleEndpoint],
        default_engine: str = "auto",
        default_max_results: int = 5,
    ):
        self._configuration = configuration
        self._model_name = model_name
        self._endpoint_factory = endpoint_factory
        self._default_engine = default_engine
        self._default_max_results = default_max_results

    async def search(
        self,
        query: str,
        engine: str,
        max_results: int,
        allowed_domains: Optional[List[str]],
        excluded_domains: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not self._configuration.resolved_api_key:
            raise ValueError("WEB_SEARCH_API_KEY is required for openrouter web search")
        if not self._model_name:
            raise ValueError("WEB_SEARCH_MODEL_NAME is required for openrouter web search")

        payload = _openrouter_web_search_payload(
            model_name=self._model_name,
            query=query,
            engine=engine,
            max_results=max_results,
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "OpenRouter web search request payload: %s",
                json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True),
            )
        endpoint = self._endpoint_factory(
            self._configuration.resolved_api_key,
            self._configuration.base_url,
        )
        raw = await endpoint.chat_completion_raw(
            payload=payload,
            max_retries=self._configuration.max_retries,
            retry_delay_seconds=self._configuration.retry_delay_seconds,
            call_name="OpenRouter web search",
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "OpenRouter web search raw response: %s",
                json.dumps(raw, default=str, ensure_ascii=False, sort_keys=True),
            )
        return normalize_openrouter_web_search_result(
            raw=raw,
            query=query,
            engine=engine,
            max_results=max_results,
            default_model_name=self._model_name,
        )


# ---------------------------------------------------------------------------
# Payload / response helpers
# ---------------------------------------------------------------------------


def _openrouter_web_search_payload(
    *,
    model_name: str,
    query: str,
    engine: str,
    max_results: int,
    allowed_domains: Optional[List[str]],
    excluded_domains: Optional[List[str]],
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "engine": engine,
        "max_results": int(max_results),
    }
    if allowed_domains:
        parameters["allowed_domains"] = allowed_domains
    if excluded_domains:
        parameters["excluded_domains"] = excluded_domains

    return {
        "model": model_name,
        "messages": [{"role": "user", "content": query}],
        "tools": [
            {
                "type": "openrouter:web_search",
                "parameters": parameters,
            }
        ],
    }


def normalize_openrouter_web_search_result(
    *,
    raw: Dict[str, Any],
    query: str,
    engine: str,
    max_results: int,
    default_model_name: str,
) -> Dict[str, Any]:
    answer = _extract_openrouter_answer(raw)
    sources = _extract_sources_from_openrouter_response(raw)
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "search_params": {
            "engine": engine,
            "max_results": int(max_results),
        },
        "usage": _extract_openrouter_usage(raw),
        "model": str(raw.get("model") or default_model_name),
    }


def _extract_openrouter_answer(raw: Dict[str, Any]) -> str:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _extract_openrouter_usage(raw: Dict[str, Any]) -> Dict[str, Any]:
    usage = raw.get("usage")
    return usage if isinstance(usage, dict) else {}


def _extract_sources_from_openrouter_response(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations = raw.get("citations")
    if not isinstance(citations, list):
        return []

    sources = [_citation_to_source(idx, item) for idx, item in enumerate(citations, start=1)]
    return [source for source in sources if source is not None]


def _citation_to_source(index: int, item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"rank": index, "url": item, "title": "", "snippet": ""}
    if isinstance(item, dict) and item.get("url"):
        return _citation_source_from_dict(index, item)
    return None


def _citation_source_from_dict(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rank": index,
        "url": str(item.get("url") or ""),
        "title": str(item.get("title") or ""),
        "snippet": str(item.get("snippet") or item.get("text") or ""),
    }


def web_search_model(
    *,
    provider: Any,
    configuration: OpenAIConfiguration,
    model_name: str,
    endpoint_factory: Callable[[Optional[str], Optional[str]], OpenAICompatibleEndpoint],
    default_engine: str = "auto",
    default_max_results: int = 5,
) -> WebSearchModel:
    key = str(getattr(provider, "value", provider)).strip().lower()
    if key == "openrouter":
        return OpenRouterWebSearchAdapter(
            configuration=configuration,
            model_name=model_name,
            endpoint_factory=endpoint_factory,
            default_engine=default_engine,
            default_max_results=default_max_results,
        )
    raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER: {provider}")
