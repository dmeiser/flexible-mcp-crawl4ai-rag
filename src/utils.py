"""
Wiring layer: provider factories, service singletons, and facade functions.

All configuration, models, enums, and domain logic have been extracted into
dedicated modules.  This module remains as the application-level glue that
wires providers to services and exposes convenience facade functions consumed
by the rest of the codebase.

Re-exports are provided for backward compatibility so that existing
``from src.utils import X`` statements continue to work.
"""

import asyncio
import logging
from typing import List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

# ---------------------------------------------------------------------------
# Re-exports — keep ``from src.utils import X`` working everywhere
# ---------------------------------------------------------------------------
from src.config import (  # noqa: F401
    ChunkStrategy,
    ContentClass,
    MarkdownIndexPolicy,
    Settings,
    settings,
)
from src.exceptions import EmbeddingError  # noqa: F401
from src.models import (  # noqa: F401
    CodeExample,
    CrawledPage,
    EvictionAuditLog,
    Source,
    SourcePolicy,
    StoragePolicy,
    engine,
    get_session,
)
from src.providers.openai_stack import (
    ChatCompletionRetryStrategy,
    EmbeddingsProvider,
    OpenAICompatibleEndpoint,
    OpenAIConfiguration,
)
from src.services.contextual_enrichment_service import ContextualEnrichmentService
from src.services.embedding_service import EmbeddingService
from src.services.scoring_service import compute_staleness_score, compute_value_score  # noqa: F401
from src.services.tombstone_service import (  # noqa: F401
    _extract_record_source,
    get_db_size_bytes,
    tombstone_records,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider factory functions
# ---------------------------------------------------------------------------
_async_chat_completion_with_retries = ChatCompletionRetryStrategy.async_chat_completion_with_retries
_sync_chat_completion_with_retries = ChatCompletionRetryStrategy.sync_chat_completion_with_retries
_retry_backoff_seconds = ChatCompletionRetryStrategy.retry_backoff_seconds


def _openai_configuration(
    api_key: Optional[str],
    base_url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    timeout_seconds: float = 90.0,
) -> OpenAIConfiguration:
    return OpenAIConfiguration(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        timeout_seconds=timeout_seconds,
    )


def _openai_compatible_endpoint(api_key: Optional[str], base_url: Optional[str] = None) -> OpenAICompatibleEndpoint:
    return OpenAICompatibleEndpoint(
        _openai_configuration(api_key=api_key, base_url=base_url),
        async_openai_cls=AsyncOpenAI,
        openai_cls=OpenAI,
        async_chat_retry_fn=_async_chat_completion_with_retries,
        sync_chat_retry_fn=_sync_chat_completion_with_retries,
    )


def _embeddings_provider_from_settings() -> EmbeddingsProvider:
    configuration = _openai_configuration(
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.effective_embedding_base_url,
        max_retries=settings.effective_embedding_max_retries,
        retry_delay_seconds=settings.effective_embedding_retry_delay_seconds,
    )
    return EmbeddingsProvider(
        configuration=configuration,
        model_name=settings.effective_embedding_model_name,
        normalize_fn=_normalize,
        async_openai_cls=AsyncOpenAI,
        openai_cls=OpenAI,
        async_chat_retry_fn=_async_chat_completion_with_retries,
        sync_chat_retry_fn=_sync_chat_completion_with_retries,
    )


# ---------------------------------------------------------------------------
# Service singletons
# ---------------------------------------------------------------------------
_embedding_service = EmbeddingService(
    provider_factory=_embeddings_provider_from_settings,
    error_cls=EmbeddingError,
)
_contextual_enrichment_service = ContextualEnrichmentService(
    endpoint_factory=_openai_compatible_endpoint,
    logger=logger,
)


# ---------------------------------------------------------------------------
# Embedding facade
# ---------------------------------------------------------------------------
def _normalize(vec: List[float]) -> List[float]:
    return EmbeddingService.normalize(vec)


async def create_embedding(text: str) -> List[float]:
    """Create a normalized embedding for a single text string."""
    if not text or not text.strip():
        raise EmbeddingError("Attempted to create embedding for empty string.")
    return await _create_openai_embedding(text)


async def _create_openai_embedding(text: str) -> List[float]:
    return await _embedding_service.create_openai_embedding(text)


async def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Create embeddings for multiple texts in parallel."""
    if not texts:
        return []
    return await asyncio.gather(*[create_embedding(t) for t in texts])


# ---------------------------------------------------------------------------
# Contextual enrichment facade
# ---------------------------------------------------------------------------
async def generate_contextual_text(full_document: str, chunk: str) -> Tuple[str, bool]:
    """Generate a contextual summary prefix for a chunk using an LLM."""
    if not settings.USE_CONTEXTUAL_EMBEDDINGS or not settings.effective_contextual_model_name:
        return chunk, False

    try:
        context_text = await _request_contextual_summary(full_document, chunk)
        enriched = _combine_context_and_chunk(context_text, chunk)
        if enriched is not None:
            return enriched, True
        return chunk, False
    except Exception as exc:
        logger.warning(f"Contextual embedding LLM call failed: {exc}")
        return chunk, False


def _context_prompt(full_document: str, chunk: str) -> str:
    return ContextualEnrichmentService.context_prompt(full_document, chunk)


async def _request_contextual_summary(full_document: str, chunk: str) -> str:
    return await _contextual_enrichment_service.request_contextual_summary(settings, full_document, chunk)


def _combine_context_and_chunk(context_text: str, chunk: str) -> Optional[str]:
    return _contextual_enrichment_service.combine_context_and_chunk(settings, context_text, chunk)
