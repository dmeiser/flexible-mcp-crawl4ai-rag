from typing import Any, Callable, Optional, Tuple


class ContextualEnrichmentService:
    """Service for contextual text enrichment using an LLM."""

    def __init__(self, endpoint_factory: Callable[..., Any], logger: Any) -> None:
        self._endpoint_factory = endpoint_factory
        self._logger = logger

    async def generate_contextual_text(self, settings: Any, full_document: str, chunk: str) -> Tuple[str, bool]:
        if not settings.USE_CONTEXTUAL_EMBEDDINGS or not settings.effective_contextual_model_name:
            return chunk, False

        try:
            context_text = await self.request_contextual_summary(settings, full_document, chunk)
            enriched = self.combine_context_and_chunk(settings, context_text, chunk)
            if enriched is not None:
                return enriched, True
            return chunk, False
        except Exception as exc:
            self._logger.warning(f"Contextual embedding LLM call failed: {exc}")
            return chunk, False

    @staticmethod
    def context_prompt(full_document: str, chunk: str) -> str:
        return (
            f"<document>\n{full_document[:25000]}\n</document>\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Given the document and the specific chunk, write a 1-2 sentence "
            "summary capturing the chunk's topic and its relationship to the "
            "wider document. Use keywords that help semantic search.\n"
            "Summary:"
        )

    async def request_contextual_summary(self, settings: Any, full_document: str, chunk: str) -> str:
        if not settings.effective_contextual_model_name:
            self._logger.warning(
                "CONTEXTUAL_LLM_MODEL_NAME (or DEFAULT_LLM_MODEL_NAME) "
                "is not configured; skipping contextual enrichment."
            )
            return ""

        endpoint = self._endpoint_factory(
            api_key=settings.effective_contextual_api_key,
            base_url=settings.effective_contextual_base_url,
        )
        resp = await endpoint.chat_completion(
            request_kwargs={
                "model": settings.effective_contextual_model_name,
                "messages": [{"role": "user", "content": self.context_prompt(full_document, chunk)}],
                "max_tokens": 150,
            },
            max_retries=settings.effective_contextual_max_retries,
            retry_delay_seconds=settings.effective_contextual_retry_delay_seconds,
            call_name="contextual LLM",
        )

        content = resp.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""

    def combine_context_and_chunk(self, settings: Any, context_text: str, chunk: str) -> Optional[str]:
        if not context_text:
            return None
        combined = f"Context: {context_text}\n\n{chunk}"
        return combined[: settings.CHUNK_SIZE * 2]
