import asyncio
import json
from typing import Any, Callable, Dict, Optional, Tuple


def _json_object_bounds(text: str) -> Optional[Tuple[int, int]]:
    """Return (start, end) slice indices of the outermost ``{...}`` in *text*, or ``None``."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    return start, end


def _validate_kg_payload(payload: Any) -> bool:
    return isinstance(payload.get("entities"), list) and isinstance(payload.get("relationships"), list)


def _parse_kg_json_content(content: str) -> Optional[Dict[str, Any]]:
    """Strip markdown fences and parse the outermost JSON object from *content*.

    Returns a dict with ``entities`` and ``relationships`` lists, or ``None``
    if parsing fails or the structure is invalid.
    """
    stripped = content.strip().strip("`").strip()
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    bounds = _json_object_bounds(stripped)
    if bounds is None:
        return None
    payload = json.loads(stripped[bounds[0] : bounds[1]])
    if not _validate_kg_payload(payload):
        return None
    return {"entities": payload["entities"], "relationships": payload["relationships"]}


class OpenAIEndpointAdapter:
    """Thin adapter so AsyncOpenAI satisfies the chat_completion(request_kwargs, ...) interface
    expected by KnowledgeGraphExtractionService (and mockable in tests)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def chat_completion(
        self,
        request_kwargs: Dict[str, Any],
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        call_name: str = "",
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay_seconds)
        raise last_exc  # type: ignore[misc]


class KnowledgeGraphExtractionService:
    """Service for extracting entities and relationships from text using an LLM."""

    def __init__(self, endpoint_factory: Callable[..., Any], logger: Any = None) -> None:
        self._endpoint_factory = endpoint_factory
        self._logger = logger

    @staticmethod
    def _extraction_prompt(text: str) -> str:
        return (
            "Extract all named entities and relationships from the following document.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            '{"entities": [{"name": "<string>", "type": "<PERSON|ORG|CONCEPT|API|TOOL|TECHNOLOGY|OTHER>", '
            '"description": "<one sentence>"}], '
            '"relationships": [{"source": "<entity name>", "relationship": "<verb phrase>", '
            '"target": "<entity name>"}]}\n'
            "Do not include any text outside the JSON.\n\n"
            f"Document:\n{text[:20000]}"
        )

    async def _call_kg_endpoint(self, settings: Any, text: str, source_url: str) -> Optional[str]:
        """Call the LLM and return the raw content string, or ``None`` on any error."""
        try:
            endpoint = self._endpoint_factory(
                api_key=settings.effective_kg_api_key,
                base_url=settings.effective_kg_base_url,
            )
            resp = await endpoint.chat_completion(
                request_kwargs={
                    "model": settings.effective_kg_model_name,
                    "messages": [{"role": "user", "content": self._extraction_prompt(text)}],
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
                max_retries=3,
                retry_delay_seconds=1.0,
                call_name="kg extraction LLM",
            )
            content = resp.choices[0].message.content
            return content if isinstance(content, str) else None
        except Exception as exc:
            if self._logger:
                self._logger.error("KG extraction LLM call failed for %s: %s", source_url, exc, exc_info=True)
            return None

    async def extract_knowledge_graph(self, settings: Any, text: str, source_url: str) -> Dict[str, Any]:
        empty: Dict[str, Any] = {"entities": [], "relationships": []}
        if not settings.effective_kg_model_name:
            return empty
        content = await self._call_kg_endpoint(settings, text, source_url)
        if content is None:
            return empty
        parsed = _parse_kg_json_content(content)
        return parsed if parsed is not None else empty
