import asyncio
import json
from typing import Any, Callable, Dict


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

    async def extract_knowledge_graph(self, settings: Any, text: str, source_url: str) -> Dict[str, Any]:
        empty: Dict[str, Any] = {"entities": [], "relationships": []}
        if not settings.effective_kg_model_name:
            return empty
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
                },
                max_retries=3,
                retry_delay_seconds=1.0,
                call_name="kg extraction LLM",
            )
            content = resp.choices[0].message.content
            if not isinstance(content, str):
                return empty
            # Strip markdown code fences and extract the outermost JSON object
            stripped = content.strip().strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            if start == -1 or end == 0:
                return empty
            payload = json.loads(stripped[start:end])
            if not isinstance(payload, dict):
                return empty
            if not isinstance(payload.get("entities"), list) or not isinstance(payload.get("relationships"), list):
                return empty
            return {"entities": payload["entities"], "relationships": payload["relationships"]}
        except Exception as exc:
            if self._logger:
                self._logger.warning(f"KG extraction LLM call failed for {source_url}: {exc}")
            return empty
