import json
from typing import Any, Callable, Dict


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
            payload = json.loads(content.strip())
            if not isinstance(payload, dict):
                return empty
            if not isinstance(payload.get("entities"), list) or not isinstance(payload.get("relationships"), list):
                return empty
            return {"entities": payload["entities"], "relationships": payload["relationships"]}
        except Exception as exc:
            if self._logger:
                self._logger.warning(f"KG extraction LLM call failed for {source_url}: {exc}")
            return empty
