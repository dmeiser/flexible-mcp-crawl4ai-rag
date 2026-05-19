"""Unit tests for src/services/kg_extraction_service.py — 100% coverage, offline."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.services.kg_extraction_service import KnowledgeGraphExtractionService

_VALID_JSON = (
    '{"entities": [{"name": "FastMCP", "type": "TOOL", "description": "An MCP framework."}], '
    '"relationships": [{"source": "FastMCP", "relationship": "implements", "target": "MCP"}]}'
)


def _make_settings(model_name: str = "gpt-4o") -> SimpleNamespace:
    return SimpleNamespace(
        effective_kg_model_name=model_name,
        effective_kg_api_key="key",
        effective_kg_base_url="http://llm",
    )


def _make_endpoint(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat_completion=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        )
    )


class TestKnowledgeGraphExtractionService:
    @pytest.mark.asyncio
    async def test_extract_valid_response(self):
        endpoint = _make_endpoint(_VALID_JSON)
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "FastMCP"
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["relationship"] == "implements"

    @pytest.mark.asyncio
    async def test_extract_invalid_json_returns_empty(self):
        endpoint = _make_endpoint("not json at all")
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_llm_error_returns_empty(self):
        endpoint = SimpleNamespace(chat_completion=AsyncMock(side_effect=RuntimeError("network error")))
        logger = MagicMock()
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint, logger=logger)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}
        logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_missing_keys_returns_empty(self):
        endpoint = _make_endpoint('{"nodes": [], "edges": []}')
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_no_model_returns_empty(self):
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: None)
        result = await svc.extract_knowledge_graph(_make_settings(model_name=""), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_non_string_content_returns_empty(self):
        endpoint = SimpleNamespace(
            chat_completion=AsyncMock(
                return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
            )
        )
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_non_dict_json_returns_empty(self):
        endpoint = _make_endpoint("[1, 2, 3]")
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    def test_extraction_prompt_contains_text(self):
        prompt = KnowledgeGraphExtractionService._extraction_prompt("my document text")
        assert "my document text" in prompt
        assert "entities" in prompt
        assert "relationships" in prompt

    @pytest.mark.asyncio
    async def test_extract_error_without_logger_is_silent(self):
        endpoint = SimpleNamespace(chat_completion=AsyncMock(side_effect=RuntimeError("boom")))
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_json_prefixed_fence_stripped(self):
        """Content starting with 'json' after fence stripping is handled (line 77 coverage)."""
        content = "```json\n" + _VALID_JSON + "\n```"
        # After strip("`").strip() we get: "json\n{...}"  → startswith("json") is True
        endpoint = _make_endpoint(content)
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert len(result["entities"]) == 1

    @pytest.mark.asyncio
    async def test_extract_entities_not_list_returns_empty(self):
        """Payload with non-list 'entities' field returns empty (line 84 coverage)."""
        bad_json = '{"entities": "not a list", "relationships": []}'
        endpoint = _make_endpoint(bad_json)
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_no_json_delimiters_returns_empty(self):
        """Content with no {{ or }} returns empty (empty-delimiters branch coverage)."""
        endpoint = _make_endpoint("no braces here at all")
        svc = KnowledgeGraphExtractionService(endpoint_factory=lambda **_kw: endpoint)
        result = await svc.extract_knowledge_graph(_make_settings(), "text", "https://x.com")
        assert result == {"entities": [], "relationships": []}


class TestOpenAIEndpointAdapter:
    """Tests for OpenAIEndpointAdapter — covers __init__ and chat_completion retry loop."""

    @pytest.mark.asyncio
    async def test_adapter_init_stores_client(self):
        """OpenAIEndpointAdapter.__init__ stores the client (line 11 coverage)."""
        from src.services.kg_extraction_service import OpenAIEndpointAdapter

        mock_client = MagicMock()
        adapter = OpenAIEndpointAdapter(mock_client)
        assert adapter._client is mock_client

    @pytest.mark.asyncio
    async def test_adapter_chat_completion_success(self):
        """Successful call on first attempt returns response (lines 20-28 coverage)."""
        from src.services.kg_extraction_service import OpenAIEndpointAdapter

        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        adapter = OpenAIEndpointAdapter(mock_client)
        result = await adapter.chat_completion({"model": "gpt-4", "messages": []})
        assert result is mock_response
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_chat_completion_retries_then_succeeds(self):
        """Retry on failure then success (covers retry loop body, lines 20-28)."""
        from src.services.kg_extraction_service import OpenAIEndpointAdapter

        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[RuntimeError("transient"), mock_response]
        )
        adapter = OpenAIEndpointAdapter(mock_client)
        result = await adapter.chat_completion(
            {"model": "gpt-4", "messages": []},
            max_retries=2,
            retry_delay_seconds=0.0,
        )
        assert result is mock_response
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_adapter_chat_completion_exhausts_retries(self):
        """All retries fail → raises the last exception."""
        from src.services.kg_extraction_service import OpenAIEndpointAdapter

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("always fails"))
        adapter = OpenAIEndpointAdapter(mock_client)
        with pytest.raises(RuntimeError, match="always fails"):
            await adapter.chat_completion(
                {"model": "gpt-4", "messages": []},
                max_retries=2,
                retry_delay_seconds=0.0,
            )
