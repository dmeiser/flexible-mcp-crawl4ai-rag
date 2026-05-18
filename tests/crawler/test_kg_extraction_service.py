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
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
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
        logger.warning.assert_called_once()

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
                return_value=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
                )
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
