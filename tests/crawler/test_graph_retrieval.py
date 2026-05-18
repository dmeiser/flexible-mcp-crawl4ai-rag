"""Unit tests for src/services/graph_retrieval.py — 100% coverage, offline."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.services.graph_retrieval import search_knowledge_graph


async def _fake_embed(text: str) -> list:
    return [0.1, 0.2, 0.3, 0.4]


def _make_session(seed_rows=None, neighborhood_rows=None, node_rows=None, content_rows=None):
    """Build a mock session that returns different results per execute() call in order."""
    session = MagicMock()

    def _make_result(rows):
        r = MagicMock()
        r.fetchall.return_value = rows or []
        return r

    session.execute.side_effect = [
        _make_result(seed_rows or []),
        _make_result(neighborhood_rows or []),
        _make_result(node_rows or []),
        _make_result(content_rows or []),
    ]
    return session


class TestSearchKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_search_empty_graph_returns_empty(self):
        session = MagicMock()
        r = MagicMock()
        r.fetchall.return_value = []
        session.execute.return_value = r
        results = await search_knowledge_graph(session, "query", _fake_embed)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.95)]
        neighborhood_rows = [(2, ["implements"])]
        node_rows = [
            (1, "FastMCP", "TOOL", "A framework.", "https://x.com"),
            (2, "MCP", "CONCEPT", "A protocol.", "https://x.com"),
        ]
        content_rows = [("https://x.com", "FastMCP implements the MCP protocol.")]

        session = _make_session(seed_rows, neighborhood_rows, node_rows, content_rows)
        results = await search_knowledge_graph(session, "query", _fake_embed, match_count=5, depth=2)

        assert len(results) == 2
        seed_result = next(r for r in results if r["entity_name"] == "FastMCP")
        assert seed_result["similarity_score"] == pytest.approx(0.95)
        assert seed_result["content"] == "FastMCP implements the MCP protocol."
        assert seed_result["graph_context"].startswith("FastMCP")
        assert seed_result["relationship_chain"] == []

        neighbor_result = next(r for r in results if r["entity_name"] == "MCP")
        assert neighbor_result["relationship_chain"] == ["implements"]
        assert neighbor_result["similarity_score"] == 0.0

    @pytest.mark.asyncio
    async def test_search_with_entity_type_filter(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.9)]
        session = _make_session(seed_rows, [], [(1, "FastMCP", "TOOL", "A framework.", "https://x.com")], [])
        results = await search_knowledge_graph(
            session, "query", _fake_embed, match_count=5, depth=2, entity_type_filter="TOOL"
        )
        sql_call = session.execute.call_args_list[0]
        params = sql_call[0][1]
        assert params["type_filter"] == "TOOL"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_depth_1(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.9)]
        session = _make_session(seed_rows, [], [(1, "FastMCP", "TOOL", "A framework.", "https://x.com")], [])
        await search_knowledge_graph(session, "query", _fake_embed, match_count=5, depth=1)
        neighborhood_call = session.execute.call_args_list[1]
        params = neighborhood_call[0][1]
        assert params["max_depth"] == 1

    @pytest.mark.asyncio
    async def test_search_missing_node_info_skipped(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.9)]
        neighborhood_rows = [(99, ["links"])]
        node_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com")]
        session = _make_session(seed_rows, neighborhood_rows, node_rows, [])
        results = await search_knowledge_graph(session, "query", _fake_embed)
        entity_names = [r["entity_name"] for r in results]
        assert "FastMCP" in entity_names
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_deduplicates_seed_and_neighbor(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.9)]
        neighborhood_rows = [(1, ["self_ref"])]
        node_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://x.com")]
        session = _make_session(seed_rows, neighborhood_rows, node_rows, [])
        results = await search_knowledge_graph(session, "query", _fake_embed)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_content_missing_returns_empty_string(self):
        seed_rows = [(1, "FastMCP", "TOOL", "A framework.", "https://no-content.com", 0.8)]
        session = _make_session(
            seed_rows,
            [],
            [(1, "FastMCP", "TOOL", "A framework.", "https://no-content.com")],
            [],
        )
        results = await search_knowledge_graph(session, "query", _fake_embed)
        assert results[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_duplicate_seed_ids_deduped(self):
        seed_rows = [
            (1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.9),
            (1, "FastMCP", "TOOL", "A framework.", "https://x.com", 0.8),
        ]
        session = _make_session(
            seed_rows,
            [],
            [(1, "FastMCP", "TOOL", "A framework.", "https://x.com")],
            [("https://x.com", "content")],
        )
        results = await search_knowledge_graph(session, "query", _fake_embed)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_seed_with_no_node_info_skipped(self):
        seed_rows = [(99, "Ghost", "CONCEPT", "Missing from node_rows.", "https://x.com", 0.7)]
        session = _make_session(seed_rows, [], [], [])
        results = await search_knowledge_graph(session, "query", _fake_embed)
        assert results == []
