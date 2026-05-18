"""Unit tests for src/services/graph_storage_service.py — 100% coverage, offline."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.services.graph_storage_service import _upsert_graph_edge, _upsert_graph_node, store_knowledge_graph


def _make_session(node_id: int = 1) -> MagicMock:
    session = MagicMock()
    row = (node_id,)
    result_mock = MagicMock()
    result_mock.fetchone.return_value = row
    session.execute.return_value = result_mock
    return session


async def _fake_embed(text: str) -> list:
    return [0.1, 0.2, 0.3, 0.4]


class TestStoreKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_store_empty_kg_is_noop(self):
        session = _make_session()
        await store_knowledge_graph(
            session=session,
            kg_data={"entities": [], "relationships": []},
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_knowledge_graph_creates_nodes(self):
        session = _make_session(node_id=42)
        kg_data = {
            "entities": [{"name": "FastMCP", "type": "TOOL", "description": "A framework."}],
            "relationships": [],
        }
        await store_knowledge_graph(
            session=session,
            kg_data=kg_data,
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        assert session.execute.call_count == 1
        sql_called = str(session.execute.call_args_list[0][0][0])
        assert "graph_nodes" in sql_called or "INSERT" in sql_called

    @pytest.mark.asyncio
    async def test_store_knowledge_graph_creates_edges(self):
        session = MagicMock()
        node_a_result = MagicMock()
        node_a_result.fetchone.return_value = (1,)
        node_b_result = MagicMock()
        node_b_result.fetchone.return_value = (2,)
        edge_result = MagicMock()
        edge_result.fetchone.return_value = None
        session.execute.side_effect = [node_a_result, node_b_result, edge_result]

        kg_data = {
            "entities": [
                {"name": "FastMCP", "type": "TOOL", "description": "A framework."},
                {"name": "MCP", "type": "CONCEPT", "description": "A protocol."},
            ],
            "relationships": [
                {"source": "FastMCP", "relationship": "implements", "target": "MCP"}
            ],
        }
        await store_knowledge_graph(
            session=session,
            kg_data=kg_data,
            source_url="https://x.com",
            chunk_id=5,
            create_embedding_fn=_fake_embed,
        )
        assert session.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_store_skips_unknown_entity_edges(self):
        session = _make_session(node_id=1)
        kg_data = {
            "entities": [{"name": "FastMCP", "type": "TOOL", "description": "A framework."}],
            "relationships": [
                {"source": "FastMCP", "relationship": "uses", "target": "UnknownEntity"}
            ],
        }
        await store_knowledge_graph(
            session=session,
            kg_data=kg_data,
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        assert session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_store_skips_entities_with_empty_name(self):
        session = _make_session()
        kg_data = {
            "entities": [{"name": "", "type": "TOOL", "description": "nameless"}],
            "relationships": [],
        }
        await store_knowledge_graph(
            session=session,
            kg_data=kg_data,
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_skips_incomplete_relationships(self):
        session = _make_session(node_id=1)
        kg_data = {
            "entities": [{"name": "A", "type": "CONCEPT", "description": "desc"}],
            "relationships": [
                {"source": "A", "relationship": "", "target": "B"},
                {"source": "", "relationship": "uses", "target": "B"},
                {"source": "A", "relationship": "uses", "target": ""},
            ],
        }
        await store_knowledge_graph(
            session=session,
            kg_data=kg_data,
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        assert session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_upsert_graph_node_returns_id(self):
        session = _make_session(node_id=99)
        entity = {"name": "TestEntity", "type": "ORG", "description": "A test org."}
        node_id = await _upsert_graph_node(session, entity, "https://x.com", None, [0.1, 0.2, 0.3, 0.4])
        assert node_id == 99

    @pytest.mark.asyncio
    async def test_upsert_graph_edge_calls_execute(self):
        session = MagicMock()
        session.execute.return_value = MagicMock()
        await _upsert_graph_edge(session, 1, 2, "uses", "https://x.com", None)
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_none_entities_is_noop(self):
        session = _make_session()
        await store_knowledge_graph(
            session=session,
            kg_data={"entities": None, "relationships": None},
            source_url="https://x.com",
            chunk_id=None,
            create_embedding_fn=_fake_embed,
        )
        session.execute.assert_not_called()
