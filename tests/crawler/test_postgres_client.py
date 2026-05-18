"""Unit tests for src/services/ingestion.py — 100% coverage, offline."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.services.ingestion import store_crawled_documents

# ---------------------------------------------------------------------------
# Tests: store_crawled_documents
# ---------------------------------------------------------------------------


class TestStoreCrawledDocuments:
    @pytest.mark.asyncio
    async def test_no_markdown_skips_add(self):
        """Results with empty markdown are skipped; add_documents_to_db not called."""
        session = MagicMock()
        crawl_results = [{"url": "https://x.com", "markdown": ""}]

        with patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock) as mock_add:
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 0
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_markdown_key_skips_add(self):
        """Results missing the 'markdown' key are skipped."""
        session = MagicMock()
        crawl_results = [{"url": "https://x.com"}]

        with patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock) as mock_add:
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 0
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_result_stored(self):
        """A single page with content gets chunked and stored."""
        session = MagicMock()
        crawl_results = [{"url": "https://x.com/page", "markdown": "# Title\n\nSome content."}]

        with (
            patch(
                "src.services.ingestion.chunk_text_with_heading_metadata",
                new_callable=AsyncMock,
                return_value=[("# Title\n\nSome content.", {"heading_path": [], "heading_level": 0})],
            ) as mock_chunk,
            patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock, return_value=1) as mock_add,
        ):
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 1
        mock_chunk.assert_called_once_with("# Title\n\nSome content.")
        mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_results_batched_together(self):
        """Two pages are chunked; add_documents_to_db is called once with all data."""
        session = MagicMock()
        crawl_results = [
            {"url": "https://x.com/a", "markdown": "Para one.\n\nPara two."},
            {"url": "https://x.com/b", "markdown": "Single para."},
        ]

        async def fake_chunk(text):
            return [(p, {"heading_path": [], "heading_level": 0}) for p in text.split("\n\n") if p.strip()]

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock, return_value=3) as mock_add,
        ):
            pages, chunks = await store_crawled_documents(session, crawl_results, "sitemap")

        assert pages == 2
        # 2 chunks from /a  +  1 chunk from /b  =  3 total
        assert chunks == 3
        mock_add.assert_called_once()
        # all three chunks were passed in the single call
        _, call_urls, call_contents, *_ = mock_add.call_args.args
        assert len(call_urls) == 3
        assert len(call_contents) == 3

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Metadata dict contains the expected fields for each chunk."""
        session = MagicMock()
        crawl_results = [{"url": "https://example.com/page", "markdown": "Hello world."}]
        captured_metas: list = []

        async def fake_chunk(text):
            return [(text, {"heading_path": [], "heading_level": 0})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            captured_metas.extend(metas)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            await store_crawled_documents(session, crawl_results, "webpage_single")

        assert len(captured_metas) == 1
        meta = captured_metas[0]
        assert meta["url"] == "https://example.com/page"
        assert meta["source"] == "example.com"
        assert meta["crawl_type"] == "webpage_single"
        assert meta["chunk_index"] == 0
        assert "crawl_time" in meta
        assert "crawl_timestamp" in meta
        assert meta["content_class"] == "text"
        assert meta["is_active"] is True
        assert isinstance(meta["content_hash"], str) and len(meta["content_hash"]) == 64
        assert "heading_path" in meta
        assert "heading_level" in meta

    @pytest.mark.asyncio
    async def test_reference_metadata_preserved(self):
        """Variant reference metadata is forwarded into stored chunk metadata."""
        session = MagicMock()
        crawl_results = [
            {
                "url": "https://example.com/page",
                "markdown": "Hello world.",
                "selected_variant": "raw_markdown",
                "variant_values": {
                    "references_markdown": "[1]: https://example.com/ref Example reference",
                    "markdown_with_citations": "Hello world [1]",
                },
            }
        ]
        captured_metas: list = []

        async def fake_chunk(text):
            return [(text, {"heading_path": [], "heading_level": 0})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            captured_metas.extend(metas)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            await store_crawled_documents(session, crawl_results, "webpage_single")

        assert captured_metas[0]["markdown_variant"] == "raw_markdown"
        assert captured_metas[0]["has_citations"] is True
        assert captured_metas[0]["link_references"][0]["url"] == "https://example.com/ref"

    @pytest.mark.asyncio
    async def test_empty_results_list(self):
        """Empty crawl_results returns (0, 0) without calling add_documents_to_db."""
        session = MagicMock()
        with patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock) as mock_add:
            pages, chunks = await store_crawled_documents(session, [], "webpage")

        assert pages == 0
        assert chunks == 0
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_empty_and_valid(self):
        """Empty-markdown entries are skipped; valid ones are stored."""
        session = MagicMock()
        crawl_results = [
            {"url": "https://x.com/empty", "markdown": ""},
            {"url": "https://x.com/good", "markdown": "Content."},
        ]

        async def fake_chunk(text):
            return [(text, {"heading_path": [], "heading_level": 0})]

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", new_callable=AsyncMock, return_value=1),
        ):
            pages, chunks = await store_crawled_documents(session, crawl_results, "sitemap")

        assert pages == 2  # total results in list
        assert chunks == 1  # only one chunk stored

    @pytest.mark.asyncio
    async def test_extra_metadata_is_forwarded(self):
        """Optional source/artifact metadata keys are forwarded when present."""
        session = MagicMock()
        crawl_results = [
            {
                "url": "https://example.com/with-meta",
                "markdown": "Hello.",
                "source_change_id": "etag:abc",
                "link_graph": {"total_links": 1},
                "media_metadata": {"image_count": 0},
                "session_id": "session-123",
                "run_id": "run-xyz",
            }
        ]
        captured_metas: list = []

        async def fake_chunk(text):
            return [(text, {"heading_path": [], "heading_level": 0})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            captured_metas.extend(metas)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 1
        assert captured_metas[0]["source_change_id"] == "etag:abc"
        assert captured_metas[0]["link_graph"]["total_links"] == 1
        assert captured_metas[0]["media_metadata"]["image_count"] == 0
        assert captured_metas[0]["session_id"] == "session-123"
        assert captured_metas[0]["run_id"] == "run-xyz"

    @pytest.mark.asyncio
    async def test_heading_path_metadata_populated(self):
        """Heading path metadata is stored when chunking produces heading info."""
        session = MagicMock()
        crawl_results = [{"url": "https://example.com/doc", "markdown": "# Guide\nContent."}]
        captured_metas: list = []

        async def fake_chunk(text):
            return [("Content.", {"heading_path": ["Guide"], "heading_level": 1})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            captured_metas.extend(metas)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            await store_crawled_documents(session, crawl_results, "webpage")

        assert captured_metas[0]["heading_path"] == ["Guide"]
        assert captured_metas[0]["heading_level"] == 1

    @pytest.mark.asyncio
    async def test_embed_text_includes_heading_prefix(self):
        """embed_texts passed to add_documents_to_db include heading prefix when path is set."""
        session = MagicMock()
        crawl_results = [{"url": "https://example.com/doc", "markdown": "# Guide\nContent."}]
        captured_embed_texts: list = []

        async def fake_chunk(text):
            return [("Content.", {"heading_path": ["Guide"], "heading_level": 1})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            if embed_texts:
                captured_embed_texts.extend(embed_texts)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            await store_crawled_documents(session, crawl_results, "webpage")

        assert len(captured_embed_texts) == 1
        assert "[Guide]" in captured_embed_texts[0]
        assert "Content." in captured_embed_texts[0]

    @pytest.mark.asyncio
    async def test_embed_text_no_heading_is_plain_chunk(self):
        """embed_texts is the plain chunk text when no heading path."""
        session = MagicMock()
        crawl_results = [{"url": "https://example.com/plain", "markdown": "Plain text."}]
        captured_embed_texts: list = []

        async def fake_chunk(text):
            return [("Plain text.", {"heading_path": [], "heading_level": 0})]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs, embed_texts=None):
            if embed_texts:
                captured_embed_texts.extend(embed_texts)
            return 1

        with (
            patch("src.services.ingestion.chunk_text_with_heading_metadata", side_effect=fake_chunk),
            patch("src.services.ingestion.add_documents_to_db", side_effect=capture_add),
        ):
            await store_crawled_documents(session, crawl_results, "webpage")

        assert captured_embed_texts == ["Plain text."]


# ---------------------------------------------------------------------------
# Tests: _index_knowledge_graphs
# ---------------------------------------------------------------------------


class TestIndexKnowledgeGraphs:
    @pytest.mark.asyncio
    async def test_no_op_when_graph_index_disabled(self):
        """Returns immediately when USE_GRAPH_INDEX is False."""
        from src.services.ingestion import _index_knowledge_graphs

        session = MagicMock()
        mock_settings = MagicMock()
        mock_settings.USE_GRAPH_INDEX = False

        with patch("src.config.settings", mock_settings):
            await _index_knowledge_graphs(session, ["https://x.com"], ["content"], MagicMock())

        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_no_model_name(self):
        """Returns immediately when no KG model name is configured."""
        from src.services.ingestion import _index_knowledge_graphs

        session = MagicMock()
        mock_settings = MagicMock()
        mock_settings.USE_GRAPH_INDEX = True
        mock_settings.effective_kg_model_name = None

        with patch("src.config.settings", mock_settings):
            await _index_knowledge_graphs(session, ["https://x.com"], ["content"], MagicMock())

        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_no_endpoint_factory(self):
        """Returns immediately when endpoint_factory is None."""
        from src.services.ingestion import _index_knowledge_graphs

        session = MagicMock()
        mock_settings = MagicMock()
        mock_settings.USE_GRAPH_INDEX = True
        mock_settings.effective_kg_model_name = "some-model"

        with patch("src.config.settings", mock_settings):
            await _index_knowledge_graphs(session, ["https://x.com"], ["content"], None)

        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_extracts_and_stores_kg(self):
        """When enabled, extracts KG and stores it for each URL."""
        from src.services.ingestion import _index_knowledge_graphs

        session = MagicMock()
        mock_settings = MagicMock()
        mock_settings.USE_GRAPH_INDEX = True
        mock_settings.effective_kg_model_name = "some-model"
        endpoint_factory = MagicMock()

        kg_data = {
            "entities": [{"entity_name": "Python", "entity_type": "language", "description": "a language"}],
            "relationships": [],
        }

        mock_extractor = MagicMock()
        mock_extractor.extract_knowledge_graph = AsyncMock(return_value=kg_data)

        with (
            patch("src.config.settings", mock_settings),
            patch(
                "src.services.kg_extraction_service.KnowledgeGraphExtractionService",
                return_value=mock_extractor,
            ),
            patch("src.services.graph_storage_service.store_knowledge_graph", new_callable=AsyncMock) as mock_store,
            patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2]),
        ):
            await _index_knowledge_graphs(session, ["https://x.com/page"], ["page content"], endpoint_factory)

        mock_extractor.extract_knowledge_graph.assert_called_once()
        mock_store.assert_called_once()
