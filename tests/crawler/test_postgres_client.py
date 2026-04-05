"""Unit tests for src/crawler/postgres_client.py — 100% coverage, offline."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.crawler.postgres_client import store_crawled_documents


# ---------------------------------------------------------------------------
# Tests: store_crawled_documents
# ---------------------------------------------------------------------------

class TestStoreCrawledDocuments:
    @pytest.mark.asyncio
    async def test_no_markdown_skips_add(self):
        """Results with empty markdown are skipped; add_documents_to_db not called."""
        session = MagicMock()
        crawl_results = [{"url": "https://x.com", "markdown": ""}]

        with patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock) as mock_add:
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 0
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_markdown_key_skips_add(self):
        """Results missing the 'markdown' key are skipped."""
        session = MagicMock()
        crawl_results = [{"url": "https://x.com"}]

        with patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock) as mock_add:
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 0
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_result_stored(self):
        """A single page with content gets chunked and stored."""
        session = MagicMock()
        crawl_results = [
            {"url": "https://x.com/page", "markdown": "# Title\n\nSome content."}
        ]

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings",
                   new_callable=AsyncMock,
                   return_value=["# Title\n\nSome content."]) as mock_chunk, \
             patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock, return_value=1) as mock_add:
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
            return [p for p in text.split("\n\n") if p.strip()]

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings",
                   side_effect=fake_chunk), \
             patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock, return_value=3) as mock_add:
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
            return [text]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs):
            captured_metas.extend(metas)
            return 1

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings",
                   side_effect=fake_chunk), \
             patch("src.crawler.postgres_client.add_documents_to_db",
                   side_effect=capture_add):
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

    @pytest.mark.asyncio
    async def test_reference_metadata_preserved(self):
        """Variant reference metadata is forwarded into stored chunk metadata."""
        session = MagicMock()
        crawl_results = [{
            "url": "https://example.com/page",
            "markdown": "Hello world.",
            "selected_variant": "raw_markdown",
            "variant_values": {
                "references_markdown": "[1]: https://example.com/ref Example reference",
                "markdown_with_citations": "Hello world [1]",
            },
        }]
        captured_metas: list = []

        async def fake_chunk(text):
            return [text]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs):
            captured_metas.extend(metas)
            return 1

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings", side_effect=fake_chunk), \
             patch("src.crawler.postgres_client.add_documents_to_db", side_effect=capture_add):
            await store_crawled_documents(session, crawl_results, "webpage_single")

        assert captured_metas[0]["markdown_variant"] == "raw_markdown"
        assert captured_metas[0]["has_citations"] is True
        assert captured_metas[0]["link_references"][0]["url"] == "https://example.com/ref"

    @pytest.mark.asyncio
    async def test_empty_results_list(self):
        """Empty crawl_results returns (0, 0) without calling add_documents_to_db."""
        session = MagicMock()
        with patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock) as mock_add:
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
            return [text]

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings",
                   side_effect=fake_chunk), \
             patch("src.crawler.postgres_client.add_documents_to_db",
                   new_callable=AsyncMock, return_value=1):
            pages, chunks = await store_crawled_documents(session, crawl_results, "sitemap")

        assert pages == 2   # total results in list
        assert chunks == 1  # only one chunk stored

    @pytest.mark.asyncio
    async def test_extra_metadata_is_forwarded(self):
        """Optional source/artifact metadata keys are forwarded when present."""
        session = MagicMock()
        crawl_results = [{
            "url": "https://example.com/with-meta",
            "markdown": "Hello.",
            "source_change_id": "etag:abc",
            "link_graph": {"total_links": 1},
            "media_metadata": {"image_count": 0},
            "session_id": "session-123",
            "run_id": "run-xyz",
        }]
        captured_metas: list = []

        async def fake_chunk(text):
            return [text]

        async def capture_add(sess, urls, contents, metas, chunk_nums, full_docs):
            captured_metas.extend(metas)
            return 1

        with patch("src.crawler.postgres_client.chunk_text_according_to_settings", side_effect=fake_chunk), \
             patch("src.crawler.postgres_client.add_documents_to_db", side_effect=capture_add):
            pages, chunks = await store_crawled_documents(session, crawl_results, "webpage")

        assert pages == 1
        assert chunks == 1
        assert captured_metas[0]["source_change_id"] == "etag:abc"
        assert captured_metas[0]["link_graph"]["total_links"] == 1
        assert captured_metas[0]["media_metadata"]["image_count"] == 0
        assert captured_metas[0]["session_id"] == "session-123"
        assert captured_metas[0]["run_id"] == "run-xyz"


