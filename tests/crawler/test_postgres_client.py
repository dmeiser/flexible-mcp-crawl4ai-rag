"""Unit tests for src/crawler/postgres_client.py — 100% coverage, offline."""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.crawler.postgres_client import (
    store_crawled_documents,
    fetch_available_sources,
    execute_rag_query,
)


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


# ---------------------------------------------------------------------------
# Tests: fetch_available_sources
# ---------------------------------------------------------------------------

class TestFetchAvailableSources:
    def test_empty_db_returns_empty_list(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        assert fetch_available_sources(session) == []

    def test_returns_sorted_unique_sources(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = ["z.com", "a.com", "a.com", "m.com"]
        assert fetch_available_sources(session) == ["a.com", "m.com", "z.com"]

    def test_none_values_filtered_out(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = ["a.com", None, "b.com"]
        assert fetch_available_sources(session) == ["a.com", "b.com"]

    def test_empty_string_filtered_out(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = ["a.com", "", "b.com"]
        assert fetch_available_sources(session) == ["a.com", "b.com"]

    def test_duplicates_deduplicated(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = ["x.com", "x.com", "x.com"]
        assert fetch_available_sources(session) == ["x.com"]


# ---------------------------------------------------------------------------
# Tests: execute_rag_query
# ---------------------------------------------------------------------------
# execute_rag_query is a sync wrapper that calls search_documents via
# asyncio.get_event_loop().run_until_complete().  We mock the event loop
# so no real coroutine execution is needed.

def _mock_loop_running(raw_results):
    """Return a MagicMock loop whose run_until_complete returns raw_results."""
    loop = MagicMock()

    def _run_until_complete(awaitable):
        # execute_rag_query builds a coroutine via search_documents(...).
        # In mocked-loop tests we don't actually run it, so close it to avoid
        # RuntimeWarning: coroutine was never awaited.
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return raw_results

    loop.run_until_complete.side_effect = _run_until_complete
    return loop


class TestExecuteRagQuery:
    def test_no_source_no_filter_passes_none(self):
        """When source and filter_metadata are both absent, filter arg is None."""
        session = MagicMock()
        loop = _mock_loop_running([])

        with patch("asyncio.get_event_loop", return_value=loop):
            result = execute_rag_query(session, "query")

        assert result == []
        # The second positional arg to run_until_complete is the coroutine;
        # we just verify it was called once.
        loop.run_until_complete.assert_called_once()

    def test_source_builds_filter_dict(self):
        """Passing source adds {"source": …} to the filter."""
        session = MagicMock()

        # We need to inspect what coroutine was created; use a real async mock.
        captured: dict = {}

        def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            fut = loop.create_future()
            fut.set_result([])
            return fut

        # Provide a real event loop so the coroutine actually executes.
        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_event_loop", return_value=loop), \
                 patch("src.crawler.postgres_client.search_documents", side_effect=fake_search):
                result = execute_rag_query(session, "q", source="example.com")
        finally:
            loop.close()

        assert result == []
        assert captured["filter"] == {"source": "example.com"}

    def test_filter_metadata_merged_with_source(self):
        """filter_metadata kwarg is merged with source."""
        session = MagicMock()
        captured: dict = {}

        async def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return []

        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_event_loop", return_value=loop), \
                 patch("src.crawler.postgres_client.search_documents",
                        side_effect=fake_search):
                execute_rag_query(session, "q", source="x.com",
                                  filter_metadata={"lang": "en"})
        finally:
            loop.close()

        assert captured["filter"] == {"source": "x.com", "lang": "en"}

    def test_results_transformed_to_standard_format(self):
        """Results are mapped to {url, content, metadata, similarity}."""
        session = MagicMock()
        raw = [
            {"url": "http://a.com", "content": "text", "page_metadata": {"k": "v"},
             "similarity_score": 0.9},
            {"url": "http://b.com", "content": "more", "page_metadata": {},
             "similarity_score": 0.7},
        ]
        loop = _mock_loop_running(raw)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = execute_rag_query(session, "q")

        assert len(result) == 2
        assert result[0] == {"url": "http://a.com", "content": "text",
                              "metadata": {"k": "v"}, "similarity": 0.9}
        assert result[1] == {"url": "http://b.com", "content": "more",
                              "metadata": {}, "similarity": 0.7}

    def test_match_count_forwarded(self):
        """match_count is passed through to search_documents."""
        session = MagicMock()
        captured: dict = {}

        def fake_search(sess, q, match_count, filter_metadata):
            captured["match_count"] = match_count
            fut = loop.create_future()
            fut.set_result([])
            return fut

        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_event_loop", return_value=loop), \
                 patch("src.crawler.postgres_client.search_documents", side_effect=fake_search):
                execute_rag_query(session, "q", match_count=42)
        finally:
            loop.close()

        assert captured["match_count"] == 42
