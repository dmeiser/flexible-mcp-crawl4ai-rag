"""Unit tests for src/crawler/tool_definitions.py — 100% coverage, offline."""
import contextlib
import json
import os
import pytest
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

import src.crawler.tool_definitions as td


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(crawler=None):
    ctx = MagicMock()
    ctx.lifespan_context = MagicMock()
    ctx.lifespan_context.crawler = crawler or AsyncMock()
    return ctx


def _make_crawl_result(url="https://x.com/page", markdown="# H1\n\nContent."):
    return {"url": url, "markdown": markdown}


def _make_get_session(session=None):
    """Return a side_effect callable that yields a context-manager session."""
    _session = session or MagicMock()

    def _gen():
        @contextmanager
        def _ctx():
            yield _session
        yield _ctx()

    return _gen


# ---------------------------------------------------------------------------
# Tests: _get_crawler
# ---------------------------------------------------------------------------

class TestGetCrawler:
    def test_none_lifespan_raises(self):
        ctx = MagicMock()
        ctx.lifespan_context = None
        with pytest.raises(RuntimeError, match="lifespan context missing"):
            td._get_crawler(ctx)

    def test_missing_crawler_attr_raises(self):
        ctx = MagicMock()
        ctx.lifespan_context = MagicMock(spec=[])  # no 'crawler' attr
        with pytest.raises(RuntimeError, match="lifespan context missing"):
            td._get_crawler(ctx)

    def test_returns_crawler(self):
        mock_crawler = AsyncMock()
        ctx = _make_ctx(crawler=mock_crawler)
        assert td._get_crawler(ctx) is mock_crawler


# ---------------------------------------------------------------------------
# Tests: crawl_single_page
# ---------------------------------------------------------------------------

class TestCrawlSinglePage:
    @pytest.mark.asyncio
    async def test_missing_lifespan_returns_error_json(self):
        ctx = MagicMock()
        ctx.lifespan_context = None
        result = await td.crawl_single_page(ctx, "https://x.com")
        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_crawl_failure_returns_error_json(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = MagicMock(success=False, markdown=None,
                                                    error_message="Timeout")
        ctx = _make_ctx(crawler=mock_crawler)
        with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                   new_callable=AsyncMock, return_value=[]):
            result = await td.crawl_single_page(ctx, "https://x.com")
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_success_no_agentic_rag(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = MagicMock(
            success=True, markdown="# H\n\nContent.",
            links={"internal": [], "external": []},
        )
        ctx = _make_ctx(crawler=mock_crawler)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.settings",
                                      MagicMock(USE_AGENTIC_RAG=False)))
            stack.enter_context(patch("src.crawler.tool_definitions.get_session",
                                      side_effect=_make_get_session()))
            stack.enter_context(patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                                      new_callable=AsyncMock, return_value=["# H\n\nContent."]))
            stack.enter_context(patch("src.crawler.tool_definitions.add_documents_to_db",
                                      new_callable=AsyncMock, return_value=1))
            result = await td.crawl_single_page(ctx, "https://x.com/page")

        data = json.loads(result)
        assert data["success"] is True
        assert data["chunks_stored"] == 1

    @pytest.mark.asyncio
    async def test_success_agentic_rag_with_code_blocks(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = MagicMock(
            success=True, markdown="```python\np()\n```",
            links={"internal": [], "external": []},
        )
        ctx = _make_ctx(crawler=mock_crawler)
        code_blocks = [{"language": "python", "content": "p()"}]

        # get_session is called twice (once for docs, once for code)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.settings",
                                      MagicMock(USE_AGENTIC_RAG=True)))
            stack.enter_context(patch("src.crawler.tool_definitions.get_session",
                                      side_effect=_make_get_session()))
            stack.enter_context(patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                                      new_callable=AsyncMock, return_value=["chunk"]))
            stack.enter_context(patch("src.crawler.tool_definitions.add_documents_to_db",
                                      new_callable=AsyncMock, return_value=1))
            stack.enter_context(patch("src.crawler.tool_definitions.extract_code_blocks",
                                      return_value=code_blocks))
            mock_add_code_patch = stack.enter_context(
                patch("src.crawler.tool_definitions.add_code_examples_to_db",
                      new_callable=AsyncMock, return_value=1)
            )
            result = await td.crawl_single_page(ctx, "https://x.com/page")

        data = json.loads(result)
        assert data["success"] is True
        mock_add_code_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_agentic_rag_no_code_blocks_skips_add_code(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = MagicMock(
            success=True, markdown="No code here.", links={},
        )
        ctx = _make_ctx(crawler=mock_crawler)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.settings",
                                      MagicMock(USE_AGENTIC_RAG=True)))
            stack.enter_context(patch("src.crawler.tool_definitions.get_session",
                                      side_effect=_make_get_session()))
            stack.enter_context(patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                                      new_callable=AsyncMock, return_value=["chunk"]))
            stack.enter_context(patch("src.crawler.tool_definitions.add_documents_to_db",
                                      new_callable=AsyncMock, return_value=1))
            stack.enter_context(patch("src.crawler.tool_definitions.extract_code_blocks",
                                      return_value=[]))
            mock_add_code = stack.enter_context(
                patch("src.crawler.tool_definitions.add_code_examples_to_db",
                      new_callable=AsyncMock)
            )
            result = await td.crawl_single_page(ctx, "https://x.com/page")

        data = json.loads(result)
        assert data["success"] is True
        mock_add_code.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: smart_crawl_url
# ---------------------------------------------------------------------------

def _scrawl_stack(stack, crawl_results, pages=1, chunks=1, extra_patches=None):
    """Push the common smart_crawl_url patches onto the ExitStack."""
    stack.enter_context(patch("src.crawler.tool_definitions.settings",
                              MagicMock(USE_AGENTIC_RAG=False)))
    stack.enter_context(patch("src.crawler.tool_definitions.get_session",
                              side_effect=_make_get_session()))
    stack.enter_context(patch("src.crawler.tool_definitions.store_crawled_documents",
                              new_callable=AsyncMock, return_value=(pages, chunks)))
    if extra_patches:
        for p in extra_patches:
            stack.enter_context(p)


class TestSmartCrawlUrl:
    @pytest.mark.asyncio
    async def test_txt_url_uses_crawl_markdown_file(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/data.txt")]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=True))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            mock_cmf = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_markdown_file",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com/data.txt")

        data = json.loads(result)
        assert data["success"] is True
        assert data["crawl_type"] == "text_file"
        mock_cmf.assert_called_once()

    @pytest.mark.asyncio
    async def test_sitemap_with_urls(self):
        ctx = _make_ctx()
        sitemap_urls = ["https://x.com/p1", "https://x.com/p2"]
        crawl_results = [_make_crawl_result(u) for u in sitemap_urls]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=True))
            stack.enter_context(patch("src.crawler.tool_definitions.parse_sitemap",
                                      new_callable=AsyncMock, return_value=sitemap_urls))
            mock_batch = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_batch",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results, pages=2, chunks=2)
            result = await td.smart_crawl_url(ctx, "https://x.com/sitemap.xml")

        data = json.loads(result)
        assert data["success"] is True
        assert data["crawl_type"] == "sitemap"
        mock_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_sitemap_no_urls_returns_error(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.is_txt", return_value=False), \
             patch("src.crawler.tool_definitions.is_sitemap", return_value=True), \
             patch("src.crawler.tool_definitions.parse_sitemap",
                   new_callable=AsyncMock, return_value=[]):
            result = await td.smart_crawl_url(ctx, "https://x.com/sitemap.xml")

        data = json.loads(result)
        assert data["success"] is False
        assert "sitemap" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_follow_links_uses_recursive_crawl(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result()]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            mock_rec = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com",
                                               follow_links=True, url_pattern=r"/docs/.*")

        data = json.loads(result)
        assert data["success"] is True
        assert data["crawl_type"] == "webpage_recursive"
        mock_rec.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_single_page_crawl(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result()]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            mock_cmf = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_markdown_file",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com/page")

        data = json.loads(result)
        assert data["success"] is True
        assert data["crawl_type"] == "webpage_single"

    @pytest.mark.asyncio
    async def test_empty_crawl_results_returns_error(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.is_txt", return_value=False), \
             patch("src.crawler.tool_definitions.is_sitemap", return_value=False), \
             patch("src.crawler.tool_definitions.crawl_markdown_file",
                   new_callable=AsyncMock, return_value=[]):
            result = await td.smart_crawl_url(ctx, "https://x.com/page")

        data = json.loads(result)
        assert data["success"] is False
        assert "No content" in data["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.is_txt",
                   side_effect=RuntimeError("boom")):
            result = await td.smart_crawl_url(ctx, "https://x.com")

        data = json.loads(result)
        assert data["success"] is False
        assert "boom" in data["error"]

    @pytest.mark.asyncio
    async def test_result_includes_urls_sample_truncated(self):
        ctx = _make_ctx()
        # 7 results → sample shows first 5 + "..."
        crawl_results = [_make_crawl_result(f"https://x.com/{i}") for i in range(7)]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.crawl_markdown_file",
                                      new_callable=AsyncMock, return_value=crawl_results))
            _scrawl_stack(stack, crawl_results, pages=7, chunks=7)
            result = await td.smart_crawl_url(ctx, "https://x.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["urls_crawled_sample"][-1] == "..."
        assert len(data["urls_crawled_sample"]) == 6  # 5 + "..."

    @pytest.mark.asyncio
    async def test_crawler_not_initialized_returns_error_json(self):
        ctx = MagicMock()
        ctx.lifespan_context = None

        result = await td.smart_crawl_url(ctx, "https://x.com")
        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Tests: get_available_sources
# ---------------------------------------------------------------------------

class TestGetAvailableSources:
    @pytest.mark.asyncio
    async def test_success(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.fetch_available_sources",
                   return_value=["a.com", "b.com"]):
            result = await td.get_available_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["sources"] == ["a.com", "b.com"]
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.fetch_available_sources",
                   side_effect=RuntimeError("db down")):
            result = await td.get_available_sources(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "db down" in data["error"]


# ---------------------------------------------------------------------------
# Tests: perform_rag_query
# ---------------------------------------------------------------------------

class TestPerformRagQuery:
    @pytest.mark.asyncio
    async def test_basic_query_no_reranking(self):
        ctx = _make_ctx()
        raw_results = [
            {"url": "u", "content": "c", "page_metadata": {"x": 1}, "similarity_score": 0.8}
        ]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=False)), \
             patch("src.crawler.tool_definitions.search_documents",
                   new_callable=AsyncMock, return_value=raw_results):
            result = await td.perform_rag_query(ctx, "test query")

        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["url"] == "u"

    @pytest.mark.asyncio
    async def test_source_filter_forwarded(self):
        ctx = _make_ctx()
        captured: dict = {}

        async def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return []

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=False)), \
             patch("src.crawler.tool_definitions.search_documents", side_effect=fake_search):
            await td.perform_rag_query(ctx, "q", source="docs.x.com")

        assert captured["filter"] == {"source": "docs.x.com"}

    @pytest.mark.asyncio
    async def test_no_source_passes_none_filter(self):
        ctx = _make_ctx()
        captured: dict = {}

        async def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return []

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=False)), \
             patch("src.crawler.tool_definitions.search_documents", side_effect=fake_search):
            await td.perform_rag_query(ctx, "q")

        assert captured["filter"] is None

    @pytest.mark.asyncio
    async def test_reranking_applied(self):
        ctx = _make_ctx()
        raw_results = [
            {"url": "u1", "content": "c", "page_metadata": {}, "similarity_score": 0.8},
            {"url": "u2", "content": "d", "page_metadata": {}, "similarity_score": 0.6},
        ]
        reranked = [raw_results[1], raw_results[0]]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=True)), \
             patch("src.crawler.tool_definitions.search_documents",
                   new_callable=AsyncMock, return_value=raw_results), \
             patch("src.crawler.tool_definitions.rerank_results",
                   return_value=reranked) as mock_rerank:
            result = await td.perform_rag_query(ctx, "q", match_count=2)

        data = json.loads(result)
        assert data["success"] is True
        mock_rerank.assert_called_once_with("q", raw_results, top_k=2)
        assert data["results"][0]["url"] == "u2"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=False)), \
             patch("src.crawler.tool_definitions.search_documents",
                   side_effect=RuntimeError("search crash")):
            result = await td.perform_rag_query(ctx, "q")

        data = json.loads(result)
        assert data["success"] is False
        assert "search crash" in data["error"]


# ---------------------------------------------------------------------------
# Tests: search_code_examples (tool)
# ---------------------------------------------------------------------------

class TestSearchCodeExamplesTool:
    @pytest.mark.asyncio
    async def test_success(self):
        ctx = _make_ctx()
        code_results = [
            {"url": "u", "language": "python", "content": "print()",
             "summary": "prints", "similarity_score": 0.95}
        ]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_code_examples",
                   new_callable=AsyncMock, return_value=code_results):
            result = await td.search_code_examples(ctx, "print hello world")

        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["language"] == "python"

    @pytest.mark.asyncio
    async def test_language_filter_forwarded(self):
        ctx = _make_ctx()
        captured: dict = {}

        async def fake_search(sess, q, match_count, language):
            captured["language"] = language
            return []

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_code_examples",
                   side_effect=fake_search):
            await td.search_code_examples(ctx, "q", language="rust")

        assert captured["language"] == "rust"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_code_examples",
                   side_effect=RuntimeError("vector fail")):
            result = await td.search_code_examples(ctx, "q")

        data = json.loads(result)
        assert data["success"] is False
        assert "vector fail" in data["error"]
