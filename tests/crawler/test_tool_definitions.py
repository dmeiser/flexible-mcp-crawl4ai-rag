"""Unit tests for src/crawler/tool_definitions.py — 100% coverage, offline."""
import contextlib
import json
import os
import pytest
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
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
    async def test_auto_mode_detects_recursive_from_follow_links(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/p1")]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            mock_rec = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com", follow_links=True)

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "auto"
        assert data["detected_strategy"] == "webpage_recursive"
        assert data["adaptive_applied"] is True
        mock_rec.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_single_mode_overrides_follow_links(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/page")]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            mock_single = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_markdown_file",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            mock_rec = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                      new_callable=AsyncMock)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(
                ctx,
                "https://x.com/page",
                follow_links=True,
                crawl_mode="single",
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "single"
        assert data["detected_strategy"] == "webpage_single"
        assert data["adaptive_applied"] is False
        mock_single.assert_called_once()
        mock_rec.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_mode_falls_back_to_auto(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/page")]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.is_txt", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.is_sitemap", return_value=False))
            stack.enter_context(patch("src.crawler.tool_definitions.crawl_markdown_file",
                                      new_callable=AsyncMock, return_value=crawl_results))
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com/page", crawl_mode="nonsense")

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "auto"
        assert data["detected_strategy"] == "webpage_single"
        assert data["adaptive_applied"] is True

    @pytest.mark.asyncio
    async def test_explicit_mode_txt(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/data.txt")]

        with contextlib.ExitStack() as stack:
            mock_cmf = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_markdown_file",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com/data.txt", crawl_mode="txt")

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "txt"
        assert data["detected_strategy"] == "text_file"
        mock_cmf.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_mode_sitemap(self):
        ctx = _make_ctx()
        sitemap_urls = ["https://x.com/p1", "https://x.com/p2"]
        crawl_results = [_make_crawl_result(u) for u in sitemap_urls]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("src.crawler.tool_definitions.parse_sitemap",
                                      new_callable=AsyncMock, return_value=sitemap_urls))
            mock_batch = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_batch",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results, pages=2, chunks=2)
            result = await td.smart_crawl_url(ctx, "https://x.com/sitemap.xml", crawl_mode="sitemap")

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "sitemap"
        assert data["detected_strategy"] == "sitemap"
        mock_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_mode_recursive(self):
        ctx = _make_ctx()
        crawl_results = [_make_crawl_result("https://x.com/p1")]

        with contextlib.ExitStack() as stack:
            mock_rec = stack.enter_context(
                patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                      new_callable=AsyncMock, return_value=crawl_results)
            )
            _scrawl_stack(stack, crawl_results)
            result = await td.smart_crawl_url(ctx, "https://x.com", crawl_mode="recursive")

        data = json.loads(result)
        assert data["success"] is True
        assert data["requested_mode"] == "recursive"
        assert data["detected_strategy"] == "webpage_recursive"
        mock_rec.assert_called_once()

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
    async def test_phase9_filters_forwarded(self):
        ctx = _make_ctx()
        captured: dict = {}

        async def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return []

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.settings", MagicMock(USE_RERANKING=False)), \
             patch("src.crawler.tool_definitions.search_documents", side_effect=fake_search):
            await td.perform_rag_query(
                ctx,
                "q",
                source="docs.x.com",
                content_class="structured",
                markdown_variant="fit_markdown",
                extraction_strategy="llm",
            )

        assert captured["filter"] == {
            "source": "docs.x.com",
            "content_class": "structured",
            "markdown_variant": "fit_markdown",
            "extraction_strategy": "llm",
        }

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



# ---------------------------------------------------------------------------
# Helpers for Phase 9.5/9.6 tool tests
# ---------------------------------------------------------------------------

def _make_mock_page(uid=1, value_score=0.1, staleness_score=0.8, hit_count=0,
                    is_pinned=False, tombstoned_at=None, is_active=True,
                    source="test.com"):
    rec = MagicMock()
    rec.id = uid
    rec.url = f"https://{source}/page-{uid}"
    rec.chunk_number = 0
    rec.content = "test content with enough characters"
    rec.value_score = value_score
    rec.staleness_score = staleness_score
    rec.hit_count = hit_count
    rec.is_pinned = is_pinned
    rec.tombstoned_at = tombstoned_at
    rec.is_active = is_active
    rec.page_metadata = {"source": source}
    rec.first_seen_at = datetime.now(timezone.utc)
    rec.crawl_timestamp = datetime.now(timezone.utc)
    # Remove auto-generated page_metadata so ex_metadata branch is reachable if needed
    return rec


def _make_mock_code(uid=10, value_score=0.2, staleness_score=0.5, hit_count=0,
                    is_pinned=False, tombstoned_at=None, is_active=True,
                    source="test.com"):
    rec = MagicMock()
    rec.id = uid
    rec.url = f"https://{source}/code-{uid}"
    rec.chunk_number = 0
    rec.content = "print('hello world')"
    rec.value_score = value_score
    rec.staleness_score = staleness_score
    rec.hit_count = hit_count
    rec.is_pinned = is_pinned
    rec.tombstoned_at = tombstoned_at
    rec.is_active = is_active
    rec.ex_metadata = {"source": source}
    rec.first_seen_at = datetime.now(timezone.utc)
    rec.crawl_timestamp = datetime.now(timezone.utc)
    return rec


# ---------------------------------------------------------------------------
# Tests: compute_value_scores
# ---------------------------------------------------------------------------

class TestComputeValueScores:
    @pytest.mark.asyncio
    async def test_updates_pages_and_examples(self):
        ctx = _make_ctx()
        mock_page = _make_mock_page()
        mock_code = _make_mock_code()

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = []          # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [mock_page] # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = [mock_code] # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.compute_value_scores(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["updated_crawled_pages"] == 1
        assert data["updated_code_examples"] == 1
        session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_source_filter_applied(self):
        ctx = _make_ctx()
        page_match = _make_mock_page(uid=1, source="target.com")
        page_other = _make_mock_page(uid=2, source="other.com")

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = []                           # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [page_match, page_other]    # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = []                           # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.compute_value_scores(ctx, source="target.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["updated_crawled_pages"] == 1  # only target.com matched

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("db crash")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.compute_value_scores(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "db crash" in data["error"]


# ---------------------------------------------------------------------------
# Tests: preview_eviction_plan
# ---------------------------------------------------------------------------

class TestPreviewEvictionPlan:
    @pytest.mark.asyncio
    async def test_dry_run_returns_candidates(self):
        ctx = _make_ctx()
        page = _make_mock_page(uid=1, value_score=0.1)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [page]  # CrawledPage
            elif call_n[0] == 2:
                m.all.return_value = []      # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.preview_eviction_plan(ctx, limit=10, dry_run=True)

        data = json.loads(result)
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["candidates_count"] == 1
        assert data["total_evicted"] == 0

    @pytest.mark.asyncio
    async def test_not_dry_run_tombstones_pages(self):
        """Covers the page_ids tombstone branch."""
        ctx = _make_ctx()
        page = _make_mock_page(uid=5, value_score=0.05)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [page]  # CrawledPage
            elif call_n[0] == 2:
                m.all.return_value = []      # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions.tombstone_records",
                   return_value=1) as mock_ts:
            result = await td.preview_eviction_plan(ctx, limit=10, dry_run=False)

        data = json.loads(result)
        assert data["success"] is True
        assert data["dry_run"] is False
        mock_ts.assert_called()

    @pytest.mark.asyncio
    async def test_not_dry_run_tombstones_code_examples(self):
        """Covers the code_ids tombstone branch."""
        ctx = _make_ctx()
        code = _make_mock_code(uid=20, value_score=0.03)
        # Make page_metadata not a dict so ex_metadata branch is used → real string source
        code.page_metadata = None

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = []      # CrawledPage — empty
            elif call_n[0] == 2:
                m.all.return_value = [code]  # CodeExample — one record
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions.tombstone_records",
                   return_value=1) as mock_ts:
            result = await td.preview_eviction_plan(ctx, limit=10, dry_run=False)

        data = json.loads(result)
        assert data["success"] is True
        mock_ts.assert_called_once()

    @pytest.mark.asyncio
    async def test_source_filter_excludes_other_sources(self):
        ctx = _make_ctx()
        p_match = _make_mock_page(uid=1, source="target.com", value_score=0.1)
        p_other = _make_mock_page(uid=2, source="other.com", value_score=0.2)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [p_match, p_other]  # CrawledPage
            elif call_n[0] == 2:
                m.all.return_value = []                  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.preview_eviction_plan(ctx, source="target.com")

        data = json.loads(result)
        assert data["candidates_count"] == 1
        assert data["candidates"][0]["source"] == "target.com"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("query fail")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.preview_eviction_plan(ctx)

        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Tests: enforce_storage_budget
# ---------------------------------------------------------------------------

class TestEnforceStorageBudget:
    @pytest.mark.asyncio
    async def test_below_threshold_returns_ok(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.return_value.first.return_value = None  # no policy → defaults

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=1_000_000):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "ok"
        assert data["actions_taken"] == []

    @pytest.mark.asyncio
    async def test_with_custom_policy_reads_attributes(self):
        """Covers the else branch when StoragePolicy is not None (lines 2322-2327)."""
        ctx = _make_ctx()
        policy = MagicMock()
        policy.max_db_size_gb = 5.0
        policy.warn_threshold = 0.80
        policy.high_threshold = 0.90
        policy.hard_threshold = 1.00
        policy.tombstone_grace_hours = 48
        policy.target_post_evict_ratio = 0.70

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy  # custom StoragePolicy
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=1_000_000):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "ok"

    @pytest.mark.asyncio
    async def test_force_runs_compact(self):
        ctx = _make_ctx()

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None  # StoragePolicy
            else:
                m.all.return_value = []      # tombstoned queries
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=1_000_000):
            result = await td.enforce_storage_budget(ctx, force=True)

        data = json.loads(result)
        assert data["success"] is True
        assert isinstance(data["actions_taken"], list)

    @pytest.mark.asyncio
    async def test_compact_deletes_expired_tombstones(self):
        ctx = _make_ctx()
        expired_rec = MagicMock()
        deleted = []
        session = MagicMock()
        session.delete.side_effect = lambda r: deleted.append(r)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None      # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = [expired_rec]  # CrawledPage expired
            elif call_n[0] == 3:
                m.all.return_value = []             # CodeExample expired
            else:
                m.all.return_value = []
            return m

        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=1_000_000):
            result = await td.enforce_storage_budget(ctx, force=True)

        data = json.loads(result)
        assert data["success"] is True
        assert len(deleted) == 1
        assert any("compacted 1" in a for a in data["actions_taken"])

    @pytest.mark.asyncio
    async def test_high_pressure_tombstones_stale(self):
        ctx = _make_ctx()
        stale_rec = _make_mock_page(uid=99, staleness_score=0.9, is_pinned=False)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None      # StoragePolicy
            elif call_n[0] in (2, 3):
                m.all.return_value = []          # compact: no expired
            elif call_n[0] == 4:
                m.all.return_value = [stale_rec] # high: stale CrawledPage
            elif call_n[0] == 5:
                m.all.return_value = []          # high: stale CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        # 95% → above high threshold (0.90)
        high_usage = int(0.95 * 10.0 * 1024 ** 3)

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=high_usage), \
             patch("src.crawler.tool_definitions.tombstone_records",
                   return_value=1) as mock_ts:
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "high"
        mock_ts.assert_called()

    @pytest.mark.asyncio
    async def test_hard_pressure_value_evicts(self):
        ctx = _make_ctx()
        low_val_rec = _make_mock_page(uid=77, value_score=0.01, is_pinned=False)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None      # StoragePolicy
            elif call_n[0] in (2, 3):
                m.all.return_value = []          # compact: no expired
            elif call_n[0] in (4, 5):
                m.all.return_value = []          # high: no stale
            elif call_n[0] == 6:
                m.all.return_value = [low_val_rec]  # hard: low-value CrawledPage
            elif call_n[0] == 7:
                m.all.return_value = []          # hard: low-value CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        # 105% → above hard threshold (1.00)
        over = int(1.05 * 10.0 * 1024 ** 3)

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=over), \
             patch("src.crawler.tool_definitions.tombstone_records",
                   return_value=1) as mock_ts:
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "critical"
        assert mock_ts.call_count >= 1

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("session failed")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=0):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "session failed" in data["error"]


# ---------------------------------------------------------------------------
# Tests: pin_records / unpin_records
# ---------------------------------------------------------------------------

class TestPinRecords:
    @pytest.mark.asyncio
    async def test_crawled_pages_pinned(self):
        ctx = _make_ctx()
        rec = _make_mock_page(uid=1, is_pinned=False)
        session = MagicMock()
        session.exec.return_value.all.return_value = [rec]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.pin_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is True
        assert data["pinned_count"] == 1
        assert rec.is_pinned is True

    @pytest.mark.asyncio
    async def test_code_examples_pinned(self):
        ctx = _make_ctx()
        rec = _make_mock_code(uid=2, is_pinned=False)
        session = MagicMock()
        session.exec.return_value.all.return_value = [rec]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.pin_records(ctx, record_ids=[2], table="code_examples")

        data = json.loads(result)
        assert data["success"] is True
        assert rec.is_pinned is True

    @pytest.mark.asyncio
    async def test_invalid_table_returns_error(self):
        ctx = _make_ctx()
        result = await td.pin_records(ctx, record_ids=[1], table="bad_table")
        data = json.loads(result)
        assert data["success"] is False
        assert "Unknown table" in data["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("pin fail")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.pin_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is False
        assert "pin fail" in data["error"]


class TestUnpinRecords:
    @pytest.mark.asyncio
    async def test_crawled_pages_unpinned(self):
        ctx = _make_ctx()
        rec = _make_mock_page(uid=1, is_pinned=True)
        session = MagicMock()
        session.exec.return_value.all.return_value = [rec]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.unpin_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is True
        assert data["unpinned_count"] == 1
        assert rec.is_pinned is False

    @pytest.mark.asyncio
    async def test_code_examples_unpinned(self):
        ctx = _make_ctx()
        rec = _make_mock_code(uid=3, is_pinned=True)
        session = MagicMock()
        session.exec.return_value.all.return_value = [rec]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.unpin_records(ctx, record_ids=[3], table="code_examples")

        data = json.loads(result)
        assert data["success"] is True
        assert rec.is_pinned is False

    @pytest.mark.asyncio
    async def test_invalid_table_returns_error(self):
        ctx = _make_ctx()
        result = await td.unpin_records(ctx, record_ids=[1], table="wrong")
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("unpin fail")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.unpin_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Tests: index_storage_report
# ---------------------------------------------------------------------------

class TestIndexStorageReport:
    @pytest.mark.asyncio
    async def test_returns_table_stats(self):
        ctx = _make_ctx()
        session = MagicMock()
        # None → _count() returns 0, StoragePolicy uses defaults
        session.exec.return_value.first.return_value = None
        session.exec.return_value.all.return_value = []

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=500_000_000):
            result = await td.index_storage_report(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert "tables" in data
        assert "crawled_pages" in data["tables"]
        assert "code_examples" in data["tables"]
        assert "pressure_level" in data
        assert "usage_ratio" in data

    @pytest.mark.asyncio
    async def test_group_by_source_adds_by_source(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.return_value.first.return_value = None
        session.exec.return_value.all.return_value = [
            ("example.com", 10, 8),
            ("other.com", 5, 5),
        ]

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=500_000_000):
            result = await td.index_storage_report(ctx, group_by="source")

        data = json.loads(result)
        assert data["success"] is True
        assert "by_source" in data
        assert len(data["by_source"]) == 2

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("storage query fail")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions._get_db_size_bytes",
                   return_value=0):
            result = await td.index_storage_report(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "storage query fail" in data["error"]


# ---------------------------------------------------------------------------
# Tests: restore_tombstoned_records
# ---------------------------------------------------------------------------

class TestRestoreTombstonedRecords:
    @pytest.mark.asyncio
    async def test_within_grace_window_restores(self):
        ctx = _make_ctx()
        tombstoned_at = datetime.now(timezone.utc) - timedelta(hours=1)
        rec = _make_mock_page(uid=1, tombstoned_at=tombstoned_at, is_active=False)
        policy = MagicMock()
        policy.tombstone_grace_hours = 24

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy  # StoragePolicy
            else:
                m.all.return_value = [rec]
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 1
        assert data["skipped_count"] == 0
        assert rec.tombstoned_at is None
        assert rec.is_active is True

    @pytest.mark.asyncio
    async def test_outside_grace_window_skips(self):
        ctx = _make_ctx()
        tombstoned_at = datetime.now(timezone.utc) - timedelta(hours=48)
        rec = _make_mock_page(uid=2, tombstoned_at=tombstoned_at, is_active=False)
        policy = MagicMock()
        policy.tombstone_grace_hours = 24

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy
            else:
                m.all.return_value = [rec]
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[2])

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 0
        assert data["skipped_count"] == 1

    @pytest.mark.asyncio
    async def test_naive_tombstoned_at_gets_utc(self):
        """Naive datetime is treated as UTC and compared correctly."""
        ctx = _make_ctx()
        naive_ts = datetime.now() - timedelta(hours=1)  # no timezone
        rec = _make_mock_page(uid=3, tombstoned_at=naive_ts, is_active=False)
        policy = MagicMock()
        policy.tombstone_grace_hours = 24

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy
            else:
                m.all.return_value = [rec]
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[3])

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 1

    @pytest.mark.asyncio
    async def test_no_policy_uses_default_grace(self):
        ctx = _make_ctx()
        tombstoned_at = datetime.now(timezone.utc) - timedelta(hours=1)
        rec = _make_mock_page(uid=7, tombstoned_at=tombstoned_at, is_active=False)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None  # no policy → default 24h
            else:
                m.all.return_value = [rec]
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[7])

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 1

    @pytest.mark.asyncio
    async def test_code_examples_table(self):
        ctx = _make_ctx()
        tombstoned_at = datetime.now(timezone.utc) - timedelta(hours=2)
        rec = _make_mock_code(uid=10, tombstoned_at=tombstoned_at, is_active=False)
        policy = MagicMock()
        policy.tombstone_grace_hours = 24

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy
            else:
                m.all.return_value = [rec]
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[10],
                                                         table="code_examples")

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 1

    @pytest.mark.asyncio
    async def test_invalid_table_returns_error(self):
        ctx = _make_ctx()
        result = await td.restore_tombstoned_records(ctx, record_ids=[1],
                                                     table="bad_table")
        data = json.loads(result)
        assert data["success"] is False
        assert "Unknown table" in data["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("restore fail")

        with patch("src.crawler.tool_definitions.get_session",
                   side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[1])

        data = json.loads(result)
        assert data["success"] is False
        assert "restore fail" in data["error"]


class TestRecrawlDueSources:
    @pytest.mark.asyncio
    async def test_no_policies_returns_empty(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions.smart_crawl_url", new_callable=AsyncMock) as mock_scrawl:
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 0
        assert data["recrawled_count"] == 0
        mock_scrawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_due_source_recrawls_successfully(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 1

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=5),
        )

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch(
                 "src.crawler.tool_definitions.smart_crawl_url",
                 new_callable=AsyncMock,
                 return_value=json.dumps({"success": True, "pages_crawled": 1}),
             ) as mock_scrawl:
            result = await td.recrawl_due_sources(ctx, max_concurrent=3)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["recrawled_count"] == 1
        mock_scrawl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_due_source_skipped(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 24

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=2),
        )

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch("src.crawler.tool_definitions.smart_crawl_url", new_callable=AsyncMock) as mock_scrawl:
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 0
        assert data["recrawled_count"] == 0
        mock_scrawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_source_filter_and_naive_timestamp(self):
        ctx = _make_ctx()
        p1 = MagicMock()
        p1.source = "target.com"
        p1.recrawl_interval_hours = 1
        p2 = MagicMock()
        p2.source = "other.com"
        p2.recrawl_interval_hours = 1

        session = MagicMock()
        session.exec.return_value.all.return_value = [p1, p2]
        # Naive datetime exercises tz normalization branch.
        session.execute.return_value.first.return_value = (
            datetime.now() - timedelta(hours=3),
        )

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch(
                 "src.crawler.tool_definitions.smart_crawl_url",
                 new_callable=AsyncMock,
                 return_value=json.dumps({"success": True, "pages_crawled": 1}),
             ) as mock_scrawl:
            result = await td.recrawl_due_sources(ctx, source="target.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["recrawled_count"] == 1
        mock_scrawl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_last_crawled_marks_due(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "nocrawl.example"
        policy.recrawl_interval_hours = 24

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        # No row returned => last_crawled is None => due
        session.execute.return_value.first.return_value = None

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch(
                 "src.crawler.tool_definitions.smart_crawl_url",
                 new_callable=AsyncMock,
                 return_value=json.dumps({"success": True, "pages_crawled": 1}),
             ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["recrawled_count"] == 1

    @pytest.mark.asyncio
    async def test_recrawl_failure_is_recorded(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "https://already-absolute.example"
        policy.recrawl_interval_hours = 1

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (
            datetime.now(timezone.utc) - timedelta(hours=8),
        )

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)), \
             patch(
                 "src.crawler.tool_definitions.smart_crawl_url",
                 new_callable=AsyncMock,
                 return_value=json.dumps({"success": False, "error": "crawl failed"}),
             ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["failed_count"] == 1
        assert "crawl failed" in data["failures"][0]["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("recrawl query failed")

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "recrawl query failed" in data["error"]


class TestPruneStaleContent:
    @pytest.mark.asyncio
    async def test_non_force_with_default_policy(self):
        ctx = _make_ctx()
        session = MagicMock()

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None  # no StoragePolicy => defaults
            else:
                m.all.return_value = []
            return m
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.prune_stale_content(ctx, force=False)

        data = json.loads(result)
        assert data["success"] is True
        assert data["grace_hours"] == 24
        assert data["hard_deleted_count"] == 0

    @pytest.mark.asyncio
    async def test_force_deletes_all_tombstones(self):
        ctx = _make_ctx()
        session = MagicMock()

        rec1 = MagicMock()
        rec2 = MagicMock()
        rec3 = MagicMock()
        deleted = []
        session.delete.side_effect = lambda r: deleted.append(r)

        policy = MagicMock()
        policy.tombstone_grace_hours = 48

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = policy
            elif call_n[0] == 2:
                m.all.return_value = [rec1, rec2]
            else:
                m.all.return_value = [rec3]
            return m
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.prune_stale_content(ctx, force=True)

        data = json.loads(result)
        assert data["success"] is True
        assert data["hard_deleted_count"] == 3
        assert len(deleted) == 3

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("prune failed")

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.prune_stale_content(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "prune failed" in data["error"]


class TestHardDeleteTombstones:
    @pytest.mark.asyncio
    async def test_delete_all_when_no_max_age(self):
        ctx = _make_ctx()
        session = MagicMock()

        rec1 = MagicMock()
        rec2 = MagicMock()
        deleted = []
        session.delete.side_effect = lambda r: deleted.append(r)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [rec1] if call_n[0] == 1 else [rec2]
            return m
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.hard_delete_tombstones(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["hard_deleted_count"] == 2
        assert len(deleted) == 2

    @pytest.mark.asyncio
    async def test_max_age_filter_path(self):
        ctx = _make_ctx()
        session = MagicMock()

        rec1 = MagicMock()
        deleted = []
        session.delete.side_effect = lambda r: deleted.append(r)

        call_n = [0]
        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [rec1] if call_n[0] == 1 else []
            return m
        session.exec.side_effect = exec_se

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.hard_delete_tombstones(ctx, max_age_hours=24)

        data = json.loads(result)
        assert data["success"] is True
        assert data["max_age_hours"] == 24
        assert data["hard_deleted_count"] == 1
        assert len(deleted) == 1

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("hard delete failed")

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.hard_delete_tombstones(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "hard delete failed" in data["error"]
