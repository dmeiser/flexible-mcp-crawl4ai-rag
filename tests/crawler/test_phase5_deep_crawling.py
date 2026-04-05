"""Unit tests for Phase 5 deep crawling features in src/crawler/tool_definitions.py"""
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
# Tests: Phase 5 Deep Crawling Implementation
# ---------------------------------------------------------------------------

class TestValidateLinkFilter:
    def test_none_returns_none(self):
        """_validate_link_filter returns None for None input."""
        assert td._validate_link_filter(None) is None
    
    def test_empty_string_returns_none(self):
        """_validate_link_filter returns None for empty string."""
        assert td._validate_link_filter("") is None
        assert td._validate_link_filter("   ") is None
    
    def test_invalid_type_returns_none(self):
        """_validate_link_filter returns None for non-string inputs."""
        assert td._validate_link_filter(123) is None
        assert td._validate_link_filter([]) is None
    
    def test_valid_regex_returned(self):
        """_validate_link_filter returns valid regex patterns."""
        pattern = r".*\.example\.com.*"
        assert td._validate_link_filter(pattern) == pattern
    
    def test_invalid_regex_returns_none(self):
        """_validate_link_filter returns None for invalid regex."""
        assert td._validate_link_filter(r"[invalid[") is None


class TestCrawlToMarkdownWithDeepCrawling:
    @pytest.mark.asyncio
    async def test_default_shallow_crawl(self):
        """Default parameters use shallow crawl (max_depth=1, follow_links=False)."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(
            success=True,
            markdown="# Content",
            url="https://example.com"
        )
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                   new_callable=AsyncMock, return_value=[]):
            result = await td.crawl_to_markdown(ctx, "https://example.com")
        
        data = json.loads(result)
        assert data["success"] is True
        assert data["pages_crawled"] == 1
        assert data["max_depth_configured"] == 1
        assert data["follow_links_enabled"] is False
        assert data["deep_crawl_mode"] == "single_page"
        assert "artifacts" in data
        # Should use simple arun, not recursive crawl
        mock_crawler.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_max_depth_clamped(self):
        """max_depth is clamped to valid range 1-10."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(success=True, markdown="# C")
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                   new_callable=AsyncMock, return_value=[]):
            # Test lower bound clamping
            result = await td.crawl_to_markdown(ctx, "https://x.com", max_depth=0)
            data = json.loads(result)
            assert data["max_depth_configured"] == 1
            
            # Test upper bound clamping
            result = await td.crawl_to_markdown(ctx, "https://x.com", max_depth=50)
            data = json.loads(result)
            assert data["max_depth_configured"] == 10
    
    @pytest.mark.asyncio
    async def test_deep_crawl_when_enabled(self):
        """follow_links=True with max_depth > 1 uses recursive crawl."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(
            success=True,
            markdown="# Page 1",
            url="https://example.com/p1",
            depth=0
        )
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                   new_callable=AsyncMock, return_value=[mock_result]) as mock_recursive:
            with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                       new_callable=AsyncMock, return_value=[]):
                result = await td.crawl_to_markdown(
                    ctx, "https://example.com", max_depth=3, follow_links=True
                )
        
        data = json.loads(result)
        assert data["success"] is True
        assert data["follow_links_enabled"] is True
        assert data["deep_crawl_mode"] == "compatibility_recursive"
        # Should call recursive crawl function
        mock_recursive.assert_called_once()
        call_kwargs = mock_recursive.call_args[1]
        assert call_kwargs["max_depth"] == 3
        assert call_kwargs["start_urls"] == ["https://example.com"]
    
    @pytest.mark.asyncio
    async def test_multiple_pages_from_deep_crawl(self):
        """Deep crawl aggregates results from multiple pages."""
        mock_crawler = AsyncMock()
        mock_results = [
            MagicMock(success=True, markdown="# P1", url="https://example.com/p1", depth=0),
            MagicMock(success=True, markdown="# P2", url="https://example.com/p2", depth=1),
            MagicMock(success=True, markdown="# P3", url="https://example.com/p3", depth=1),
        ]
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                   new_callable=AsyncMock, return_value=mock_results):
            with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                       new_callable=AsyncMock, return_value=[]):
                result = await td.crawl_to_markdown(
                    ctx, "https://example.com", max_depth=2, follow_links=True
                )
        
        data = json.loads(result)
        assert data["pages_crawled"] == 3
    
    @pytest.mark.asyncio
    async def test_failed_crawl_returns_error(self):
        """Failed crawl with no results returns error."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(success=False, markdown=None, error_message="Timeout")
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)
        
        result = await td.crawl_to_markdown(ctx, "https://example.com")
        data = json.loads(result)
        assert data["success"] is False
        assert "Timeout" in data["error"]
    
    @pytest.mark.asyncio
    async def test_deep_crawl_with_metadata(self):
        """Deep crawl stores depth metadata when indexing."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(
            success=True,
            markdown="# Content",
            url="https://example.com/p1",
            depth=1
        )
        ctx = _make_ctx(crawler=mock_crawler)
        
        captured_metas = []
        async def capture_add_documents(session, urls, contents, metas, chunks, fulldocs):
            captured_metas.extend(metas)
            return len(urls)
        
        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                   new_callable=AsyncMock, return_value=[mock_result]):
            with patch("src.crawler.tool_definitions.chunk_text_according_to_settings",
                       new_callable=AsyncMock, return_value=["Chunk"]):
                with patch("src.crawler.tool_definitions.add_documents_to_db",
                           new_callable=AsyncMock, side_effect=capture_add_documents):
                    with patch("src.crawler.tool_definitions.get_session",
                               side_effect=_make_get_session()):
                        result = await td.crawl_to_markdown(
                            ctx, "https://example.com",
                            index_result=True, max_depth=2, follow_links=True
                        )
        
        data = json.loads(result)
        assert data["success"] is True
        assert len(captured_metas) > 0
        meta = captured_metas[0]
        assert meta["depth"] == 1
        assert meta["max_depth"] == 2
        assert meta["follow_links"] is True


class TestCrawlManyUrlsWithDeepCrawling:
    @pytest.mark.asyncio
    async def test_default_batch_crawl(self):
        """Default parameters use batch arun_many."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(
            success=True,
            markdown="# Content",
            url="https://example.com"
        )
        mock_crawler.arun_many.return_value = [mock_result]
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.store_crawled_documents",
                   new_callable=AsyncMock, return_value=(1, 0)):
            with patch("src.crawler.tool_definitions.get_session",
                       side_effect=_make_get_session()):
                result = await td.crawl_many_urls(
                    ctx, ["https://example.com"], index_result=False
                )
        
        data = json.loads(result)
        assert data["success"] is True
        assert data["pages_crawled"] == 1
        assert data["follow_links_enabled"] is False
        # Should use batch arun_many
        mock_crawler.arun_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_with_deep_crawl(self):
        """Batch crawl uses recursive crawl per starting URL when follow_links=True."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock(
            success=True,
            markdown="# Content",
            url="https://example.com",
            depth=0
        )
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                   new_callable=AsyncMock, return_value=[mock_result]) as mock_recursive:
            with patch("src.crawler.tool_definitions.store_crawled_documents",
                       new_callable=AsyncMock, return_value=(1, 0)):
                with patch("src.crawler.tool_definitions.get_session",
                           side_effect=_make_get_session()):
                    result = await td.crawl_many_urls(
                        ctx, ["https://example.com", "https://other.com"],
                        max_depth=2, follow_links=True, index_result=False
                    )
        
        data = json.loads(result)
        assert data["success"] is True
        # One call per starting URL
        assert mock_recursive.call_count == 2
    
    @pytest.mark.asyncio
    async def test_no_urls_returns_error(self):
        """Empty URL list returns error."""
        mock_crawler = AsyncMock()
        ctx = _make_ctx(crawler=mock_crawler)
        
        result = await td.crawl_many_urls(ctx, [])
        data = json.loads(result)
        assert data["success"] is False
        assert "No URLs provided" in data["error"]
    
    @pytest.mark.asyncio
    async def test_batch_crawl_exception(self):
        """Batch crawl exception is logged and returned."""
        mock_crawler = AsyncMock()
        ctx = _make_ctx(crawler=mock_crawler)
        
        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links",
                   new_callable=AsyncMock, side_effect=RuntimeError("Network error")):
            result = await td.crawl_many_urls(
                ctx, ["https://example.com"],
                max_depth=2, follow_links=True
            )
        
        data = json.loads(result)
        assert data["success"] is False
        assert "Network error" in data["error"]


class TestWaveAndPhase4HelperCoverage:
    def test_build_run_config_allowlist_and_cache_mode(self):
        cfg = td._build_run_config({
            "cache_mode": "bypass",
            "wait_for": "body",
            "unsafe": "no",
        })
        assert getattr(cfg, "wait_for") == "body"
        assert not hasattr(cfg, "unsafe")

    def test_build_run_config_invalid_cache_mode_falls_back(self):
        cfg = td._build_run_config({"cache_mode": "not-real"})
        assert cfg.cache_mode is not None

    def test_extract_markdown_variants_none(self):
        out = td._extract_markdown_variants(None)
        assert out["raw_markdown"] == ""
        assert out["fit_html"] == ""

    def test_build_extraction_strategy_css_xpath_regex_and_invalid(self):
        css = td._build_extraction_strategy("css", schema={"title": "h1"})
        xpath = td._build_extraction_strategy("xpath", schema={"title": "//h1"})
        regex = td._build_extraction_strategy("regex", patterns={"email": r".+@.+"})
        bad = td._build_extraction_strategy("unknown")
        assert css is not None
        assert xpath is not None
        assert regex is not None
        assert bad is None

    def test_build_extraction_strategy_missing_schema_returns_none(self):
        assert td._build_extraction_strategy("css", schema=None) is None
        assert td._build_extraction_strategy("xpath", schema=None) is None

    def test_build_extraction_strategy_regex_default_and_llm(self):
        regex_default = td._build_extraction_strategy("regex")
        assert regex_default is not None

        with patch("src.crawler.tool_definitions.LLMConfig") as mock_cfg, \
             patch("src.crawler.tool_definitions.LLMExtractionStrategy") as mock_llm:
            mock_cfg.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            llm = td._build_extraction_strategy("llm", instruction="extract")
            assert llm is not None
            mock_cfg.assert_called_once()
            mock_llm.assert_called_once()

    def test_session_helpers(self):
        assert td._normalize_session_id(None) is None
        assert td._normalize_session_id("") is None
        assert td._normalize_session_id("   ") is None
        assert td._normalize_session_id("  abc  ") == "abc"
        assert td._normalize_session_id(123) is None

        merged = td._merge_run_config_with_session({"cache_mode": "bypass"}, " sess-1 ")
        assert merged["session_id"] == "sess-1"

        unchanged = td._merge_run_config_with_session({"cache_mode": "bypass"}, "   ")
        assert unchanged == {"cache_mode": "bypass"}

    def test_build_browser_config_allowlist(self):
        cfg = td._build_browser_config({
            "browser_type": "chromium",
            "headless": False,
            "viewport_width": 1200,
            "enable_stealth": True,
            "unsafe": "drop-me",
        })
        assert getattr(cfg, "headless") is False
        assert getattr(cfg, "browser_type") == "chromium"
        assert getattr(cfg, "enable_stealth") is True
        assert not hasattr(cfg, "unsafe")

        cfg2 = td._build_browser_config(None)
        assert getattr(cfg2, "headless") is True

        cfg3 = td._build_browser_config({"browser_type": "chromium"})
        assert getattr(cfg3, "headless") is True

    def test_build_markdown_generator_defaults_none(self):
        assert td._build_markdown_generator() is None

    def test_build_markdown_generator_pruning_and_source(self):
        gen = td._build_markdown_generator(
            markdown_options={"ignore_links": True},
            content_source="fit_html",
            content_filter="pruning",
            content_filter_query="python",
            content_filter_threshold=0.6,
        )
        assert gen is not None
        assert getattr(gen, "content_source") == "fit_html"

    def test_build_markdown_generator_bm25_and_invalid_source(self):
        gen = td._build_markdown_generator(
            content_source="not-real",
            content_filter="bm25",
            content_filter_query="docs",
            content_filter_threshold=1.3,
        )
        assert gen is not None
        assert getattr(gen, "content_source") == "cleaned_html"

    def test_build_markdown_generator_llm_filter(self):
        with patch("src.crawler.tool_definitions.LLMConfig") as mock_cfg, \
             patch("src.crawler.tool_definitions.LLMContentFilter") as mock_filter, \
             patch("src.crawler.tool_definitions.DefaultMarkdownGenerator") as mock_gen:
            mock_cfg.return_value = MagicMock()
            mock_filter.return_value = MagicMock()
            mock_gen.return_value = MagicMock(content_source="cleaned_html")

            gen = td._build_markdown_generator(
                content_filter="llm",
                content_filter_instruction="Keep only key facts",
                llm_provider="openai/gpt-4o-mini",
            )

            assert gen is not None
            mock_cfg.assert_called_once_with(provider="openai/gpt-4o-mini")
            mock_filter.assert_called_once()
            mock_gen.assert_called_once()

    def test_build_markdown_generator_invalid_filter_ignored(self):
        gen = td._build_markdown_generator(
            content_filter="not-a-filter",
            markdown_options={"ignore_links": True},
        )
        assert gen is not None

    def test_infer_source_type(self):
        assert td._infer_source_type("https://example.com") == "remote_url"
        assert td._infer_source_type("file:///tmp/a.md") == "local_file"
        assert td._infer_source_type("raw:<h1>x</h1>") == "raw_html"
        assert td._infer_source_type("https://example.com", session_id="s-1") == "session_derived"

    def test_json_safe_artifact(self):
        assert td._json_safe_artifact(None) is None
        assert td._json_safe_artifact("x") == "x"
        assert td._json_safe_artifact({"k": "v"}) == {"k": "v"}
        assert td._json_safe_artifact([1, 2]) == [1, 2]
        assert td._json_safe_artifact(MagicMock()) is None


class TestAdditionalToolPathCoverage:
    @pytest.mark.asyncio
    async def test_crawl_to_markdown_extraction_result_and_url_fallback(self):
        mock_crawler = AsyncMock()
        # no r.url on purpose -> exercise non-string fallback branch
        mock_result = MagicMock(success=True, markdown="# A")
        mock_result.extracted_content = {"title": "A"}
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)

        result = await td.crawl_to_markdown(
            ctx,
            "https://fallback-url.test",
            extraction_strategy="regex",
            extraction_patterns={"title": r"# (.+)"},
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["extraction_result"] == {"title": "A"}

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_session_id_applied(self):
        mock_crawler = AsyncMock()
        mock_result = MagicMock(success=True, markdown="# ok", url="https://x.com")
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)

        out = await td.crawl_to_markdown(ctx, "https://x.com", session_id="  auth-session-1  ")
        data = json.loads(out)

        assert data["success"] is True
        assert data["session_id_applied"] == "auth-session-1"
        config = mock_crawler.arun.call_args.kwargs["config"]
        assert getattr(config, "session_id", None) == "auth-session-1"

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_markdown_controls_applied(self):
        mock_crawler = AsyncMock()
        mock_result = MagicMock(success=True, markdown="# ok", url="https://x.com")
        mock_crawler.arun.return_value = mock_result
        ctx = _make_ctx(crawler=mock_crawler)

        out = await td.crawl_to_markdown(
            ctx,
            "https://x.com",
            markdown_options={"ignore_links": True},
            content_source="fit_html",
            content_filter="pruning",
            content_filter_query="python",
            content_filter_threshold=0.55,
        )
        data = json.loads(out)

        assert data["success"] is True
        assert data["markdown_options_applied"] is True
        assert data["content_source_applied"] == "fit_html"
        assert data["content_filter_applied"] == "pruning"
        config = mock_crawler.arun.call_args.kwargs["config"]
        assert getattr(config, "markdown_generator", None) is not None

    @pytest.mark.asyncio
    async def test_crawl_many_urls_markdown_controls_applied(self):
        ok = MagicMock(success=True, markdown="# md", url="https://x.com")
        mock_crawler = AsyncMock()
        mock_crawler.arun_many.return_value = [ok]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_many_urls(
            ctx,
            ["https://x.com"],
            index_result=False,
            markdown_options={"ignore_images": True},
            content_source="fit_html",
            content_filter="bm25",
            content_filter_query="tutorial",
            content_filter_threshold=1.2,
        ))

        assert data["success"] is True
        assert data["markdown_options_applied"] is True
        assert data["content_source_applied"] == "fit_html"
        assert data["content_filter_applied"] == "bm25"
        config = mock_crawler.arun_many.call_args.kwargs["config"]
        assert getattr(config, "markdown_generator", None) is not None

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_skips_failed_item_and_sets_extraction_meta(self):
        failed = MagicMock(success=False, markdown=None)
        ok = MagicMock(success=True, markdown="# ok", url="https://ok.test", extracted_content={"k": "v"})
        ctx = _make_ctx(crawler=AsyncMock())

        captured_metas = []

        async def cap_add(session, urls, contents, metas, chunks, fulldocs):
            captured_metas.extend(metas)
            return len(urls)

        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links", new_callable=AsyncMock, return_value=[failed, ok]), \
             patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.chunk_text_according_to_settings", new_callable=AsyncMock, return_value=["c1"]), \
             patch("src.crawler.tool_definitions.add_documents_to_db", new_callable=AsyncMock, side_effect=cap_add):
            out = await td.crawl_to_markdown(
                ctx,
                "https://seed.test",
                follow_links=True,
                max_depth=2,
                extraction_strategy="regex",
                extraction_patterns={"k": "v"},
                index_result=True,
            )

        data = json.loads(out)
        assert data["success"] is True
        assert data["extraction_result"] == {"k": "v"}
        assert captured_metas and captured_metas[0].get("extraction_strategy") == "regex"

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_session_source_type_metadata(self):
        ok = MagicMock(success=True, markdown="# ok", url="https://ok.test", extracted_content={"k": "v"})
        ctx = _make_ctx(crawler=AsyncMock())

        captured_metas = []
        async def cap_add(session, urls, contents, metas, chunks, fulldocs):
            captured_metas.extend(metas)
            return len(urls)

        with patch("src.crawler.tool_definitions.crawl_recursive_internal_links", new_callable=AsyncMock, return_value=[ok]), \
             patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.chunk_text_according_to_settings", new_callable=AsyncMock, return_value=["c1"]), \
             patch("src.crawler.tool_definitions.add_documents_to_db", new_callable=AsyncMock, side_effect=cap_add):
            out = await td.crawl_to_markdown(
                ctx,
                "https://seed.test",
                follow_links=True,
                max_depth=2,
                session_id="sess-1",
                index_result=True,
            )

        data = json.loads(out)
        assert data["success"] is True
        assert captured_metas and captured_metas[0].get("source_type") == "session_derived"

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_handles_result_without_success_markdown(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = MagicMock(success=False, markdown=None, error_message="failed")
        ctx = _make_ctx(crawler=mock_crawler)
        data = json.loads(await td.crawl_to_markdown(ctx, "https://x.com"))
        assert data["success"] is False
        assert "failed" in data["error"]

    @pytest.mark.asyncio
    async def test_crawl_to_markdown_exception_path(self):
        ctx = _make_ctx(crawler=AsyncMock())
        with patch("src.crawler.tool_definitions._build_run_config", side_effect=RuntimeError("cfg-break")):
            data = json.loads(await td.crawl_to_markdown(ctx, "https://x.com"))
        assert data["success"] is False
        assert "cfg-break" in data["error"]

    @pytest.mark.asyncio
    async def test_crawl_many_urls_error_and_empty_variant_paths(self):
        class EmptyMd:
            raw_markdown = ""
            fit_markdown = ""
            markdown_with_citations = ""
            references_markdown = ""

            def __str__(self):
                return ""

        ok_empty = MagicMock(success=True, markdown=EmptyMd(), url="https://x.com")
        bad = MagicMock(success=False, markdown=None, url="https://y.com", error_message="no content")
        mock_crawler = AsyncMock()
        mock_crawler.arun_many.return_value = [ok_empty, bad]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_many_urls(ctx, ["https://x.com", "https://y.com"], index_result=False))
        assert data["success"] is True
        assert len(data["errors"]) >= 2
        assert any("Empty selected markdown variant" in e["error"] for e in data["errors"])

    @pytest.mark.asyncio
    async def test_crawl_many_urls_with_extraction_and_index(self):
        ok = MagicMock(success=True, markdown="# md", url="https://x.com")
        ok.extracted_content = {"k": "v"}
        mock_crawler = AsyncMock()
        mock_crawler.arun_many.return_value = [ok]
        ctx = _make_ctx(crawler=mock_crawler)

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.store_crawled_documents", new_callable=AsyncMock, return_value=(1, 2)):
            data = json.loads(await td.crawl_many_urls(
                ctx,
                ["https://x.com"],
                extraction_strategy="regex",
                extraction_patterns={"k": "v"},
                index_result=True,
            ))
        assert data["success"] is True
        assert data["chunks_stored"] == 2

    @pytest.mark.asyncio
    async def test_crawl_many_urls_session_id_applied(self):
        ok = MagicMock(success=True, markdown="# md", url="https://x.com")
        mock_crawler = AsyncMock()
        mock_crawler.arun_many.return_value = [ok]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_many_urls(
            ctx,
            ["https://x.com"],
            index_result=False,
            session_id="session-shared",
        ))

        assert data["success"] is True
        assert data["session_id_applied"] == "session-shared"
        config = mock_crawler.arun_many.call_args.kwargs["config"]
        assert getattr(config, "session_id", None) == "session-shared"

    @pytest.mark.asyncio
    async def test_crawl_local_file_and_raw_html_paths(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True})) as ctm:
            out = await td.crawl_local_file(ctx, "README.md")
            assert json.loads(out)["success"] is True
            assert ctm.await_count == 1

        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True})):
            out2 = await td.crawl_raw_html(ctx, "<h1>x</h1>")
            assert json.loads(out2)["success"] is True

        out3 = await td.crawl_raw_html(ctx, "")
        assert json.loads(out3)["success"] is False

        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            fail_local = await td.crawl_local_file(ctx, "file:///tmp/a.md")
            assert json.loads(fail_local)["success"] is False

        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, side_effect=RuntimeError("boom2")):
            fail_raw = await td.crawl_raw_html(ctx, "<p>x</p>")
            assert json.loads(fail_raw)["success"] is False

    @pytest.mark.asyncio
    async def test_search_documents_alias_and_document_fetch_tools(self):
        ctx = _make_ctx()

        with patch("src.crawler.tool_definitions.perform_rag_query", new_callable=AsyncMock, return_value=json.dumps({"success": True, "results": []})):
            alias_out = await td.search_documents_tool(ctx, "q")
            assert json.loads(alias_out)["success"] is True

        row = MagicMock(id=7, url="https://x.com", chunk_number=0, content="abc", page_metadata={"a": 1})
        session = MagicMock()
        session.exec.return_value.first.return_value = row
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            got = json.loads(await td.get_document_by_id(ctx, 7))
            assert got["success"] is True
            assert got["document"]["id"] == 7

        session2 = MagicMock()
        session2.exec.return_value.first.return_value = None
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session2)):
            missing = json.loads(await td.get_document_by_id(ctx, 99))
            assert missing["success"] is False

        with patch("src.crawler.tool_definitions.get_session", side_effect=RuntimeError("db-error")):
            err = json.loads(await td.get_document_by_id(ctx, 1))
            assert err["success"] is False

    @pytest.mark.asyncio
    async def test_get_markdown_by_url_success_and_empty(self):
        ctx = _make_ctx()
        row1 = MagicMock(content="a")
        row2 = MagicMock(content="b")
        session = MagicMock()
        session.exec.return_value.all.return_value = [row1, row2]
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            out = json.loads(await td.get_markdown_by_url(ctx, "https://x.com"))
            assert out["success"] is True
            assert out["chunk_count"] == 2

        session2 = MagicMock()
        session2.exec.return_value.all.return_value = []
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session2)):
            out2 = json.loads(await td.get_markdown_by_url(ctx, "https://x.com"))
            assert out2["success"] is False

        with patch("src.crawler.tool_definitions.get_session", side_effect=RuntimeError("db-error-2")):
            out3 = json.loads(await td.get_markdown_by_url(ctx, "https://x.com"))
            assert out3["success"] is False

    @pytest.mark.asyncio
    async def test_crawl_url_dispatch_markdown(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True, "type": "markdown"})) as md:
            out = json.loads(await td.crawl_url(ctx, "https://x.com", mode="markdown", session_id="s1"))
            assert out["success"] is True
            assert out["type"] == "markdown"
            md.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_crawl_url_dispatch_smart(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.smart_crawl_url", new_callable=AsyncMock, return_value=json.dumps({"success": True, "type": "smart"})) as sm:
            out = json.loads(await td.crawl_url(ctx, "https://x.com", mode="smart", link_filter=r"/docs/.*"))
            assert out["success"] is True
            assert out["type"] == "smart"
            sm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_crawl_url_dispatch_legacy(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_single_page", new_callable=AsyncMock, return_value=json.dumps({"success": True, "type": "legacy"})) as lg:
            out = json.loads(await td.crawl_url(ctx, "https://x.com", mode="legacy"))
            assert out["success"] is True
            assert out["type"] == "legacy"
            lg.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_crawl_url_dispatch_deep(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_deep", new_callable=AsyncMock, return_value=json.dumps({"success": True, "type": "deep"})) as dp:
            out = json.loads(await td.crawl_url(
                ctx,
                "https://x.com",
                mode="deep",
                link_filter="*/docs/*",
                content_types=["text/html"],
                relevance_query="python",
                relevance_threshold=0.2,
                seo_threshold=0.4,
                seo_keywords=["docs"],
                scorer_type="none",
            ))
            assert out["success"] is True
            assert out["type"] == "deep"
            dp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_crawl_url_invalid_mode(self):
        ctx = _make_ctx()
        out = json.loads(await td.crawl_url(ctx, "https://x.com", mode="weird"))
        assert out["success"] is False
        assert "Invalid mode" in out["error"]


# ---------------------------------------------------------------------------
# Tests: Wave 3 — native Crawl4AI deep crawl strategies
# ---------------------------------------------------------------------------

class TestBuildDeepCrawlStrategy:
    def test_default_returns_bfs(self):
        s = td._build_deep_crawl_strategy()
        from crawl4ai import BFSDeepCrawlStrategy
        assert isinstance(s, BFSDeepCrawlStrategy)

    def test_dfs(self):
        s = td._build_deep_crawl_strategy(strategy="dfs")
        from crawl4ai import DFSDeepCrawlStrategy
        assert isinstance(s, DFSDeepCrawlStrategy)

    def test_best_first(self):
        s = td._build_deep_crawl_strategy(strategy="best_first")
        from crawl4ai import BestFirstCrawlingStrategy
        assert isinstance(s, BestFirstCrawlingStrategy)

    def test_unknown_strategy_falls_back_to_bfs(self):
        # Any unrecognised strategy string should still return BFS (default branch)
        s = td._build_deep_crawl_strategy(strategy="bfssss")
        from crawl4ai import BFSDeepCrawlStrategy
        assert isinstance(s, BFSDeepCrawlStrategy)

    def test_max_depth_clamped(self):
        s = td._build_deep_crawl_strategy(max_depth=0)
        assert s.max_depth == 1
        s2 = td._build_deep_crawl_strategy(max_depth=99)
        assert s2.max_depth == 10

    def test_url_pattern_filter_attached(self):
        s = td._build_deep_crawl_strategy(url_pattern="*/docs/*")
        # filter_chain should contain a URLPatternFilter
        filters = s.filter_chain.filters
        from crawl4ai import URLPatternFilter
        assert any(isinstance(f, URLPatternFilter) for f in filters)

    def test_domain_filter_attached(self):
        s = td._build_deep_crawl_strategy(allowed_domains=["example.com"])
        from crawl4ai import DomainFilter
        assert any(isinstance(f, DomainFilter) for f in s.filter_chain.filters)

    def test_keyword_scorer_attached(self):
        s = td._build_deep_crawl_strategy(keywords=["python", "tutorial"], strategy="best_first")
        assert s.url_scorer is not None

    def test_no_keywords_scorer_is_none(self):
        s = td._build_deep_crawl_strategy()
        assert s.url_scorer is None

    def test_empty_filter_chain_when_no_filters(self):
        s = td._build_deep_crawl_strategy()
        assert s.filter_chain is not None

    def test_content_type_filter_attached(self):
        s = td._build_deep_crawl_strategy(content_types=["text/html"])
        from crawl4ai import ContentTypeFilter
        assert any(isinstance(f, ContentTypeFilter) for f in s.filter_chain.filters)

    def test_content_relevance_filter_attached(self):
        s = td._build_deep_crawl_strategy(relevance_query="python", relevance_threshold=0.3)
        from crawl4ai import ContentRelevanceFilter
        assert any(isinstance(f, ContentRelevanceFilter) for f in s.filter_chain.filters)

    def test_seo_filter_attached(self):
        s = td._build_deep_crawl_strategy(seo_threshold=0.5, seo_keywords=["docs"])
        from crawl4ai import SEOFilter
        assert any(isinstance(f, SEOFilter) for f in s.filter_chain.filters)

    def test_scorer_type_none_disables_scorer(self):
        s = td._build_deep_crawl_strategy(
            strategy="best_first",
            keywords=["python"],
            scorer_type="none",
        )
        assert s.url_scorer is None

    def test_invalid_scorer_type_falls_back_to_keyword(self):
        s = td._build_deep_crawl_strategy(
            strategy="best_first",
            keywords=["python"],
            scorer_type="weird",
        )
        assert s.url_scorer is not None


class TestCrawlDeep:
    def _make_result(self, url="https://example.com", success=True, markdown="# Hello"):
        r = MagicMock()
        r.success = success
        r.url = url
        r.error_message = None
        r.depth = 0
        # markdown object with raw_markdown attribute
        if markdown is not None:
            md_obj = MagicMock()
            md_obj.raw_markdown = markdown
            md_obj.fit_markdown = "fit: " + markdown
            md_obj.markdown_with_citations = markdown
            md_obj.references_markdown = ""
            r.markdown = md_obj
        else:
            r.markdown = None
        return r

    @pytest.mark.asyncio
    async def test_success_bfs_no_index(self):
        r1 = self._make_result("https://example.com", markdown="# Page 1")
        r2 = self._make_result("https://example.com/about", markdown="# Page 2")
        container = [r1, r2]

        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = container
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is True
        assert data["strategy"] == "bfs"
        assert data["pages_crawled"] == 2
        assert data["pages_indexed"] == 0
        assert data["chunks_stored"] == 0
        # Run config passed to arun should have deep_crawl_strategy set
        config = mock_crawler.arun.call_args.kwargs["config"]
        assert config.deep_crawl_strategy is not None

    @pytest.mark.asyncio
    async def test_success_dfs_with_index(self):
        r1 = self._make_result("https://example.com", markdown="# DFS Page")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.store_crawled_documents", new_callable=AsyncMock, return_value=(1, 3)):
            data = json.loads(await td.crawl_deep(
                ctx, "https://example.com", strategy="dfs", index_result=True
            ))
        assert data["success"] is True
        assert data["strategy"] == "dfs"
        assert data["chunks_stored"] == 3
        assert data["pages_indexed"] == 1

    @pytest.mark.asyncio
    async def test_success_best_first_with_keywords(self):
        r1 = self._make_result("https://example.com/docs", markdown="# Docs")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(
            ctx, "https://example.com",
            strategy="best_first",
            keywords=["docs", "python"],
            index_result=False,
        ))
        assert data["success"] is True
        assert data["strategy"] == "best_first"
        # scorer should be wired in; verify by checking config's deep_crawl_strategy
        config = mock_crawler.arun.call_args.kwargs["config"]
        assert config.deep_crawl_strategy.url_scorer is not None

    @pytest.mark.asyncio
    async def test_invalid_strategy_returns_error(self):
        ctx = _make_ctx()
        data = json.loads(await td.crawl_deep(ctx, "https://example.com", strategy="sideways"))
        assert data["success"] is False
        assert "Invalid strategy" in data["error"]

    @pytest.mark.asyncio
    async def test_no_successful_results_returns_error(self):
        bad = self._make_result(success=False, markdown=None)
        bad.error_message = "timeout"
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [bad]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is False
        assert "No pages crawled successfully" in data["error"]
        assert len(data["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_mixed_results_partial_success(self):
        good = self._make_result("https://example.com", markdown="# Good")
        bad = self._make_result("https://example.com/fail", success=False, markdown=None)
        bad.error_message = "404"
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [good, bad]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is True
        assert data["pages_crawled"] == 1
        assert len(data["errors"]) == 1

    @pytest.mark.asyncio
    async def test_fit_variant_selected(self):
        r1 = self._make_result("https://example.com", markdown="# Fit Test")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(
            ctx, "https://example.com", markdown_variant="fit", index_result=False
        ))
        assert data["success"] is True
        assert data["selected_variant"] == "fit_markdown"

    @pytest.mark.asyncio
    async def test_max_depth_and_max_pages_clamped_in_response(self):
        r1 = self._make_result()
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(
            ctx, "https://example.com", max_depth=0, max_pages=9999, index_result=False
        ))
        assert data["max_depth_configured"] == 1
        assert data["max_pages_configured"] == 500

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = RuntimeError("network error")
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is False
        assert "network error" in data["error"]

    @pytest.mark.asyncio
    async def test_url_pattern_filter_wired_into_strategy(self):
        r1 = self._make_result()
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(
            ctx, "https://example.com", url_pattern="*/docs/*", index_result=False
        ))
        assert data["success"] is True
        config = mock_crawler.arun.call_args.kwargs["config"]
        from crawl4ai import URLPatternFilter
        assert any(isinstance(f, URLPatternFilter) for f in config.deep_crawl_strategy.filter_chain.filters)

    @pytest.mark.asyncio
    async def test_allowed_domains_wired_into_strategy(self):
        r1 = self._make_result()
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(
            ctx, "https://example.com", allowed_domains=["example.com"], index_result=False
        ))
        assert data["success"] is True
        config = mock_crawler.arun.call_args.kwargs["config"]
        from crawl4ai import DomainFilter
        assert any(isinstance(f, DomainFilter) for f in config.deep_crawl_strategy.filter_chain.filters)

    @pytest.mark.asyncio
    async def test_urls_crawled_sample_truncated(self):
        results = [self._make_result(url=f"https://example.com/p{i}") for i in range(8)]
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = results
        ctx = _make_ctx(crawler=mock_crawler)

        data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is True
        assert data["urls_crawled_sample"][-1] == "..."
        assert len(data["urls_crawled_sample"]) == 6  # 5 + "..."

    @pytest.mark.asyncio
    async def test_prefetch_only_disables_indexing(self):
        r1 = self._make_result("https://example.com", markdown="# P1")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r1]
        ctx = _make_ctx(crawler=mock_crawler)

        with patch("src.crawler.tool_definitions.store_crawled_documents", new_callable=AsyncMock) as store_docs:
            data = json.loads(await td.crawl_deep(
                ctx,
                "https://example.com",
                index_result=True,
                prefetch_only=True,
            ))
        assert data["success"] is True
        assert data["prefetch_only"] is True
        assert data["index_result"] is False
        assert data["pages_indexed"] == 0
        store_docs.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_result_with_empty_markdown_variant_goes_to_errors(self):
        """A result that has success=True but all markdown variants empty goes to errors."""
        r = MagicMock()
        r.success = True
        r.url = "https://example.com"
        r.error_message = None
        r.depth = 0
        r.markdown = MagicMock()

        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = [r]
        ctx = _make_ctx(crawler=mock_crawler)

        _empty_variants = {
            "raw_markdown": "",
            "fit_markdown": "",
            "markdown_with_citations": "",
            "references_markdown": "",
            "fit_html": "",
        }
        with patch("src.crawler.tool_definitions._extract_markdown_variants", return_value=_empty_variants):
            data = json.loads(await td.crawl_deep(ctx, "https://example.com", index_result=False))
        assert data["success"] is False  # all pages failed
        assert any("Empty markdown variant" in e["error"] for e in data["errors"])

    @pytest.mark.asyncio
    async def test_stream_mode_results_processed(self):
        class _AsyncResults:
            def __init__(self, rows):
                self._rows = rows

            def __aiter__(self):
                self._it = iter(self._rows)
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        r1 = self._make_result("https://example.com", markdown="# P1")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = _AsyncResults([r1])
        ctx = _make_ctx(crawler=mock_crawler)

        out = json.loads(await td.crawl_deep(
            ctx,
            "https://example.com",
            run_config={"stream": True},
            index_result=False,
            content_types=["text/html"],
            relevance_query="python",
            relevance_threshold=0.2,
            seo_threshold=0.4,
            scorer_type="none",
        ))

        assert out["success"] is True
        assert out["stream_mode"] is True
        assert out["content_type_filter_applied"] is True
        assert out["content_relevance_filter_applied"] is True
        assert out["seo_filter_applied"] is True
        assert out["scorer_type_applied"] == "none"

    @pytest.mark.asyncio
    async def test_stream_mode_failed_result_recorded_as_error(self):
        class _AsyncResults:
            def __init__(self, rows):
                self._rows = rows

            def __aiter__(self):
                self._it = iter(self._rows)
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        bad = MagicMock(success=False, markdown=None, url="https://example.com/bad", error_message="blocked")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = _AsyncResults([bad])
        ctx = _make_ctx(crawler=mock_crawler)

        out = json.loads(await td.crawl_deep(ctx, "https://example.com", run_config={"stream": True}, index_result=False))
        assert out["success"] is False
        assert out["pages_crawled"] == 0
        assert any("blocked" in e["error"] for e in out["errors"])

    @pytest.mark.asyncio
    async def test_stream_mode_empty_variant_recorded_as_error(self):
        class _AsyncResults:
            def __init__(self, rows):
                self._rows = rows

            def __aiter__(self):
                self._it = iter(self._rows)
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        good = self._make_result("https://example.com", markdown="# x")
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = _AsyncResults([good])
        ctx = _make_ctx(crawler=mock_crawler)

        _empty_variants = {
            "raw_markdown": "",
            "fit_markdown": "",
            "markdown_with_citations": "",
            "references_markdown": "",
            "fit_html": "",
        }
        with patch("src.crawler.tool_definitions._extract_markdown_variants", return_value=_empty_variants):
            out = json.loads(await td.crawl_deep(ctx, "https://example.com", run_config={"stream": True}, index_result=False))

        assert out["success"] is False
        assert any("Empty markdown variant" in e["error"] for e in out["errors"])


class TestCrawlAdaptive:
    def _make_result(self, url="https://example.com", success=True, markdown="# Hello"):
        r = MagicMock()
        r.success = success
        r.url = url
        r.error_message = None
        r.depth = 0
        if markdown is not None:
            md_obj = MagicMock()
            md_obj.raw_markdown = markdown
            md_obj.fit_markdown = "fit: " + markdown
            md_obj.markdown_with_citations = markdown
            md_obj.references_markdown = ""
            r.markdown = md_obj
        else:
            r.markdown = None
        return r

    @pytest.mark.asyncio
    async def test_invalid_strategy_returns_error(self):
        ctx = _make_ctx()
        out = json.loads(await td.crawl_adaptive(ctx, "https://example.com", query="q", strategy="bad"))
        assert out["success"] is False
        assert "Invalid strategy" in out["error"]

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        ctx = _make_ctx()
        out = json.loads(await td.crawl_adaptive(ctx, "https://example.com", query="   "))
        assert out["success"] is False
        assert "non-empty" in out["error"]

    @pytest.mark.asyncio
    async def test_success_no_index(self):
        ctx = _make_ctx(crawler=AsyncMock())
        r1 = self._make_result("https://example.com", markdown="# A")
        state = MagicMock(knowledge_base=[r1])

        adaptive_inst = MagicMock()
        adaptive_inst.digest = AsyncMock(return_value=state)
        adaptive_inst.get_relevant_content.return_value = [{"url": "https://example.com", "score": 0.9}]
        adaptive_inst.confidence = 0.88
        adaptive_inst.coverage_stats = {"pages": 1}

        with patch("src.crawler.tool_definitions.AdaptiveCrawler", return_value=adaptive_inst):
            out = json.loads(await td.crawl_adaptive(
                ctx,
                "https://example.com",
                query="what is this site about",
                index_result=False,
            ))

        assert out["success"] is True
        assert out["strategy"] == "statistical"
        assert out["pages_crawled"] == 1
        assert out["pages_indexed"] == 0
        assert out["chunks_stored"] == 0
        assert out["confidence"] == 0.88
        assert out["coverage_stats"]["pages"] == 1

    @pytest.mark.asyncio
    async def test_success_with_index(self):
        ctx = _make_ctx(crawler=AsyncMock())
        r1 = self._make_result("https://example.com", markdown="# A")
        state = MagicMock(knowledge_base=[r1])

        adaptive_inst = MagicMock()
        adaptive_inst.digest = AsyncMock(return_value=state)
        adaptive_inst.get_relevant_content.return_value = []
        adaptive_inst.confidence = 0.75
        adaptive_inst.coverage_stats = {}

        with patch("src.crawler.tool_definitions.AdaptiveCrawler", return_value=adaptive_inst), \
             patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.store_crawled_documents", new_callable=AsyncMock, return_value=(1, 2)):
            out = json.loads(await td.crawl_adaptive(
                ctx,
                "https://example.com",
                query="query",
                strategy="embedding",
                index_result=True,
            ))

        assert out["success"] is True
        assert out["strategy"] == "embedding"
        assert out["pages_indexed"] == 1
        assert out["chunks_stored"] == 2

    @pytest.mark.asyncio
    async def test_no_successful_pages_returns_error(self):
        ctx = _make_ctx(crawler=AsyncMock())
        bad = self._make_result(success=False, markdown=None)
        bad.error_message = "timeout"
        state = MagicMock(knowledge_base=[bad])

        adaptive_inst = MagicMock()
        adaptive_inst.digest = AsyncMock(return_value=state)
        adaptive_inst.get_relevant_content.return_value = []
        adaptive_inst.confidence = 0.1
        adaptive_inst.coverage_stats = {}

        with patch("src.crawler.tool_definitions.AdaptiveCrawler", return_value=adaptive_inst):
            out = json.loads(await td.crawl_adaptive(ctx, "https://example.com", query="query"))

        assert out["success"] is False
        assert "no successful pages" in out["error"].lower()
        assert out["pages_crawled"] == 0

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        ctx = _make_ctx(crawler=AsyncMock())
        adaptive_inst = MagicMock()
        adaptive_inst.digest = AsyncMock(side_effect=RuntimeError("adaptive boom"))

        with patch("src.crawler.tool_definitions.AdaptiveCrawler", return_value=adaptive_inst):
            out = json.loads(await td.crawl_adaptive(ctx, "https://example.com", query="query"))

        assert out["success"] is False
        assert "adaptive boom" in out["error"]

    @pytest.mark.asyncio
    async def test_empty_markdown_variant_paths_to_error(self):
        ctx = _make_ctx(crawler=AsyncMock())
        r1 = self._make_result("https://example.com", markdown="# A")
        state = MagicMock(knowledge_base=[r1])

        adaptive_inst = MagicMock()
        adaptive_inst.digest = AsyncMock(return_value=state)
        adaptive_inst.get_relevant_content.return_value = []
        adaptive_inst.confidence = 0.4
        adaptive_inst.coverage_stats = {}

        _empty_variants = {
            "raw_markdown": "",
            "fit_markdown": "",
            "markdown_with_citations": "",
            "references_markdown": "",
            "fit_html": "",
        }
        with patch("src.crawler.tool_definitions.AdaptiveCrawler", return_value=adaptive_inst), \
             patch("src.crawler.tool_definitions._extract_markdown_variants", return_value=_empty_variants):
            out = json.loads(await td.crawl_adaptive(ctx, "https://example.com", query="query"))

        assert out["success"] is False
        assert any("Empty markdown variant" in e["error"] for e in out["errors"])


class TestPhase1TaxonomyWrappers:
    @pytest.mark.asyncio
    async def test_crawl_with_session_validation_and_kill(self):
        ctx = _make_ctx()
        bad = json.loads(await td.crawl_with_session(ctx, url="https://x.com", session_id="   "))
        assert bad["success"] is False

        bad_action = json.loads(await td.crawl_with_session(ctx, url="https://x.com", session_id="s1", action="bad"))
        assert bad_action["success"] is False

        killed = json.loads(await td.crawl_with_session(ctx, session_id="s1", action="kill"))
        assert killed["success"] is True
        assert killed["action"] == "kill"

    @pytest.mark.asyncio
    async def test_crawl_with_session_dispatch_paths(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_many_urls", new_callable=AsyncMock, return_value=json.dumps({"success": True, "path": "many"})) as many:
            out_many = json.loads(await td.crawl_with_session(ctx, urls=["https://x.com"], session_id="sess1"))
            assert out_many["path"] == "many"
            many.assert_awaited_once()

        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True, "path": "one"})) as one:
            out_one = json.loads(await td.crawl_with_session(ctx, url="https://x.com", session_id="sess1"))
            assert out_one["path"] == "one"
            one.assert_awaited_once()

        none = json.loads(await td.crawl_with_session(ctx, session_id="sess1"))
        assert none["success"] is False

    @pytest.mark.asyncio
    async def test_inspect_session(self):
        ctx = _make_ctx()
        bad = json.loads(await td.inspect_session(ctx, "   "))
        assert bad["success"] is False

        ok = json.loads(await td.inspect_session(ctx, " s-1 "))
        assert ok["success"] is True
        assert ok["session_id"] == "s-1"

    @pytest.mark.asyncio
    async def test_create_and_kill_session(self):
        ctx = _make_ctx()
        bad_create = json.loads(await td.create_session(ctx, "   "))
        assert bad_create["success"] is False
        ok_create = json.loads(await td.create_session(ctx, " s-2 "))
        assert ok_create["success"] is True
        assert ok_create["session_id"] == "s-2"

        bad_kill = json.loads(await td.kill_session(ctx, "   "))
        assert bad_kill["success"] is False
        ok_kill = json.loads(await td.kill_session(ctx, " s-2 "))
        assert ok_kill["success"] is True
        assert ok_kill["session_id"] == "s-2"

    @pytest.mark.asyncio
    async def test_crawl_with_browser_config_paths(self):
        ctx = _make_ctx()
        crawler_instance = AsyncMock()
        md = MagicMock(raw_markdown="# x", fit_markdown="", markdown_with_citations="", references_markdown="")
        crawler_instance.arun.return_value = MagicMock(success=True, markdown=md, url="https://x.com")
        crawler_instance.__aenter__.return_value = crawler_instance
        crawler_instance.__aexit__.return_value = None

        with patch("src.crawler.tool_definitions.AsyncWebCrawler", return_value=crawler_instance):
            out = json.loads(await td.crawl_with_browser_config(
                ctx,
                "https://x.com",
                browser_config={"headless": False, "unsafe": True},
                index_result=False,
            ))
            assert out["success"] is True
            assert out["browser_config_applied"] == {"headless": False}

        with patch("src.crawler.tool_definitions.AsyncWebCrawler", return_value=crawler_instance), \
             patch("src.crawler.tool_definitions.index_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True, "chunks_stored": 2})):
            out2 = json.loads(await td.crawl_with_browser_config(ctx, "https://x.com", index_result=True))
            assert out2["success"] is True
            assert out2["chunks_stored"] == 2

        crawler_fail = AsyncMock()
        crawler_fail.__aenter__.return_value = crawler_fail
        crawler_fail.__aexit__.return_value = None
        crawler_fail.arun.return_value = MagicMock(success=False, markdown=None, error_message="nope")
        with patch("src.crawler.tool_definitions.AsyncWebCrawler", return_value=crawler_fail):
            bad = json.loads(await td.crawl_with_browser_config(ctx, "https://x.com"))
            assert bad["success"] is False

        with patch("src.crawler.tool_definitions.AsyncWebCrawler", side_effect=RuntimeError("browser-boom")):
            err = json.loads(await td.crawl_with_browser_config(ctx, "https://x.com"))
            assert err["success"] is False

    @pytest.mark.asyncio
    async def test_extract_wrappers_delegate(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True})) as ctm:
            out = json.loads(await td.extract_fit_markdown(ctx, "https://x.com"))
            assert out["success"] is True
            assert ctm.await_count == 1

            out2 = json.loads(await td.extract_structured_json(ctx, "https://x.com", extraction_strategy="css", extraction_schema={"title": "h1"}))
            assert out2["success"] is True

            out2b = json.loads(await td.extract_structured_json(ctx, file_path="README.md", extraction_strategy="css", extraction_schema={"title": "h1"}))
            assert out2b["success"] is True

            out2c = json.loads(await td.extract_structured_json(ctx, html="<h1>x</h1>", extraction_strategy="css", extraction_schema={"title": "h1"}))
            assert out2c["success"] is True

            out3 = json.loads(await td.extract_regex_entities(ctx, "https://x.com", extraction_patterns={"email": ".+@.+"}))
            assert out3["success"] is True

            out4 = json.loads(await td.extract_knowledge_graph(ctx, "https://x.com"))
            assert out4["success"] is True

        bad1 = json.loads(await td.extract_structured_json(ctx))
        assert bad1["success"] is False

        bad2 = json.loads(await td.extract_structured_json(ctx, html="   "))
        assert bad2["success"] is False

    @pytest.mark.asyncio
    async def test_extract_markdown_variants_paths(self):
        md = MagicMock(
            raw_markdown="# raw",
            fit_markdown="# fit",
            markdown_with_citations="# cited",
            references_markdown="refs",
            fit_html="<p>fit</p>",
        )
        crawler = AsyncMock()
        crawler.arun.return_value = MagicMock(success=True, markdown=md, url="https://x.com")
        ctx = _make_ctx(crawler=crawler)

        out = json.loads(await td.extract_markdown_variants(ctx, "https://x.com", index_result=False))
        assert out["success"] is True
        assert out["fit_markdown"] == "# fit"
        assert out["chunks_stored"] == 0

        with patch("src.crawler.tool_definitions.index_markdown", new_callable=AsyncMock, return_value=json.dumps({"success": True, "chunks_stored": 3})):
            out2 = json.loads(await td.extract_markdown_variants(ctx, "https://x.com", index_result=True))
            assert out2["success"] is True
            assert out2["chunks_stored"] == 3

        crawler2 = AsyncMock()
        crawler2.arun.return_value = MagicMock(success=False, markdown=None, error_message="fail")
        ctx2 = _make_ctx(crawler=crawler2)
        bad = json.loads(await td.extract_markdown_variants(ctx2, "https://x.com"))
        assert bad["success"] is False

        crawler3 = AsyncMock()
        crawler3.arun.side_effect = RuntimeError("mv-boom")
        ctx3 = _make_ctx(crawler=crawler3)
        err = json.loads(await td.extract_markdown_variants(ctx3, "https://x.com"))
        assert err["success"] is False

    @pytest.mark.asyncio
    async def test_extract_code_examples_success_and_errors(self):
        ctx = _make_ctx(crawler=AsyncMock())
        md_obj = MagicMock(raw_markdown="```python\nprint('x')\n```", fit_markdown="", markdown_with_citations="", references_markdown="")
        ctx.lifespan_context.crawler.arun.return_value = MagicMock(success=True, markdown=md_obj, url="https://x.com")
        out = json.loads(await td.extract_code_examples(ctx, "https://x.com"))
        assert out["success"] is True
        assert out["count"] >= 1

        ctx2 = _make_ctx(crawler=AsyncMock())
        ctx2.lifespan_context.crawler.arun.return_value = MagicMock(success=False, markdown=None, error_message="nope")
        bad = json.loads(await td.extract_code_examples(ctx2, "https://x.com"))
        assert bad["success"] is False

        ctx3 = _make_ctx(crawler=AsyncMock())
        ctx3.lifespan_context.crawler.arun.side_effect = RuntimeError("boom")
        err = json.loads(await td.extract_code_examples(ctx3, "https://x.com"))
        assert err["success"] is False

    @pytest.mark.asyncio
    async def test_index_markdown_and_variants(self):
        ctx = _make_ctx()
        captured_metas = []
        async def _capture_add(session, urls, contents, metas, chunks, fulldocs):
            captured_metas.extend(metas)
            return len(urls)
        with patch("src.crawler.tool_definitions.chunk_text_according_to_settings", new_callable=AsyncMock, return_value=["c1", "c2"]), \
             patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.add_documents_to_db", new_callable=AsyncMock, side_effect=_capture_add):
            ok = json.loads(await td.index_markdown(ctx, "https://x.com", "# title"))
            assert ok["success"] is True
            assert ok["chunks_stored"] == 2
            assert captured_metas and captured_metas[0].get("source_type") == "remote_url"

            fit = json.loads(await td.index_fit_markdown(ctx, "https://x.com", "# fit"))
            assert fit["success"] is True

            struct = json.loads(await td.index_structured_content(ctx, "https://x.com", {"k": "v"}))
            assert struct["success"] is True

        empty = json.loads(await td.index_markdown(ctx, "https://x.com", "  "))
        assert empty["success"] is False

        with patch("src.crawler.tool_definitions.chunk_text_according_to_settings", new_callable=AsyncMock, side_effect=RuntimeError("idx-boom")):
            err = json.loads(await td.index_markdown(ctx, "https://x.com", "# m"))
            assert err["success"] is False

    @pytest.mark.asyncio
    async def test_index_code_examples_paths(self):
        ctx = _make_ctx()
        captured = {}
        async def _cap_add_code(session, urls, contents, languages, summaries, metadatas, chunk_numbers):
            captured["metadatas"] = metadatas
            return 1
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions.add_code_examples_to_db", new_callable=AsyncMock, side_effect=_cap_add_code):
            ok = json.loads(await td.index_code_examples(ctx, "https://x.com", "```python\nprint('x')\n```"))
            assert ok["success"] is True
            assert ok["code_examples_indexed"] >= 1
            assert captured["metadatas"][0]["source_type"] == "remote_url"

        no_md = json.loads(await td.index_code_examples(ctx, "https://x.com", "   "))
        assert no_md["success"] is False

        no_code = json.loads(await td.index_code_examples(ctx, "https://x.com", "just plain text"))
        assert no_code["success"] is True
        assert no_code["code_examples_indexed"] == 0

        with patch("src.crawler.tool_definitions.extract_code_blocks", side_effect=RuntimeError("code-boom")):
            err = json.loads(await td.index_code_examples(ctx, "https://x.com", "```py\n1\n```"))
            assert err["success"] is False

    @pytest.mark.asyncio
    async def test_search_and_get_fit_tools(self):
        ctx = _make_ctx()
        with patch("src.crawler.tool_definitions.perform_rag_query", new_callable=AsyncMock, return_value=json.dumps({"success": True, "results": []})):
            out = json.loads(await td.search_documents_v2(ctx, "q"))
            assert out["success"] is True

        fake_results = [{"url": "https://x.com", "content": "{}", "page_metadata": {"content_class": "structured"}, "similarity_score": 0.9}]
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_documents_core", new_callable=AsyncMock, return_value=fake_results):
            out2 = json.loads(await td.search_structured_content(ctx, "q"))
            assert out2["success"] is True
            assert len(out2["results"]) == 1

        captured = {}
        async def _fake_struct_search(session, query, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return fake_results
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_documents_core", side_effect=_fake_struct_search):
            out2b = json.loads(await td.search_structured_content(ctx, "q", source="example.com"))
            assert out2b["success"] is True
            assert captured["filter"]["source"] == "example.com"

        captured2 = {}
        async def _fake_struct_search2(session, query, match_count, filter_metadata):
            captured2["filter"] = filter_metadata
            return fake_results
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session()), \
             patch("src.crawler.tool_definitions._search_documents_core", side_effect=_fake_struct_search2):
            out2c = json.loads(await td.search_structured_content(ctx, "q", content_class="table"))
            assert out2c["success"] is True
            assert captured2["filter"]["content_class"] == "table"

        with patch("src.crawler.tool_definitions.get_session", side_effect=RuntimeError("search-boom")):
            out3 = json.loads(await td.search_structured_content(ctx, "q"))
            assert out3["success"] is False

        fit_row = MagicMock(content="fit one", page_metadata={"markdown_variant": "fit_markdown"})
        other_row = MagicMock(content="raw one", page_metadata={"markdown_variant": "raw_markdown"})
        session = MagicMock()
        session.exec.return_value.all.return_value = [fit_row, other_row]
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session)):
            out4 = json.loads(await td.get_fit_markdown_by_url(ctx, "https://x.com"))
            assert out4["success"] is True
            assert out4["chunk_count"] == 1

        session2 = MagicMock()
        session2.exec.return_value.all.return_value = [other_row]
        with patch("src.crawler.tool_definitions.get_session", side_effect=_make_get_session(session2)):
            out5 = json.loads(await td.get_fit_markdown_by_url(ctx, "https://x.com"))
            assert out5["success"] is False

        with patch("src.crawler.tool_definitions.get_session", side_effect=RuntimeError("fit-boom")):
            out6 = json.loads(await td.get_fit_markdown_by_url(ctx, "https://x.com"))
            assert out6["success"] is False

