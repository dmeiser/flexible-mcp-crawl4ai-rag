"""Unit tests for src/tools/tool_definitions.py — 100% coverage, offline."""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

import src.tools.tool_definitions as td

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
# Tests: crawl_url + search_documents_v2
# ---------------------------------------------------------------------------


class TestCrawlUrl:
    @pytest.mark.asyncio
    async def test_markdown_mode_dispatches(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.crawl_to_markdown", new_callable=AsyncMock, return_value='{"success": true}'
        ) as mock_md:
            result = await td.crawl_url(ctx, url="https://x.com", mode="markdown", markdown_variant="fit")
        data = json.loads(result)
        assert data["success"] is True
        mock_md.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deep_mode_dispatches(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.crawl_deep",
            new_callable=AsyncMock,
            return_value='{"success": true, "pages_crawled": 1}',
        ) as mock_deep:
            result = await td.crawl_url(ctx, url="https://x.com", mode="deep", max_depth=2)
        data = json.loads(result)
        assert data["success"] is True
        mock_deep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_removed_mode_returns_error(self):
        ctx = _make_ctx()
        result = await td.crawl_url(ctx, url="https://x.com", mode="legacy")
        data = json.loads(result)
        assert data["success"] is False
        assert "removed" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        ctx = _make_ctx()
        result = await td.crawl_url(ctx, url="https://x.com", mode="invalid")
        data = json.loads(result)
        assert data["success"] is False
        assert "Invalid mode" in data["error"]


class TestSearchWeb:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self):
        ctx = _make_ctx()
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", ""),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", "model"),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is False
        assert "WEB_SEARCH_API_KEY" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_model_name_returns_error(self):
        ctx = _make_ctx()
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", "secret"),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", ""),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is False
        assert "WEB_SEARCH_MODEL_NAME" in data["error"]

    @pytest.mark.asyncio
    async def test_successful_search_no_cache(self):
        ctx = _make_ctx()
        normalized = {
            "answer": "Hello",
            "sources": [{"rank": 1, "url": "https://example.com", "title": "Example", "snippet": "Snippet"}],
            "search_params": {"engine": "auto", "max_results": 5},
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "openrouter/test-model",
        }
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", "secret"),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", "model"),
            patch.object(td.settings, "WEB_SEARCH_CACHE_ENABLED", False),
            patch("src.tools.tool_definitions.execute_web_search", new=AsyncMock(return_value=normalized)) as mock_exec,
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is True
        assert data["answer"] == "Hello"
        assert data["cached"] is False
        mock_exec.assert_awaited_once_with("query", "auto", 5, None, None)

    @pytest.mark.asyncio
    async def test_successful_search_with_cache(self):
        ctx = _make_ctx()
        session = MagicMock()
        normalized = {
            "query": "query",
            "answer": "Hello",
            "sources": [{"rank": 1, "url": "https://example.com", "title": "Example", "snippet": "Snippet"}],
            "search_params": {"engine": "auto", "max_results": 5},
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "openrouter/test-model",
        }
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", "secret"),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", "model"),
            patch.object(td.settings, "WEB_SEARCH_CACHE_ENABLED", True),
            patch("src.tools.tool_definitions.execute_web_search", new=AsyncMock(return_value=normalized)),
            patch("src.tools.tool_definitions.cache_web_search_results", new=AsyncMock(return_value=3)) as mock_cache,
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is True
        assert data["cached"] is True
        mock_cache.assert_awaited_once_with(session, normalized)

    @pytest.mark.asyncio
    async def test_cache_failure_does_not_fail_tool(self):
        ctx = _make_ctx()
        normalized = {
            "query": "query",
            "answer": "Hello",
            "sources": [],
            "search_params": {"engine": "auto", "max_results": 5},
            "usage": {},
            "model": "openrouter/test-model",
        }
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", "secret"),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", "model"),
            patch.object(td.settings, "WEB_SEARCH_CACHE_ENABLED", True),
            patch("src.tools.tool_definitions.execute_web_search", new=AsyncMock(return_value=normalized)),
            patch(
                "src.tools.tool_definitions.cache_web_search_results",
                new=AsyncMock(side_effect=RuntimeError("cache broke")),
            ),
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is True
        assert data["cached"] is False

    @pytest.mark.asyncio
    async def test_execute_web_search_exception_returns_error(self):
        ctx = _make_ctx()
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "openrouter"),
            patch.object(td.settings, "WEB_SEARCH_API_KEY", "secret"),
            patch.object(td.settings, "WEB_SEARCH_MODEL_NAME", "model"),
            patch("src.tools.tool_definitions.execute_web_search", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is False
        assert data["error"] == "boom"

    @pytest.mark.asyncio
    async def test_unsupported_provider_returns_error(self):
        ctx = _make_ctx()
        with (
            patch.object(td.settings, "WEB_SEARCH_PROVIDER", "brave"),
            patch(
                "src.tools.tool_definitions.execute_web_search",
                new=AsyncMock(side_effect=ValueError("Unsupported WEB_SEARCH_PROVIDER: brave")),
            ),
        ):
            result = await td.search_web(ctx, "query")
        data = json.loads(result)
        assert data["success"] is False
        assert "Unsupported WEB_SEARCH_PROVIDER" in data["error"]


class TestSearchDocumentsV2:
    @pytest.mark.asyncio
    async def test_basic_query_no_reranking(self):
        ctx = _make_ctx()
        raw_results = [{"url": "u", "content": "c", "page_metadata": {"x": 1}, "similarity_score": 0.8}]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", new_callable=AsyncMock, return_value=raw_results),
        ):
            result = await td.search_documents_v2(ctx, "test query")

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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", side_effect=fake_search),
        ):
            await td.search_documents_v2(ctx, "q", source="docs.x.com")

        assert captured["filter"] == {"source": "docs.x.com"}

    @pytest.mark.asyncio
    async def test_phase9_filters_forwarded(self):
        ctx = _make_ctx()
        captured: dict = {}

        async def fake_search(sess, q, match_count, filter_metadata):
            captured["filter"] = filter_metadata
            return []

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", side_effect=fake_search),
        ):
            await td.search_documents_v2(
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", side_effect=fake_search),
        ):
            await td.search_documents_v2(ctx, "q")

        assert captured["filter"] is None

    @pytest.mark.asyncio
    async def test_reranking_applied(self):
        ctx = _make_ctx()
        raw_results = [
            {"url": "u1", "content": "c", "page_metadata": {}, "similarity_score": 0.8},
            {"url": "u2", "content": "d", "page_metadata": {}, "similarity_score": 0.6},
        ]
        reranked = [raw_results[1], raw_results[0]]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=True)),
            patch("src.tools.tool_definitions.search_documents", new_callable=AsyncMock, return_value=raw_results),
            patch("src.tools.tool_definitions.rerank_results", return_value=reranked) as mock_rerank,
        ):
            result = await td.search_documents_v2(ctx, "q", match_count=2)

        data = json.loads(result)
        assert data["success"] is True
        mock_rerank.assert_called_once_with("q", raw_results, top_k=2)
        assert data["results"][0]["url"] == "u2"

    @pytest.mark.asyncio
    async def test_include_provenance_adds_provenance_payload(self):
        ctx = _make_ctx()
        raw_results = [
            {
                "url": "u",
                "content": "c",
                "page_metadata": {
                    "source": "docs.x.com",
                    "url": "u",
                    "source_type": "remote_url",
                    "markdown_variant": "raw_markdown",
                    "references_markdown": "[1]: https://example.com/ref Example reference",
                    "link_references": [{"label": "1", "url": "https://example.com/ref", "text": "Example reference"}],
                    "has_citations": True,
                },
                "similarity_score": 0.8,
            }
        ]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", new_callable=AsyncMock, return_value=raw_results),
        ):
            result = await td.search_documents_v2(ctx, "test query", include_provenance=True)

        data = json.loads(result)
        provenance = data["results"][0]["provenance"]
        assert provenance["source"] == "docs.x.com"
        assert provenance["has_citations"] is True
        assert provenance["link_references"][0]["url"] == "https://example.com/ref"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", side_effect=RuntimeError("search crash")),
        ):
            result = await td.search_documents_v2(ctx, "q")

        data = json.loads(result)
        assert data["success"] is False
        assert "search crash" in data["error"]

    @pytest.mark.asyncio
    async def test_fresh_only_filters_stale_and_expired_results(self):
        ctx = _make_ctx()
        raw_results = [
            {
                "url": "fresh",
                "content": "fresh",
                "page_metadata": {"staleness_score": 0.1, "expires_at": "2999-01-01T00:00:00+00:00"},
                "similarity_score": 0.9,
            },
            {
                "url": "stale",
                "content": "stale",
                "page_metadata": {"staleness_score": 0.9, "expires_at": "2999-01-01T00:00:00+00:00"},
                "similarity_score": 0.95,
            },
        ]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", new_callable=AsyncMock, return_value=raw_results),
        ):
            result = await td.search_documents_v2(ctx, "q", fresh_only=True, staleness_threshold=0.5)

        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["url"] == "fresh"

    @pytest.mark.asyncio
    async def test_as_of_and_recency_bias_fields_present(self):
        ctx = _make_ctx()
        raw_results = [
            {
                "url": "u",
                "content": "c",
                "page_metadata": {"crawl_timestamp": "2026-01-01T00:00:00+00:00", "staleness_score": 0.2},
                "similarity_score": 0.8,
            }
        ]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions.settings", MagicMock(USE_RERANKING=False)),
            patch("src.tools.tool_definitions.search_documents", new_callable=AsyncMock, return_value=raw_results),
        ):
            result = await td.search_documents_v2(
                ctx,
                "q",
                as_of="2027-01-01T00:00:00+00:00",
                recency_bias=0.5,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["recency_bias"] == 0.5
        assert "final_score" in data["results"][0]
        assert "freshness_score" in data["results"][0]


# ---------------------------------------------------------------------------
# Tests: search_code_examples (tool)
# ---------------------------------------------------------------------------


class TestSearchCodeExamplesTool:
    @pytest.mark.asyncio
    async def test_success(self):
        ctx = _make_ctx()
        code_results = [
            {"url": "u", "language": "python", "content": "print()", "summary": "prints", "similarity_score": 0.95}
        ]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch(
                "src.tools.tool_definitions._search_code_examples", new_callable=AsyncMock, return_value=code_results
            ),
        ):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions._search_code_examples", side_effect=fake_search),
        ):
            await td.search_code_examples(ctx, "q", language="rust")

        assert captured["language"] == "rust"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session()),
            patch("src.tools.tool_definitions._search_code_examples", side_effect=RuntimeError("vector fail")),
        ):
            result = await td.search_code_examples(ctx, "q")

        data = json.loads(result)
        assert data["success"] is False
        assert "vector fail" in data["error"]


# ---------------------------------------------------------------------------
# Helpers for Phase 9.5/9.6 tool tests
# ---------------------------------------------------------------------------


def _make_mock_page(
    uid=1,
    value_score=0.1,
    staleness_score=0.8,
    hit_count=0,
    is_pinned=False,
    tombstoned_at=None,
    is_active=True,
    source="test.com",
):
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


def _make_mock_code(
    uid=10,
    value_score=0.2,
    staleness_score=0.5,
    hit_count=0,
    is_pinned=False,
    tombstoned_at=None,
    is_active=True,
    source="test.com",
):
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
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [mock_page]  # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = [mock_code]  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [page_match, page_other]  # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = []  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.compute_value_scores(ctx, source="target.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["updated_crawled_pages"] == 1  # only target.com matched

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("db crash")

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [page]  # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = []  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.preview_eviction_plan(ctx, limit=10, dry_run=True)

        data = json.loads(result)
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["candidates_count"] == 1
        assert data["total_evicted"] == 0
        assert "last_seen_at" in data["candidates"][0]

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
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [page]  # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = []  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
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
                m.all.return_value = []  # CrawledPage eviction candidates — empty
            elif call_n[0] == 2:
                m.all.return_value = [code]  # CodeExample eviction candidates — one record
            else:
                m.all.return_value = []  # active coverage maps + SourcePolicy
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
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
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 2:
                m.all.return_value = [p_match, p_other]  # CrawledPage
            elif call_n[0] == 3:
                m.all.return_value = []  # CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.preview_eviction_plan(ctx, source="target.com")

        data = json.loads(result)
        assert data["candidates_count"] == 1
        assert data["candidates"][0]["source"] == "target.com"

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("query fail")

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=1_000_000),
        ):
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
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=1_000_000),
        ):
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
                m.all.return_value = []  # tombstoned queries
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch(
                "src.tools.tool_definitions._enforce_source_quotas",
                return_value={"quota_evicted": 0, "sources_over_quota": []},
            ),
            patch(
                "src.tools.tool_definitions._enforce_table_budgets",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=1_000_000),
        ):
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
                m.first.return_value = None  # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] == 3:
                m.all.return_value = [expired_rec]  # CrawledPage expired
            elif call_n[0] == 4:
                m.all.return_value = []  # CodeExample expired
            else:
                m.all.return_value = []
            return m

        session.exec.side_effect = exec_se

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch(
                "src.tools.tool_definitions._enforce_source_quotas",
                return_value={"quota_evicted": 0, "sources_over_quota": []},
            ),
            patch(
                "src.tools.tool_definitions._enforce_table_budgets",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=1_000_000),
        ):
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
                m.first.return_value = None  # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] in (3, 4):
                m.all.return_value = []  # compact: no expired
            elif call_n[0] == 5:
                m.all.return_value = [stale_rec]  # high: stale CrawledPage
            elif call_n[0] == 6:
                m.all.return_value = []  # high: stale CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        # 95% → above high threshold (0.90)
        high_usage = int(0.95 * 10.0 * 1024**3)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch(
                "src.tools.tool_definitions._enforce_source_quotas",
                return_value={"quota_evicted": 0, "sources_over_quota": []},
            ),
            patch(
                "src.tools.tool_definitions._enforce_table_budgets",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=high_usage),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
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
                m.first.return_value = None  # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] in (3, 4):
                m.all.return_value = []  # compact: no expired
            elif call_n[0] in (5, 6):
                m.all.return_value = []  # high: no stale
            elif call_n[0] == 7:
                m.all.return_value = [low_val_rec]  # coverage map: active CrawledPage
            elif call_n[0] == 8:
                m.all.return_value = []  # coverage map: active CodeExample
            elif call_n[0] == 9:
                m.all.return_value = [low_val_rec]  # hard: low-value CrawledPage
            elif call_n[0] == 10:
                m.all.return_value = []  # hard: low-value CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        # 105% → above hard threshold (1.00)
        over = int(1.05 * 10.0 * 1024**3)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=over),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "critical"
        assert mock_ts.call_count >= 1

    @pytest.mark.asyncio
    async def test_hard_pressure_value_evicts_code_examples_branch(self):
        """Covers hard-pressure code_ids tombstone path."""
        ctx = _make_ctx()
        low_val_code = _make_mock_code(uid=88, value_score=0.01, is_pinned=False)

        call_n = [0]

        def exec_se(stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None  # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] in (3, 4):
                m.all.return_value = []  # compact: no expired
            elif call_n[0] in (5, 6):
                m.all.return_value = []  # high: no stale
            elif call_n[0] == 7:
                m.all.return_value = []  # coverage map: active CrawledPage
            elif call_n[0] == 8:
                m.all.return_value = [low_val_code]  # coverage map: active CodeExample
            elif call_n[0] == 9:
                m.all.return_value = []  # hard: low-value CrawledPage
            elif call_n[0] == 10:
                m.all.return_value = [low_val_code]  # hard: low-value CodeExample
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        over = int(1.05 * 10.0 * 1024**3)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=over),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "critical"
        # Ensure the code_examples branch is exercised.
        assert any(call.args[2] == "code_examples" for call in mock_ts.mock_calls)

    @pytest.mark.asyncio
    async def test_hard_pressure_builds_candidates_and_tombstones_both_tables(self):
        ctx = _make_ctx()
        page_candidate = _make_mock_page(uid=101, source="docs.example.com", value_score=0.01)
        code_candidate = _make_mock_code(uid=102, source="docs.example.com", value_score=0.02)
        code_candidate.page_metadata = None

        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.first.return_value = None  # StoragePolicy
            elif call_n[0] == 2:
                m.all.return_value = []  # SourcePolicy
            elif call_n[0] in (3, 4):
                m.all.return_value = []  # compact queries
            elif call_n[0] in (5, 6):
                m.all.return_value = []  # high-pressure stale queries
            elif call_n[0] == 7:
                m.all.return_value = [page_candidate]  # hard candidates: pages
            elif call_n[0] == 8:
                m.all.return_value = [code_candidate]  # hard candidates: code
            else:
                m.all.return_value = []
            return m

        session = MagicMock()
        session.exec.side_effect = exec_se

        over = int(1.05 * 10.0 * 1024**3)
        selected = [
            {"table": "crawled_pages", "id": 101},
            {"table": "code_examples", "id": 102},
        ]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=over),
            patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch(
                "src.tools.tool_definitions._enforce_source_quotas",
                return_value={"quota_evicted": 0, "sources_over_quota": []},
            ),
            patch(
                "src.tools.tool_definitions._enforce_table_budgets",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ),
            patch("src.tools.tool_definitions._build_active_coverage_maps", return_value=({}, {})),
            patch("src.tools.tool_definitions._apply_eviction_safeguards", return_value=selected),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
            result = await td.enforce_storage_budget(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["pressure_level"] == "critical"
        called_tables = [c.args[2] for c in mock_ts.mock_calls]
        assert "crawled_pages" in called_tables
        assert "code_examples" in called_tables

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("session failed")

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=0),
        ):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=500_000_000),
        ):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=500_000_000),
        ):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=0),
        ):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.restore_tombstoned_records(ctx, record_ids=[10], table="code_examples")

        data = json.loads(result)
        assert data["success"] is True
        assert data["restored_count"] == 1

    @pytest.mark.asyncio
    async def test_invalid_table_returns_error(self):
        ctx = _make_ctx()
        result = await td.restore_tombstoned_records(ctx, record_ids=[1], table="bad_table")
        data = json.loads(result)
        assert data["success"] is False
        assert "Unknown table" in data["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_json(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("restore fail")

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.crawl_to_markdown", new_callable=AsyncMock) as mock_crawl,
        ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 0
        assert data["recrawled_count"] == 0
        mock_crawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_due_source_recrawls_successfully(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 1

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (datetime.now(timezone.utc) - timedelta(hours=5),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                return_value=json.dumps({"success": True, "pages_crawled": 1}),
            ) as mock_crawl,
        ):
            result = await td.recrawl_due_sources(ctx, max_concurrent=3)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["recrawled_count"] == 1
        mock_crawl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_due_source_skipped(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 24

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (datetime.now(timezone.utc) - timedelta(hours=2),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.crawl_to_markdown", new_callable=AsyncMock) as mock_crawl,
        ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 0
        assert data["recrawled_count"] == 0
        mock_crawl.assert_not_called()

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
        session.execute.return_value.first.return_value = (datetime.now() - timedelta(hours=3),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                return_value=json.dumps({"success": True, "pages_crawled": 1}),
            ) as mock_crawl,
        ):
            result = await td.recrawl_due_sources(ctx, source="target.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["due_count"] == 1
        assert data["recrawled_count"] == 1
        mock_crawl.assert_awaited_once()

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

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                return_value=json.dumps({"success": True, "pages_crawled": 1}),
            ),
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
        session.execute.return_value.first.return_value = (datetime.now(timezone.utc) - timedelta(hours=8),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                return_value=json.dumps({"success": False, "error": "crawl failed"}),
            ),
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "recrawl query failed" in data["error"]

    @pytest.mark.asyncio
    async def test_backoff_skips_source_until_retry_time(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 1
        policy.next_retry_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.crawl_to_markdown", new_callable=AsyncMock) as mock_crawl,
        ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["backoff_skipped_count"] == 1
        mock_crawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_dead_page_failure_tombstones_source_records(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 1
        policy.next_retry_at = None
        policy.consecutive_failures = 0
        policy.retry_backoff_base_hours = 2
        policy.max_retry_backoff_hours = 24

        page = _make_mock_page(uid=7, source="example.com")
        code = _make_mock_code(uid=8, source="example.com")

        call_n = [0]
        session = MagicMock()

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [policy]  # source policies
            elif call_n[0] == 2:
                m.all.return_value = [page]  # active pages for dead-page policy
            elif call_n[0] == 3:
                m.all.return_value = [code]  # active code rows for dead-page policy
            else:
                m.all.return_value = []
            return m

        session.exec.side_effect = exec_se
        session.execute.return_value.first.return_value = (datetime.now(timezone.utc) - timedelta(hours=8),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                return_value=json.dumps({"success": False, "error": "HTTP 404 not found"}),
            ),
            patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts,
        ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["dead_page_tombstoned"] >= 1
        assert data["failed_count"] == 1
        assert mock_ts.call_count >= 1

    @pytest.mark.asyncio
    async def test_crawl_exception_updates_backoff(self):
        ctx = _make_ctx()
        policy = MagicMock()
        policy.source = "example.com"
        policy.recrawl_interval_hours = 1
        policy.next_retry_at = None
        policy.consecutive_failures = 0
        policy.retry_backoff_base_hours = 2
        policy.max_retry_backoff_hours = 24

        session = MagicMock()
        session.exec.return_value.all.return_value = [policy]
        session.execute.return_value.first.return_value = (datetime.now(timezone.utc) - timedelta(hours=3),)

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions.crawl_to_markdown",
                new_callable=AsyncMock,
                side_effect=RuntimeError("network down"),
            ),
        ):
            result = await td.recrawl_due_sources(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["failed_count"] == 1
        assert int(policy.consecutive_failures) == 1
        assert policy.next_retry_at is not None


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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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
                m.all.return_value = []  # source policies
            elif call_n[0] == 3:
                m.all.return_value = [rec1, rec2]  # crawled pages
            else:
                m.all.return_value = [rec3]
            return m

        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            with patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 0, "code_examples": 0},
            ):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
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

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.hard_delete_tombstones(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "hard delete failed" in data["error"]


class TestPhase4StructuredExtraction:
    @pytest.mark.asyncio
    async def test_generate_schema_validates_and_caches(self, tmp_path):
        ctx = _make_ctx()
        with patch("src.tools.tool_definitions._SCHEMA_CACHE_DIR", tmp_path):
            result = await td.generate_extraction_schema(
                ctx,
                sample_html="<html><head><title>X</title></head><body><h1>H</h1><p>P</p><a href='https://x.com'>L</a></body></html>",
                strategy="css",
                cache_schema=True,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["validation"]["valid"] is True
        assert data["cached"] is True
        assert os.path.exists(data["cache_path"])
        assert len(data["schema"]["fields"]) >= 1

    @pytest.mark.asyncio
    async def test_generate_schema_requires_non_empty_html(self):
        ctx = _make_ctx()
        result = await td.generate_extraction_schema(ctx, sample_html="   ")
        data = json.loads(result)
        assert data["success"] is False

    def test_build_schema_fallback_and_xpath_mode(self):
        schema = td._build_schema_from_sample_html("<div>No known tags</div>", strategy="xpath")
        assert schema["fields"][0]["xpath"] == "//body"

    def test_validate_generated_schema_invalid_strategy_and_fields(self):
        bad_strategy = td._validate_generated_schema({"fields": []}, strategy="regex")
        assert bad_strategy["valid"] is False

        bad_fields = td._validate_generated_schema({"fields": [{}]}, strategy="css")
        assert bad_fields["valid"] is False
        assert bad_fields["errors"]

        bad_schema_type = td._validate_generated_schema("not-a-dict", strategy="css")
        assert bad_schema_type["valid"] is False

        bad_fields_shape = td._validate_generated_schema({"fields": []}, strategy="css")
        assert bad_fields_shape["valid"] is False

        bad_field_type = td._validate_generated_schema({"fields": [123]}, strategy="css")
        assert bad_field_type["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_extraction_schema_tool(self):
        ctx = _make_ctx()
        result = await td.validate_extraction_schema(
            ctx,
            schema={"fields": [{"name": "title", "selector": "h1", "type": "text"}]},
            strategy="css",
        )
        data = json.loads(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_extract_structured_json_fit_markdown_regex(self):
        ctx = _make_ctx()
        result = await td.extract_structured_json(
            ctx,
            extraction_strategy="regex",
            extraction_schema={"emails": r"[A-Z"},  # invalid regex branch -> []
            fit_markdown="Contact: user@example.com",
        )
        data = json.loads(result)
        assert data["success"] is True
        normalized = data["normalized_output"]
        assert normalized["source"]["type"] == "fit_markdown"
        assert normalized["source"]["fit_source_used"] is True

    @pytest.mark.asyncio
    async def test_extract_structured_json_fit_markdown_non_regex_rejected(self):
        ctx = _make_ctx()
        result = await td.extract_structured_json(
            ctx,
            extraction_strategy="css",
            fit_markdown="# heading",
        )
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_extract_structured_json_normalized_output_from_crawl_payload(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.crawl_to_markdown",
            new_callable=AsyncMock,
            return_value=json.dumps({"success": True, "extraction_result": [{"title": "Example"}]}),
        ) as mock_crawl:
            result = await td.extract_structured_json(
                ctx,
                url="https://example.com",
                extraction_strategy="css",
                extraction_schema={"fields": [{"name": "title", "selector": "h1", "type": "text"}]},
                fit_source=True,
            )

        data = json.loads(result)
        assert data["success"] is True
        assert data["normalized_output"]["extraction"]["record_count"] == 1
        assert data["normalized_output"]["source"]["fit_source_used"] is True
        assert mock_crawl.await_args.kwargs["content_source"] == "fit_html"

    def test_normalize_extraction_records_scalars_and_dict(self):
        assert td._normalize_extraction_records(None) == []
        assert td._normalize_extraction_records({"a": 1}) == [{"a": 1}]
        assert td._normalize_extraction_records(["x"]) == [{"value": "x"}]
        assert td._normalize_extraction_records("value") == [{"value": "value"}]

    def test_flatten_structured_content_none_and_empty_scalar(self):
        assert td._flatten_structured_content(None) == []
        assert td._flatten_structured_content("   ") == []

    def test_project_structured_content_modes(self):
        content = {"title": "X", "tags": ["a", "b"]}

        raw_text, raw_meta = td._project_structured_content(content, "raw_json")
        assert "title" in raw_text
        assert raw_meta["projection_mode"] == "raw_json"

        flat_text, flat_meta = td._project_structured_content(content, "flattened_text")
        assert "title=X" in flat_text
        assert flat_meta["projection_mode"] == "flattened_text"
        assert "raw_json" in flat_meta

        hybrid_text, hybrid_meta = td._project_structured_content(content, "not_a_mode")
        assert hybrid_meta["projection_mode"] == "hybrid"
        assert hybrid_meta["indexing_model"] == "hybrid_json_vector_v1"
        assert hybrid_text

    @pytest.mark.asyncio
    async def test_index_structured_content_forwards_projection_metadata(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.index_markdown",
            new_callable=AsyncMock,
            return_value=json.dumps({"success": True, "chunks_stored": 1}),
        ) as mock_index:
            result = await td.index_structured_content(
                ctx,
                url="https://example.com/structured",
                structured_content={"k": "v"},
                projection_mode="hybrid",
            )

        data = json.loads(result)
        assert data["success"] is True
        metadata = mock_index.await_args.kwargs["metadata"]
        assert metadata["content_class"] == "structured"
        assert metadata["projection_mode"] == "hybrid"
        assert metadata["indexing_model"] == "hybrid_json_vector_v1"

    @pytest.mark.asyncio
    async def test_extract_structured_json_returns_upstream_failure_payload(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.crawl_to_markdown",
            new_callable=AsyncMock,
            return_value=json.dumps({"success": False, "error": "upstream crawl failed"}),
        ):
            result = await td.extract_structured_json(
                ctx,
                url="https://example.com",
                extraction_strategy="css",
            )
        data = json.loads(result)
        assert data["success"] is False
        assert "upstream crawl failed" in data["error"]

    @pytest.mark.asyncio
    async def test_generate_schema_invalid_strategy_path(self):
        ctx = _make_ctx()
        result = await td.generate_extraction_schema(
            ctx,
            sample_html="<html><body><h1>x</h1></body></html>",
            strategy="regex",
        )
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_cache_schema_name_without_extension_gets_json_suffix(self, tmp_path):
        with patch("src.tools.tool_definitions._SCHEMA_CACHE_DIR", tmp_path):
            cache_path = td._cache_generated_schema(
                schema={"fields": [{"name": "title", "selector": "h1", "type": "text"}]},
                sample_html="<h1>x</h1>",
                strategy="css",
                schema_name="my_schema",
            )
        assert cache_path.endswith(".json")


class TestPhase9FreshnessAndSafeguards:
    def test_extract_source_change_id_from_headers(self):
        result = MagicMock()
        result.response_headers = {"ETag": "abc123"}
        assert td._extract_source_change_id(result) == "etag:abc123"

        result.response_headers = {"Last-Modified": "Sat, 01 Jan 2026 00:00:00 GMT"}
        assert td._extract_source_change_id(result).startswith("last-modified:")

    def test_extract_source_change_id_fallback_and_none(self):
        result = MagicMock()
        result.response_headers = None
        result.headers = {"ETag": "xyz"}
        assert td._extract_source_change_id(result) == "etag:xyz"

        result.headers = None
        assert td._extract_source_change_id(result) is None

        result.response_headers = {"X-Custom": "value"}
        assert td._extract_source_change_id(result) is None

    def test_canonical_url_key_drops_fragment(self):
        key = td._canonical_url_key("https://example.com/docs/page?a=1#section-2")
        assert key == "https://example.com/docs/page?a=1"

    def test_canonical_url_key_handles_blank_and_relative(self):
        assert td._canonical_url_key("   ") == ""
        assert td._canonical_url_key("/docs/page#frag") == "/docs/page"

    def test_is_result_fresh_as_of_without_fresh_only(self):
        metadata = {
            "staleness_score": 0.99,
            "expires_at": "2000-01-01T00:00:00+00:00",
            "crawl_timestamp": "2025-01-01T00:00:00+00:00",
        }
        # With require_fresh=False, staleness/expiry are ignored; only as_of matters.
        assert (
            td._is_result_fresh(
                metadata,
                staleness_threshold=0.1,
                as_of_dt=datetime(2026, 1, 1, tzinfo=timezone.utc),
                require_fresh=False,
            )
            is True
        )

    def test_apply_eviction_safeguards_min_active_and_last_representation(self):
        candidates = [
            {
                "id": 1,
                "source": "docs.example.com",
                "url": "https://docs.example.com/a",
                "canonical_key": "https://docs.example.com/a",
                "canonical_guard": True,
                "value_score": 0.1,
                "staleness_score": 0.9,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "id": 2,
                "source": "docs.example.com",
                "url": "https://docs.example.com/b",
                "canonical_key": "https://docs.example.com/b",
                "canonical_guard": True,
                "value_score": 0.2,
                "staleness_score": 0.8,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            },
        ]

        selected = td._apply_eviction_safeguards(
            candidates=candidates,
            source_policy_map={"docs.example.com": 1},
            source_active_counts={"docs.example.com": 2},
            canonical_active_counts={
                ("docs.example.com", "https://docs.example.com/a"): 1,
                ("docs.example.com", "https://docs.example.com/b"): 1,
            },
        )
        # Both candidates are last surviving representations, so neither can be evicted.
        assert selected == []

    def test_apply_eviction_safeguards_allows_non_last_representation(self):
        candidates = [
            {
                "id": 1,
                "source": "docs.example.com",
                "url": "https://docs.example.com/a",
                "canonical_key": "https://docs.example.com/a",
                "canonical_guard": True,
                "value_score": 0.1,
                "staleness_score": 0.9,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            }
        ]

        selected = td._apply_eviction_safeguards(
            candidates=candidates,
            source_policy_map={"docs.example.com": 0},
            source_active_counts={"docs.example.com": 4},
            canonical_active_counts={("docs.example.com", "https://docs.example.com/a"): 2},
        )
        assert [c["id"] for c in selected] == [1]

    def test_apply_eviction_safeguards_without_canonical_guard(self):
        candidates = [
            {
                "id": 1,
                "source": "docs.example.com",
                "url": "https://docs.example.com/a",
                "canonical_key": "https://docs.example.com/a",
                "canonical_guard": False,
                "value_score": 0.1,
                "staleness_score": 0.9,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        selected = td._apply_eviction_safeguards(
            candidates=candidates,
            source_policy_map={"docs.example.com": 0},
            source_active_counts={"docs.example.com": 1},
            canonical_active_counts={("docs.example.com", "https://docs.example.com/a"): 1},
        )
        assert len(selected) == 1

    def test_apply_min_active_docs_safeguard_returns_candidates_when_allowed(self):
        candidates = [
            {
                "source": "docs.example.com",
                "value_score": 0.1,
                "staleness_score": 0.5,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "source": "docs.example.com",
                "value_score": 0.2,
                "staleness_score": 0.4,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            },
        ]
        out = td._apply_min_active_docs_safeguard(candidates, {"docs.example.com": 1})
        assert len(out) == 1

    def test_apply_min_active_docs_safeguard_empty_candidates(self):
        assert td._apply_min_active_docs_safeguard([], {"docs.example.com": 1}) == []

    def test_apply_eviction_safeguards_blocks_on_min_docs(self):
        candidates = [
            {
                "id": 1,
                "source": "docs.example.com",
                "url": "https://docs.example.com/a",
                "canonical_key": "https://docs.example.com/a",
                "canonical_guard": False,
                "value_score": 0.1,
                "staleness_score": 0.9,
                "hit_count": 0,
                "last_seen_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        out = td._apply_eviction_safeguards(
            candidates,
            source_policy_map={"docs.example.com": 1},
            source_active_counts={"docs.example.com": 1},
            canonical_active_counts={("docs.example.com", "https://docs.example.com/a"): 1},
        )
        assert out == []

    def test_apply_eviction_safeguards_empty_candidates(self):
        assert td._apply_eviction_safeguards([], {}, {}, {}) == []

    def test_build_active_coverage_maps(self):
        page = MagicMock()
        page.url = "https://docs.example.com/a#frag"
        page.page_metadata = {"source": "docs.example.com", "canonical_url": "https://docs.example.com/a"}

        code = MagicMock()
        code.url = "https://docs.example.com/b"
        code.page_metadata = None
        code.ex_metadata = {"source": "docs.example.com"}

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page] if call_n[0] == 1 else [code]
            return m

        session.exec.side_effect = exec_se
        source_counts, canonical_counts = td._build_active_coverage_maps(session)
        assert source_counts["docs.example.com"] == 2
        assert canonical_counts[("docs.example.com", "https://docs.example.com/a")] == 1
        assert canonical_counts[("docs.example.com", "https://docs.example.com/b")] == 1

    def test_dead_page_error_detector(self):
        assert td._is_dead_page_error("HTTP 404 not found") is True
        assert td._is_dead_page_error("Resource Gone (410)") is True
        assert td._is_dead_page_error("timeout") is False
        assert td._is_dead_page_error(404) is False

    def test_compute_retry_backoff_hours(self):
        policy = MagicMock()
        policy.retry_backoff_base_hours = 2
        policy.max_retry_backoff_hours = 16
        policy.consecutive_failures = 1
        assert td._compute_retry_backoff_hours(policy) == 2
        policy.consecutive_failures = 3
        assert td._compute_retry_backoff_hours(policy) == 8
        policy.consecutive_failures = 6
        assert td._compute_retry_backoff_hours(policy) == 16
        policy.consecutive_failures = 0
        assert td._compute_retry_backoff_hours(policy) == 0

    def test_estimate_record_size_bytes_non_string_content(self):
        record = MagicMock()
        record.content = 1234
        record.page_metadata = {"source": "x"}
        assert td._estimate_record_size_bytes(record) > 0

    @pytest.mark.asyncio
    async def test_detect_content_drift_triggers_selective_reembed(self):
        ctx = _make_ctx()
        r1 = _make_mock_page(uid=1, staleness_score=0.95, hit_count=3)
        session = MagicMock()
        session.exec.return_value.all.return_value = [r1]

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch(
                "src.tools.tool_definitions._run_selective_reembed", new_callable=AsyncMock, return_value=1
            ) as mock_reembed,
        ):
            result = await td.detect_content_drift(ctx, trigger_selective_reembed=True)

        data = json.loads(result)
        assert data["success"] is True
        assert data["selective_reembed_executed_count"] == 1
        mock_reembed.assert_awaited_once()

    def test_enforce_source_quotas_tombstones_records(self):
        policy = MagicMock()
        policy.max_source_size_mb = 1
        # Build records exceeding quota with same source metadata
        page = _make_mock_page(uid=1, source="docs.example.com", value_score=0.1)
        page.content = "x" * (2 * 1024 * 1024)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [page]  # CrawledPage active
            else:
                m.all.return_value = []  # CodeExample active
            return m

        session.exec.side_effect = exec_se
        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._enforce_source_quotas(session, {"docs.example.com": policy})

        assert result["quota_evicted"] == 1
        mock_ts.assert_called_once()

    def test_enforce_source_quotas_no_policies(self):
        session = MagicMock()
        result = td._enforce_source_quotas(session, {})
        assert result == {"quota_evicted": 0, "sources_over_quota": []}

    def test_enforce_source_quotas_skips_invalid_or_under_limit(self):
        page = _make_mock_page(uid=1, source="docs.example.com", value_score=0.1)
        page.content = "tiny"
        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page] if call_n[0] == 1 else []
            return m

        session.exec.side_effect = exec_se
        invalid_policy = MagicMock()
        invalid_policy.max_source_size_mb = 0
        under_policy = MagicMock()
        under_policy.max_source_size_mb = 100

        result_invalid = td._enforce_source_quotas(session, {"docs.example.com": invalid_policy})
        assert result_invalid["quota_evicted"] == 0

        # reset side effect for second call
        call_n[0] = 0
        result_under = td._enforce_source_quotas(session, {"docs.example.com": under_policy})
        assert result_under["quota_evicted"] == 0

    def test_enforce_source_quotas_covers_skip_break_and_code_branch(self):
        other = _make_mock_page(uid=1, source="other.example", value_score=0.1)
        other.content = "x" * (100 * 1024)
        page_a = _make_mock_page(uid=2, source="docs.example.com", value_score=0.05)
        page_a.content = "x" * (100 * 1024)
        page_b = _make_mock_page(uid=3, source="docs.example.com", value_score=0.06)
        page_b.content = "x" * (100 * 1024)
        code = _make_mock_code(uid=4, source="docs.example.com", value_score=0.001)
        code.content = "y" * (1300 * 1024)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [other, page_a, page_b]
            else:
                m.all.return_value = [code]
            return m

        session.exec.side_effect = exec_se
        policy = MagicMock()
        policy.max_source_size_mb = 1

        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._enforce_source_quotas(session, {"docs.example.com": policy})

        assert result["quota_evicted"] >= 1
        called_tables = [c.args[2] for c in mock_ts.mock_calls]
        assert "code_examples" in called_tables

    def test_enforce_table_budgets_tombstones_rows(self):
        page = _make_mock_page(uid=1, source="docs.example.com", value_score=0.1)
        page.content = "x" * (2 * 1024 * 1024)
        code = _make_mock_code(uid=2, source="docs.example.com", value_score=0.1)
        code.content = "y" * (2 * 1024 * 1024)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [page]
            else:
                m.all.return_value = [code]
            return m

        session.exec.side_effect = exec_se
        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._enforce_table_budgets(
                session,
                max_crawled_pages_mb=1,
                max_code_examples_mb=1,
            )

        assert result["crawled_pages"] == 1
        assert result["code_examples"] == 1
        assert mock_ts.call_count == 2

    def test_enforce_table_budgets_skips_when_under_limit(self):
        page = _make_mock_page(uid=1, source="docs.example.com", value_score=0.1)
        page.content = "tiny"
        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page] if call_n[0] == 1 else []
            return m

        session.exec.side_effect = exec_se
        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._enforce_table_budgets(session, max_crawled_pages_mb=100, max_code_examples_mb=100)

        assert result["crawled_pages"] == 0
        assert result["code_examples"] == 0
        mock_ts.assert_not_called()

    def test_enforce_table_budgets_hits_break_path(self):
        page_a = _make_mock_page(uid=11, source="docs.example.com", value_score=0.1)
        page_a.content = "x" * (700 * 1024)
        page_b = _make_mock_page(uid=12, source="docs.example.com", value_score=0.2)
        page_b.content = "x" * (700 * 1024)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page_a, page_b] if call_n[0] == 1 else []
            return m

        session.exec.side_effect = exec_se
        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._enforce_table_budgets(session, max_crawled_pages_mb=1, max_code_examples_mb=None)

        assert result["crawled_pages"] == 1
        mock_ts.assert_called_once()

    def test_apply_hard_ttl_delete_deletes_expired_rows(self):
        old_ts = datetime.now(timezone.utc) - timedelta(days=400)
        page = _make_mock_page(uid=1, source="docs.example.com")
        page.last_seen_at = old_ts
        code = _make_mock_code(uid=2, source="docs.example.com")
        code.last_seen_at = old_ts

        policy = MagicMock()
        policy.ttl_days = 90

        session = MagicMock()
        deleted = []
        session.delete.side_effect = lambda r: deleted.append(r)

        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page] if call_n[0] == 1 else [code]
            return m

        session.exec.side_effect = exec_se

        result = td._apply_hard_ttl_delete(session, {"docs.example.com": policy})
        assert result["crawled_pages"] == 1
        assert result["code_examples"] == 1
        assert len(deleted) == 2

    def test_apply_hard_ttl_delete_skips_non_positive_ttl(self):
        page = _make_mock_page(uid=1, source="docs.example.com")
        page.last_seen_at = datetime.now(timezone.utc) - timedelta(days=365)
        session = MagicMock()

        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            m.all.return_value = [page] if call_n[0] == 1 else []
            return m

        session.exec.side_effect = exec_se
        policy = MagicMock()
        policy.ttl_days = 0

        result = td._apply_hard_ttl_delete(session, {"docs.example.com": policy})
        assert result["crawled_pages"] == 0

    def test_retire_source_duplicates_and_superseded(self):
        survivor = _make_mock_page(uid=1, source="docs.example.com", value_score=0.9)
        survivor.url = "https://docs.example.com/same"
        survivor.content_hash = "hash-a"
        survivor.last_crawled_at = datetime.now(timezone.utc)
        duplicate = _make_mock_page(uid=2, source="docs.example.com", value_score=0.2)
        duplicate.url = "https://docs.example.com/same"
        duplicate.content_hash = "hash-a"
        duplicate.last_crawled_at = datetime.now(timezone.utc) - timedelta(hours=2)
        superseded = _make_mock_page(uid=3, source="docs.example.com", value_score=0.1)
        superseded.url = "https://docs.example.com/same"
        superseded.content_hash = "hash-b"
        superseded.last_crawled_at = datetime.now(timezone.utc) - timedelta(hours=3)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = [survivor, duplicate, superseded]
            else:
                m.all.return_value = []
            return m

        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._retire_source_duplicates_and_superseded(session, "docs.example.com")

        assert result["duplicate_retired"] == 1
        assert result["superseded_retired"] == 1
        assert mock_ts.call_count >= 2

    def test_retire_source_duplicates_and_superseded_code_examples(self):
        survivor = _make_mock_code(uid=10, source="docs.example.com", value_score=0.9)
        survivor.url = "https://docs.example.com/same"
        survivor.content_hash = "hash-a"
        survivor.last_crawled_at = datetime.now(timezone.utc)

        duplicate = _make_mock_code(uid=11, source="docs.example.com", value_score=0.2)
        duplicate.url = "https://docs.example.com/same"
        duplicate.content_hash = "hash-a"
        duplicate.last_crawled_at = datetime.now(timezone.utc) - timedelta(hours=2)

        superseded = _make_mock_code(uid=12, source="docs.example.com", value_score=0.1)
        superseded.url = "https://docs.example.com/same"
        superseded.content_hash = "hash-b"
        superseded.last_crawled_at = datetime.now(timezone.utc) - timedelta(hours=3)

        session = MagicMock()
        call_n = [0]

        def exec_se(_stmt):
            call_n[0] += 1
            m = MagicMock()
            if call_n[0] == 1:
                m.all.return_value = []
            else:
                m.all.return_value = [survivor, duplicate, superseded]
            return m

        session.exec.side_effect = exec_se

        with patch("src.tools.tool_definitions.tombstone_records", return_value=1) as mock_ts:
            result = td._retire_source_duplicates_and_superseded(session, "docs.example.com")

        assert result["duplicate_retired"] == 1
        assert result["superseded_retired"] == 1
        called_tables = [c.args[2] for c in mock_ts.mock_calls]
        assert "code_examples" in called_tables

    @pytest.mark.asyncio
    async def test_run_selective_reembed_updates_rows(self):
        row = _make_mock_page(uid=1, source="docs.example.com", hit_count=2)
        row.content = "reembed me"
        session = MagicMock()
        session.exec.return_value.all.return_value = [row]

        with patch("src.tools.tool_definitions.create_embedding", new_callable=AsyncMock, return_value=[0.1, 0.2]):
            updated = await td._run_selective_reembed(session, [1])

        assert updated == 1
        assert row.staleness_score == 0.0
        session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_run_selective_reembed_handles_empty_ids_and_non_text(self):
        session = MagicMock()
        assert await td._run_selective_reembed(session, []) == 0

        row = _make_mock_page(uid=1)
        row.content = "   "
        session.exec.return_value.all.return_value = [row]
        updated = await td._run_selective_reembed(session, [1])
        assert updated == 0


class TestNewBudgetActionReporting:
    @pytest.mark.asyncio
    async def test_enforce_budget_reports_new_action_lines(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        session.exec.return_value.first.return_value = None

        with (
            patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)),
            patch("src.tools.tool_definitions.get_db_size_bytes", return_value=int(0.95 * 10.0 * 1024**3)),
            patch(
                "src.tools.tool_definitions._apply_hard_ttl_delete",
                return_value={"crawled_pages": 1, "code_examples": 0},
            ),
            patch(
                "src.tools.tool_definitions._enforce_source_quotas",
                return_value={"quota_evicted": 2, "sources_over_quota": ["example.com"]},
            ),
            patch(
                "src.tools.tool_definitions._enforce_table_budgets",
                return_value={"crawled_pages": 1, "code_examples": 1},
            ),
            patch("src.tools.tool_definitions.tombstone_records", return_value=0),
        ):
            result = await td.enforce_storage_budget(ctx, force=True)

        data = json.loads(result)
        assert data["success"] is True
        joined = " ".join(data["actions_taken"])
        assert "hard-ttl deleted" in joined
        assert "source-quota tombstoned" in joined
        assert "table-budget tombstoned" in joined


class TestRunIdHelpers:
    def test_normalize_run_id(self):
        assert td._normalize_run_id(None) is None
        assert td._normalize_run_id(123) is None
        assert td._normalize_run_id("   ") is None
        assert td._normalize_run_id(" run-123 ") == "run-123"

    def test_generate_run_id_prefix(self):
        run_id = td._generate_run_id("test")
        assert run_id.startswith("test-")
        assert len(run_id) > len("test-")


class TestRetrievalScopeWrappers:
    @pytest.mark.asyncio
    async def test_search_raw_markdown_forwards_variant(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.search_documents_v2", new_callable=AsyncMock, return_value='{"success": true}'
        ) as mock_search:
            result = await td.search_raw_markdown(ctx, query="q", source="docs.example.com")
        data = json.loads(result)
        assert data["success"] is True
        assert mock_search.await_args.kwargs["markdown_variant"] == "raw_markdown"

    @pytest.mark.asyncio
    async def test_search_fit_markdown_forwards_variant(self):
        ctx = _make_ctx()
        with patch(
            "src.tools.tool_definitions.search_documents_v2", new_callable=AsyncMock, return_value='{"success": true}'
        ) as mock_search:
            result = await td.search_fit_markdown(ctx, query="q", source="docs.example.com")
        data = json.loads(result)
        assert data["success"] is True
        assert mock_search.await_args.kwargs["markdown_variant"] == "fit_markdown"


class TestDetectContentDrift:
    @pytest.mark.asyncio
    async def test_detect_content_drift_classifies_rows(self):
        ctx = _make_ctx()

        r1 = _make_mock_page(uid=1, staleness_score=0.2, hit_count=0)
        r2 = _make_mock_page(uid=2, staleness_score=0.7, hit_count=0)
        r3 = _make_mock_page(uid=3, staleness_score=0.9, hit_count=5)

        session = MagicMock()
        session.exec.return_value.all.return_value = [r1, r2, r3]

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.detect_content_drift(ctx)

        data = json.loads(result)
        assert data["success"] is True
        assert data["counts"]["total"] == 3
        assert data["counts"]["unchanged"] == 1
        assert data["counts"]["minor_update"] == 1
        assert data["counts"]["major_rewrite"] == 1
        assert data["selective_reembed_candidate_count"] == 1

    @pytest.mark.asyncio
    async def test_detect_content_drift_source_filter(self):
        ctx = _make_ctx()
        include = _make_mock_page(uid=1, staleness_score=0.3, source="a.com")
        exclude = _make_mock_page(uid=2, staleness_score=0.9, source="b.com")

        session = MagicMock()
        session.exec.return_value.all.return_value = [include, exclude]

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.detect_content_drift(ctx, source="a.com")

        data = json.loads(result)
        assert data["success"] is True
        assert data["counts"]["total"] == 1

    @pytest.mark.asyncio
    async def test_detect_content_drift_exception(self):
        ctx = _make_ctx()
        session = MagicMock()
        session.exec.side_effect = RuntimeError("drift fail")

        with patch("src.tools.tool_definitions.get_session", side_effect=_make_get_session(session)):
            result = await td.detect_content_drift(ctx)

        data = json.loads(result)
        assert data["success"] is False
        assert "drift fail" in data["error"]


# ---------------------------------------------------------------------------
# Tests: guard clauses and edge cases for remaining coverage gaps
# ---------------------------------------------------------------------------


class TestNormalizedCacheMode:
    """L262: returns CacheMode instance unchanged."""

    def test_cache_mode_instance_returned_as_is(self):
        from crawl4ai import CacheMode

        mode = CacheMode.ENABLED
        assert td._normalized_cache_mode(mode) is mode


class TestNormalizeListRecordItem:
    """L598: returns None when item is None."""

    def test_none_item_returns_none(self):
        assert td._normalize_list_record_item(None) is None


class TestNormalizedNonEmptyString:
    """L1033: returns None when value is not a string."""

    def test_non_string_returns_none(self):
        assert td._normalized_non_empty_string(42) is None


class TestNormalizedHeaderMap:
    """L1054: skips entries with None key or None value."""

    def test_none_key_skipped(self):
        result = td._normalized_header_map({None: "value", "key": "v2"})
        assert None not in result
        assert result.get("key") == "v2"

    def test_none_value_skipped(self):
        result = td._normalized_header_map({"key": None})
        assert result == {}


class TestParseDatetimeString:
    """L1087: returns None when stripped value is empty."""

    def test_whitespace_only_returns_none(self):
        assert td._parse_datetime_string("   ") is None


class TestPassesAsOfFilter:
    """L1139: returns True when crawl_ts cannot be parsed."""

    def test_no_crawl_timestamp_returns_true(self):
        as_of = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert td._passes_as_of_filter({}, as_of) is True


class TestStalenessIdScores:
    """L1393: skips rows whose id is not an int."""

    def test_non_int_id_row_is_skipped(self):
        row = MagicMock()
        row.id = "not_an_int"
        result = td._row_id_scores([row])
        assert result == []


class TestSourceQuotaCandidate:
    """L1405: returns None when row_id is not an int."""

    def test_non_int_id_returns_none(self):
        row = MagicMock()
        row.id = "bad"
        assert td._source_quota_candidate(row, "crawled_pages") is None


class TestAppendSourceQuotaCandidate:
    """L1460: returns early when candidate is None."""

    def test_non_int_id_no_append(self):
        row = MagicMock()
        row.id = "bad"
        row.page_metadata = {"source": "src1"}
        rows_by_source: dict = {}
        td._append_source_quota_candidate(rows_by_source, {"src1": MagicMock()}, row, "crawled_pages")
        assert rows_by_source == {}


class TestTableBudgetCandidates:
    """L1533: skips rows whose id is not an int."""

    def test_non_int_id_row_is_skipped(self):
        row = MagicMock()
        row.id = "bad"
        result = td._table_budget_candidates([row])
        assert result == []


class TestTombstoneTableBudgetRows:
    """L1581: returns 0 immediately for empty row_ids."""

    def test_empty_row_ids_returns_zero(self):
        session = MagicMock()
        result = td._tombstone_table_budget_rows(session, [], "crawled_pages")
        assert result == 0
        session.exec.assert_not_called()


class TestTtlDaysForSource:
    """L1600: returns 90 when source not in policies."""

    def test_missing_source_returns_default_ttl(self):
        assert td._ttl_days_for_source({}, "unknown_source") == 90


class TestClassifyDuplicateAndSupersededIds:
    """L1700: returns empty lists for empty input."""

    def test_empty_list_returns_empty_tuples(self):
        dupes, superseded = td._classify_duplicate_and_superseded_ids([])
        assert dupes == []
        assert superseded == []


class TestAccumulateDuplicateOrSuperseded:
    """L1718: returns early when classification is None."""

    def test_non_int_id_no_append(self):
        row = MagicMock()
        row.id = "bad"
        duplicate_ids: list = []
        superseded_ids: list = []
        td._accumulate_duplicate_or_superseded(row, set(), duplicate_ids, superseded_ids)
        assert duplicate_ids == []
        assert superseded_ids == []


class TestDuplicateClassification:
    """L1735: returns None when row_id is not an int."""

    def test_non_int_id_returns_none(self):
        row = MagicMock()
        row.id = "not_an_int"
        assert td._duplicate_classification(row, set()) is None


class TestResultToDocEntry:
    """L2013: returns None when selected markdown is empty."""

    def test_empty_selected_variant_returns_none(self):
        result = MagicMock()
        result.success = True
        result.markdown = MagicMock()
        with patch.object(td, "_result_markdown_variants", return_value={"raw_markdown": ""}):
            with patch.object(td, "_selected_variant_markdown", return_value=""):
                doc = td._result_to_doc_entry(result, "raw_markdown", "https://x.com", False)
        assert doc is None


class TestBatchResultHasContent:
    """L2340: _batch_result_has_content directly exercised."""

    def test_success_with_markdown_returns_true(self):
        result = MagicMock()
        result.success = True
        result.markdown = "# Content"
        assert td._batch_result_has_content(result) is True

    def test_failed_result_returns_false(self):
        result = MagicMock()
        result.success = False
        result.markdown = "# Content"
        assert td._batch_result_has_content(result) is False


class TestAppliedContentFilter:
    """L2427: returns None for unknown filter string."""

    def test_unknown_filter_returns_none(self):
        assert td._applied_content_filter("totally_unknown_filter_9999") is None


class TestSelectedRowVariant:
    """L3242: returns None for empty rows list."""

    def test_empty_rows_returns_none(self):
        assert td._selected_row_variant([]) is None


class TestAppendPaginatedUrl:
    """L4535: returns early for non-string candidate."""

    def test_non_string_candidate_not_appended(self):
        urls: list = []
        td._append_paginated_url(urls, 42)
        assert urls == []


class TestStoreIndexMarkdownPayload:
    """L5372: returns (chunks_stored, None) when chunks_stored <= 0."""

    @pytest.mark.asyncio
    async def test_zero_chunks_returns_none_id(self):
        session = MagicMock()
        payload = {
            "db_urls": [],
            "db_contents": [],
            "db_metas": [],
            "db_chunks": [],
            "db_fulldocs": [],
        }
        with patch("src.tools.tool_definitions.add_documents_to_db", new=AsyncMock(return_value=0)):
            stored, chunk_id = await td._store_index_markdown_payload(session, "https://x.com", payload)
        assert stored == 0
        assert chunk_id is None


class TestFirstChunkId:
    """L5445: returns None when first_row is None."""

    def test_no_rows_returns_none(self):
        session = MagicMock()
        session.exec.return_value.first.return_value = None
        result = td._first_chunk_id(session, "https://x.com")
        assert result is None


class TestMaybeFilterFreshResults:
    """L5479: returns results unchanged when no filter is needed."""

    def test_no_filter_returns_results_unchanged(self):
        results = [{"id": 1}, {"id": 2}]
        out = td._maybe_filter_fresh_results(results, 0.5, None, False)
        assert out is results


class TestCandidateFromRecord:
    """L5986: returns None when rec_id is not an int."""

    def test_non_int_id_returns_none(self):
        rec = MagicMock()
        rec.id = "bad"
        rec.page_metadata = {}
        assert td._candidate_from_record("crawled_pages", rec) is None


class TestTableEvictionCandidates:
    """L6042: skips records where _candidate_from_record returns None."""

    def test_bad_id_record_skipped(self):
        session = MagicMock()
        bad_record = MagicMock()
        bad_record.id = "bad"
        bad_record.page_metadata = {}

        with patch.object(td, "_eviction_records_for_table", return_value=[bad_record]):
            result = td._table_eviction_candidates(session, MagicMock(), "crawled_pages", 10, None)
        assert result == []


class TestPressureLevel:
    """L6116: returns 'warning' for warn_pct <= ratio < high_pct."""

    def test_warning_level(self):
        assert td._pressure_level(0.85, 0.80, 0.90, 0.95) == "warning"

    def test_ok_level(self):
        assert td._pressure_level(0.50, 0.80, 0.90, 0.95) == "ok"


class TestRowsToEvictionCandidates:
    """L6261: skips rows whose id is not an int."""

    def test_non_int_id_row_skipped(self):
        row = MagicMock()
        row.id = "bad"
        row.page_metadata = {}
        result = td._rows_to_eviction_candidates([row], "crawled_pages")
        assert result == []


class TestEnforceHardPressureEviction:
    """L6304: returns 0 when db_size is already at or below target."""

    def test_db_size_below_target_returns_zero(self):
        session = MagicMock()
        evicted = td._enforce_hard_pressure_eviction(session, 100, 200, 0.9, {})
        assert evicted == 0


class TestRecordRecrawlFailureSuccess:
    """L6710, L6718: early return when policy is None."""

    def test_failure_with_none_policy_is_noop(self):
        now = datetime.now(tz=timezone.utc)
        td._record_recrawl_failure(None, now)  # should not raise

    def test_success_with_none_policy_is_noop(self):
        td._record_recrawl_success(None)  # should not raise


class TestSourceRecordIds:
    """L6734, L6736: continue when id is not int or source doesn't match."""

    def test_non_int_id_row_skipped(self):
        session = MagicMock()
        row = MagicMock()
        row.id = "bad"
        row.page_metadata = {"source": "src1"}
        session.exec.return_value.all.return_value = [row]
        with patch("src.tools.tool_definitions.select"):
            result = td._source_record_ids(session, MagicMock(), "src1")
        assert result == []

    def test_wrong_source_row_skipped(self):
        session = MagicMock()
        row = MagicMock()
        row.id = 42
        row.page_metadata = {"source": "other_source"}
        session.exec.return_value.all.return_value = [row]
        with patch("src.tools.tool_definitions.select"):
            result = td._source_record_ids(session, MagicMock(), "src1")
        assert result == []


class TestEligibleReembedRowId:
    """L7048, L7052: guard clauses in _eligible_reembed_row_id."""

    def test_non_int_id_returns_none(self):
        row = MagicMock()
        row.id = "bad"
        assert td._eligible_reembed_row_id(row, {1, 2, 3}) is None

    def test_zero_hit_count_returns_none(self):
        row = MagicMock()
        row.id = 5
        row.hit_count = 0
        assert td._eligible_reembed_row_id(row, {5}) is None
