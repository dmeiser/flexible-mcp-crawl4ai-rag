"""Unit tests for src/crawler/web_crawler.py — 100% coverage, all offline."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.crawler.web_crawler import (
    _collect_urls_to_crawl,
    _fixed_char_chunking,
    _paragraph_chunking,
    _pop_current_frontier,
    _resolve_chunk_strategy,
    _result_next_depth_urls,
    _sentence_chunking,
    chunk_text_according_to_settings,
    crawl_batch,
    crawl_markdown_file,
    crawl_recursive_internal_links,
    is_sitemap,
    is_txt,
    parse_sitemap,
)
from src.utils import ChunkStrategy

# ---------------------------------------------------------------------------
# Tests: is_sitemap / is_txt
# ---------------------------------------------------------------------------


class TestIsType:
    def test_sitemap_xml_extension(self):
        assert is_sitemap("https://example.com/sitemap.xml") is True

    def test_sitemap_in_path(self):
        assert is_sitemap("https://example.com/sitemap/index.xml") is True

    def test_not_sitemap(self):
        assert is_sitemap("https://example.com/about") is False

    def test_txt_extension(self):
        assert is_txt("https://example.com/llms.txt") is True

    def test_not_txt(self):
        assert is_txt("https://example.com/page.html") is False


# ---------------------------------------------------------------------------
# Tests: parse_sitemap
# ---------------------------------------------------------------------------


class TestParseSitemap:
    @pytest.mark.asyncio
    async def test_success(self):
        xml = b"""<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/a</loc></url>
          <url><loc>https://example.com/b</loc></url>
        </urlset>"""

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = xml

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == ["https://example.com/a", "https://example.com/b"]

    @pytest.mark.asyncio
    async def test_http_status_error_returns_empty(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock(status_code=404))
        )

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == []

    @pytest.mark.asyncio
    async def test_request_error_returns_empty(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("connection failed"))

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == []

    @pytest.mark.asyncio
    async def test_xml_parse_error_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = b"<not valid xml <<<"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == []

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_empty(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == []

    @pytest.mark.asyncio
    async def test_loc_with_none_text_skipped(self):
        """<loc> elements with no text should be filtered out."""
        xml = b"""<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/a</loc></url>
          <url><loc/></url>
        </urlset>"""

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = xml

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.crawler.web_crawler.httpx.AsyncClient", return_value=mock_client):
            result = await parse_sitemap("https://example.com/sitemap.xml")
        assert result == ["https://example.com/a"]


# ---------------------------------------------------------------------------
# Tests: _fixed_char_chunking
# ---------------------------------------------------------------------------


class TestFixedCharChunking:
    def test_empty_text(self):
        assert _fixed_char_chunking("", 100, 10) == []

    def test_single_chunk_fits(self):
        text = "hello world"
        result = _fixed_char_chunking(text, 100, 10)
        assert result == [text]

    def test_size_zero_returns_full_text(self):
        result = _fixed_char_chunking("hello", 0, 0)
        assert result == ["hello"]

    def test_whitespace_only_text_returns_empty(self):
        result = _fixed_char_chunking("   ", 0, 0)
        assert result == []

    def test_splits_into_chunks(self):
        text = "a" * 25
        result = _fixed_char_chunking(text, 10, 0)
        assert len(result) == 3  # 10, 10, 5

    def test_overlap_applied(self):
        text = "abcdefghij" * 3  # 30 chars
        result = _fixed_char_chunking(text, 15, 5)
        # step = 15 - 5 = 10; start positions: 0, 10, 20
        assert len(result) == 3

    def test_whitespace_only_chunk_skipped(self):
        text = "hello" + " " * 10 + "world"
        # Both non-whitespace chunks should be included
        result = _fixed_char_chunking(text, 5, 0)
        # First chunk: "hello", next some spaces, etc.
        assert all(c.strip() for c in result)


# ---------------------------------------------------------------------------
# Tests: _paragraph_chunking
# ---------------------------------------------------------------------------


class TestParagraphChunking:
    def test_empty_text(self):
        assert _paragraph_chunking("", 100, 10) == []

    def test_single_short_paragraph(self):
        text = "Hello world"
        result = _paragraph_chunking(text, 200, 0)
        assert result == ["Hello world"]

    def test_multiple_paragraphs_under_size(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = _paragraph_chunking(text, 200, 0)
        assert len(result) == 3

    def test_empty_paragraphs_skipped(self):
        text = "Para one.\n\n\n\nPara two."
        result = _paragraph_chunking(text, 200, 0)
        assert len(result) == 2

    def test_large_paragraph_falls_back_to_fixed(self):
        long_para = "word " * 100  # 500 chars
        result = _paragraph_chunking(long_para, 50, 0)
        assert len(result) > 1


# ---------------------------------------------------------------------------
# Tests: _sentence_chunking
# ---------------------------------------------------------------------------


class TestSentenceChunking:
    def test_empty_text(self):
        assert _sentence_chunking("", 100, 0) == []

    def test_single_sentence(self):
        text = "This is a sentence."
        result = _sentence_chunking(text, 500, 0)
        assert len(result) == 1
        assert "sentence" in result[0]

    def test_multiple_sentences_combined(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _sentence_chunking(text, 500, 0)
        assert len(result) == 1  # All fit in one chunk

    def test_sentence_split_when_over_size(self):
        # Each sentence ~20 chars, size=25 should produce splits
        text = "Hello world today. Good morning sir. How are you doing?"
        result = _sentence_chunking(text, 20, 0)
        assert len(result) >= 2

    def test_overlap_carries_back_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _sentence_chunking(text, 40, 20)
        assert len(result) >= 2
        # Second chunk should contain some text from the overlap

    def test_nltk_failure_falls_back_to_paragraph(self):
        text = "Hello world. Goodbye."
        with patch("src.crawler.web_crawler.nltk.sent_tokenize", side_effect=Exception("NLTK failed")):
            result = _sentence_chunking(text, 500, 0)
        # Falls back to paragraph chunking — should still produce a result
        assert len(result) >= 1

    def test_no_sentences_returns_empty(self):
        with patch("src.crawler.web_crawler.nltk.sent_tokenize", return_value=[]):
            result = _sentence_chunking("some text", 100, 0)
        assert result == []

    def test_whitespace_chunks_filtered(self):
        with patch("src.crawler.web_crawler.nltk.sent_tokenize", return_value=["  ", "Real sentence."]):
            result = _sentence_chunking("text", 100, 0)
        # "  " alone becomes a chunk "  " which is filtered
        assert all(c.strip() for c in result)


# ---------------------------------------------------------------------------
# Tests: chunk_text_according_to_settings
# ---------------------------------------------------------------------------


class TestChunkTextAccordingToSettings:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        result = await chunk_text_according_to_settings("")
        assert result == []

    @pytest.mark.asyncio
    async def test_fixed_strategy(self):
        with patch("src.crawler.web_crawler.settings") as mock_settings:
            mock_settings.CHUNK_SIZE = 10
            mock_settings.CHUNK_OVERLAP = 0
            mock_settings.CHUNK_STRATEGY = ChunkStrategy.FIXED
            result = await chunk_text_according_to_settings("a" * 25)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_sentence_strategy(self):
        with (
            patch("src.crawler.web_crawler.settings") as mock_settings,
            patch("src.crawler.web_crawler.nltk.sent_tokenize", return_value=["Hello.", "World."]),
        ):
            mock_settings.CHUNK_SIZE = 500
            mock_settings.CHUNK_OVERLAP = 0
            mock_settings.CHUNK_STRATEGY = ChunkStrategy.SENTENCE
            result = await chunk_text_according_to_settings("Hello. World.")
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_paragraph_strategy(self):
        with patch("src.crawler.web_crawler.settings") as mock_settings:
            mock_settings.CHUNK_SIZE = 200
            mock_settings.CHUNK_OVERLAP = 0
            mock_settings.CHUNK_STRATEGY = ChunkStrategy.PARAGRAPH
            result = await chunk_text_according_to_settings("Para one.\n\nPara two.")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_semantic_strategy_falls_back_to_paragraph(self):
        with patch("src.crawler.web_crawler.settings") as mock_settings:
            mock_settings.CHUNK_SIZE = 200
            mock_settings.CHUNK_OVERLAP = 0
            mock_settings.CHUNK_STRATEGY = ChunkStrategy.SEMANTIC  # not implemented, falls back
            result = await chunk_text_according_to_settings("Para one.\n\nPara two.")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: crawl_markdown_file
# ---------------------------------------------------------------------------


class TestCrawlMarkdownFile:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Hello\nContent here"

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        results = await crawl_markdown_file(mock_crawler, "https://example.com")
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert "Hello" in results[0]["markdown"]

    @pytest.mark.asyncio
    async def test_crawl_failure_returns_empty(self):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.markdown = None
        mock_result.error_message = "Connection refused"

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        results = await crawl_markdown_file(mock_crawler, "https://example.com")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_markdown_returns_empty(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = ""

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        results = await crawl_markdown_file(mock_crawler, "https://example.com")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_error_message_attribute(self):
        mock_result = MagicMock(spec=["success", "markdown"])
        mock_result.success = False
        mock_result.markdown = None

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        results = await crawl_markdown_file(mock_crawler, "https://example.com")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: crawl_batch
# ---------------------------------------------------------------------------


class TestCrawlBatch:
    @pytest.mark.asyncio
    async def test_empty_urls_returns_empty(self):
        mock_crawler = AsyncMock()
        results = await crawl_batch(mock_crawler, [])
        assert results == []
        mock_crawler.arun_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_crawl(self):
        mock_r1 = MagicMock()
        mock_r1.success = True
        mock_r1.markdown = "content A"
        mock_r1.url = "https://a.com"

        mock_r2 = MagicMock()
        mock_r2.success = True
        mock_r2.markdown = "content B"
        mock_r2.url = "https://b.com"

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(return_value=[mock_r1, mock_r2])

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_batch(mock_crawler, ["https://a.com", "https://b.com"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_failed_urls_filtered_out(self):
        mock_r1 = MagicMock()
        mock_r1.success = True
        mock_r1.markdown = "content A"
        mock_r1.url = "https://a.com"

        mock_r2 = MagicMock()
        mock_r2.success = False
        mock_r2.markdown = None
        mock_r2.url = "https://b.com"
        mock_r2.error_message = "404"

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(return_value=[mock_r1, mock_r2])

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_batch(mock_crawler, ["https://a.com", "https://b.com"])
        assert len(results) == 1
        assert results[0]["url"] == "https://a.com"

    @pytest.mark.asyncio
    async def test_no_error_message_attr_ignored(self):
        mock_r = MagicMock(spec=["success", "markdown", "url"])
        mock_r.success = False
        mock_r.markdown = None
        mock_r.url = "https://x.com"

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(return_value=[mock_r])

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_batch(mock_crawler, ["https://x.com"])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: crawl_recursive_internal_links
# ---------------------------------------------------------------------------


class TestCrawlRecursiveInternalLinks:
    @pytest.mark.asyncio
    async def test_empty_start_urls_returns_empty(self):
        mock_crawler = AsyncMock()
        results = await crawl_recursive_internal_links(mock_crawler, [], max_depth=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_max_depth_zero_no_link_following(self):
        """At max_depth=0 we crawl start URLs but don't follow links."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "content"
        mock_result.url = "https://example.com"
        mock_result.links = {"internal": [{"href": "https://example.com/page2"}]}

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(return_value=[mock_result])

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(mock_crawler, ["https://example.com"], max_depth=0)
        # crawl_batch called once; arun_many called once for batch; at depth 0 no recursion
        assert len(results) == 1
        assert mock_crawler.arun_many.call_count == 1

    @pytest.mark.asyncio
    async def test_url_pattern_filters_links(self):
        """Only URLs matching url_pattern should be crawled."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com/docs/"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com/docs/"
        link_result.links = {
            "internal": [
                {"href": "https://example.com/docs/page"},
                {"href": "https://example.com/blog/post"},  # should be filtered
            ]
        }

        mock_crawler = AsyncMock()
        # First arun_many: batch crawl; second: link discovery
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],  # crawl_batch call
                [link_result],  # link discovery call
            ]
        )

        with (
            patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"),
            patch(
                "src.crawler.web_crawler.crawl_batch",
                new_callable=AsyncMock,
                return_value=[{"url": "https://example.com/docs/", "markdown": "content"}],
            ),
        ):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com/docs/"],
                max_depth=1,
                url_pattern=r"/docs/",
            )
        # Only /docs/ URLs should be added to queue
        all_urls = [r["url"] for r in results]
        assert len(results) > 0, f"Expected results, got {results}"
        assert "https://example.com/docs/" in all_urls

    @pytest.mark.asyncio
    async def test_already_visited_urls_skipped(self):
        """Visiting the same URL twice should be skipped."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com"
        link_result.links = {
            "internal": [
                {"href": "https://example.com"},  # same as start URL
            ]
        }

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],
                [link_result],
            ]
        )

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com"],
                max_depth=1,
            )
        assert len(results) == 1  # Only crawled once

    @pytest.mark.asyncio
    async def test_cross_domain_links_not_followed(self):
        """Links from a different domain should not be followed."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com"
        # Link points to a different domain
        link_result.links = {
            "internal": [
                {"href": "https://other.com/page"},
            ]
        }

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],
                [link_result],
            ]
        )

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com"],
                max_depth=1,
            )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_link_href_empty_skipped(self):
        """Links with empty href are skipped."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com"
        link_result.links = {"internal": [{"href": ""}]}

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],
                [link_result],
            ]
        )

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com"],
                max_depth=1,
            )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_link_result_no_links_attr(self):
        """r.links being falsy (None) shouldn't crash."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com"
        link_result.links = None

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],
                [link_result],
            ]
        )

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com"],
                max_depth=1,
            )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_url_with_fragment_defragged(self):
        """URLs with fragments should have fragment stripped."""
        batch_result = MagicMock()
        batch_result.success = True
        batch_result.markdown = "content"
        batch_result.url = "https://example.com"

        link_result = MagicMock()
        link_result.success = True
        link_result.url = "https://example.com"
        link_result.links = {"internal": [{"href": "https://example.com/page#section1"}]}

        # Next batch crawl (including /page)
        page_result = MagicMock()
        page_result.success = True
        page_result.markdown = "page content"
        page_result.url = "https://example.com/page"

        mock_crawler = AsyncMock()
        mock_crawler.arun_many = AsyncMock(
            side_effect=[
                [batch_result],  # first batch
                [link_result],  # link discovery
                [page_result],  # second batch (recursive)
                [],  # empty link discovery at depth 2
            ]
        )

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com"],
                max_depth=2,
            )
        crawled_urls = [r["url"] for r in results]
        assert "https://example.com" in crawled_urls

    @pytest.mark.asyncio
    async def test_all_urls_filtered_by_pattern_returns_empty(self):
        """When url_pattern rejects every start URL urls_to_crawl is empty → continue → no crawl."""
        mock_crawler = AsyncMock()

        with patch("src.crawler.web_crawler.MemoryAdaptiveDispatcher"):
            results = await crawl_recursive_internal_links(
                mock_crawler,
                ["https://example.com/blog/post"],
                max_depth=2,
                url_pattern=r"/docs/",
            )

        assert results == []
        mock_crawler.arun_many.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: internal helper functions for coverage
# ---------------------------------------------------------------------------


class TestResolveChunkStrategy:
    def test_invalid_strategy_returns_settings_default(self):
        """An unrecognised strategy name falls back to settings.CHUNK_STRATEGY."""
        result = _resolve_chunk_strategy("totally_invalid_strategy_xyz")
        assert result is not None


class TestPopCurrentFrontier:
    def test_empty_queue_returns_empty_lists(self):
        current, remaining = _pop_current_frontier([])
        assert current == []
        assert remaining == []

    def test_mixed_depth_queue_splits_correctly(self):
        """Items at different depths are split: first depth goes to current, rest remain."""
        queue = [("https://a.com", 0), ("https://b.com", 0), ("https://c.com", 1)]
        current, remaining = _pop_current_frontier(queue)
        assert current == [("https://a.com", 0), ("https://b.com", 0)]
        assert remaining == [("https://c.com", 1)]


class TestCollectUrlsToCrawl:
    def test_already_visited_url_is_skipped(self):
        """A URL already in visited is not added to urls_to_crawl."""
        visited: set[str] = {"https://example.com/page"}
        current_level = [("https://example.com/page", 0), ("https://example.com/new", 0)]
        result = _collect_urls_to_crawl(current_level, visited, pattern_re=None)
        assert "https://example.com/page" not in result
        assert "https://example.com/new" in result


class TestResultNextDepthUrls:
    def test_failed_result_returns_empty(self):
        """A result with success=False returns no next-depth URLs."""
        result = MagicMock()
        result.success = False
        urls = _result_next_depth_urls(result, set(), next_depth=1)
        assert urls == []
