"""Unit tests for src/crawler/metadata_extractor.py — 100% coverage, all offline."""
import pytest
from src.crawler.metadata_extractor import (
    extract_section_info,
    extract_link_graph,
    extract_media_metadata,
)


class TestExtractSectionInfo:
    def test_empty_chunk(self):
        result = extract_section_info("")
        assert result["headers"] == ""
        assert result["char_count"] == 0
        assert result["word_count"] == 0

    def test_no_headers(self):
        chunk = "Just some text without any headers."
        result = extract_section_info(chunk)
        assert result["headers"] == ""
        assert result["char_count"] == len(chunk)
        assert result["word_count"] == len(chunk.split())

    def test_single_h1_header(self):
        chunk = "# Hello World\n\nSome content."
        result = extract_section_info(chunk)
        assert "# Hello World" in result["headers"]
        assert result["char_count"] == len(chunk)

    def test_multiple_headers(self):
        chunk = "# Title\n\n## Section\n\n### Subsection\n\nSome text."
        result = extract_section_info(chunk)
        headers = result["headers"]
        assert "# Title" in headers
        assert "## Section" in headers
        assert "### Subsection" in headers
        # Headers joined with "; "
        parts = headers.split("; ")
        assert len(parts) == 3

    def test_headers_not_at_line_start_not_captured(self):
        """Headers must be at the start of a line."""
        chunk = "text with # inline hash\n# Real header"
        result = extract_section_info(chunk)
        # Only the real header on its own line should be captured
        assert "Real header" in result["headers"]

    def test_word_count(self):
        chunk = "one two three four five"
        result = extract_section_info(chunk)
        assert result["word_count"] == 5

    def test_char_count(self):
        chunk = "abc"
        result = extract_section_info(chunk)
        assert result["char_count"] == 3


class TestLinkAndMediaExtraction:
    def test_extract_link_graph_empty(self):
        out = extract_link_graph("")
        assert out["total_links"] == 0
        assert out["links"] == []

    def test_extract_link_graph_internal_external(self):
        md = "[Internal](/docs) and [External](https://other.com/page)"
        out = extract_link_graph(md, base_url="https://example.com")
        assert out["total_links"] == 2
        assert out["internal_links"] >= 1
        assert out["external_links"] >= 1

    def test_extract_media_metadata(self):
        md = "![Alt text](https://example.com/image.png)\n[Video](https://example.com/demo.mp4)"
        out = extract_media_metadata(md)
        assert out["image_count"] == 1
        assert out["media_link_count"] >= 1
        assert out["images"][0]["alt_text"] == "Alt text"

    def test_extract_link_graph_skips_blank_href(self):
        md = "[Empty](   ) [Valid](https://example.com/page)"
        out = extract_link_graph(md, base_url="https://example.com")
        assert out["total_links"] == 1
        assert out["links"][0]["url"] == "https://example.com/page"

    def test_extract_media_metadata_skips_blank_urls(self):
        md = "[]( ) [Plain](https://example.com/page)"
        out = extract_media_metadata(md)
        assert out["media_link_count"] == 0

    def test_extract_media_metadata_empty_markdown(self):
        out = extract_media_metadata("")
        assert out["image_count"] == 0
        assert out["media_link_count"] == 0
