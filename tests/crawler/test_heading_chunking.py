"""Tests for heading-aware chunking functions in web_crawler.py."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

import pytest
from unittest.mock import patch, AsyncMock

from src.services.web_crawler import (
    _parse_markdown_sections,
    _build_heading_path,
    heading_chunk_text,
    chunk_text_with_heading_metadata,
)
from src.config import ChunkStrategy


def test_parse_markdown_sections_no_headings():
    """Text with no headings returns single section with level=0."""
    sections = _parse_markdown_sections("Hello world")
    assert len(sections) == 1
    assert sections[0]["level"] == 0
    assert sections[0]["heading"] == ""
    assert "Hello world" in sections[0]["content"]


def test_parse_markdown_sections_with_headings():
    """Text with headings is split into sections."""
    text = "# Title\nContent1\n## Section\nContent2"
    sections = _parse_markdown_sections(text)
    assert len(sections) == 2
    assert sections[0]["level"] == 1
    assert sections[0]["heading"] == "Title"
    assert sections[1]["level"] == 2
    assert sections[1]["heading"] == "Section"


def test_parse_markdown_sections_content_captured():
    """Content between headings is captured correctly."""
    text = "# H1\nLine one.\nLine two.\n## H2\nLine three."
    sections = _parse_markdown_sections(text)
    assert "Line one." in sections[0]["content"]
    assert "Line two." in sections[0]["content"]
    assert "Line three." in sections[1]["content"]


def test_parse_markdown_sections_empty_content():
    """Section with no content between headings gets empty string."""
    text = "# H1\n## H2\nContent"
    sections = _parse_markdown_sections(text)
    assert sections[0]["content"] == ""
    assert "Content" in sections[1]["content"]


def test_parse_markdown_sections_all_heading_levels():
    """All heading levels 1–6 are parsed correctly."""
    lines = [f"{'#' * i} H{i}" for i in range(1, 7)]
    text = "\n".join(lines)
    sections = _parse_markdown_sections(text)
    assert len(sections) == 6
    for i, section in enumerate(sections, start=1):
        assert section["level"] == i
        assert section["heading"] == f"H{i}"


def test_build_heading_path():
    stack = [{"level": 1, "heading": "A", "content": ""}, {"level": 2, "heading": "B", "content": ""}]
    assert _build_heading_path(stack) == ["A", "B"]


def test_build_heading_path_empty():
    assert _build_heading_path([]) == []


def test_heading_chunk_text_no_headings():
    """Falls back to paragraph chunking with empty paths for text without headings."""
    text = "Hello world. " * 50
    results = heading_chunk_text(text, size=200, overlap=20)
    assert len(results) > 0
    for chunk, path in results:
        assert path == []


def test_heading_chunk_text_with_headings():
    """Chunks respect heading boundaries and include correct paths."""
    text = "# Main\nThis is main content.\n## Sub\nThis is sub content."
    results = heading_chunk_text(text, size=500, overlap=0)
    assert len(results) >= 1
    paths = [path for _, path in results]
    assert any("Main" in p for p in paths)


def test_heading_chunk_text_heading_hierarchy():
    """Heading path reflects correct nesting."""
    text = "# H1\nContent A\n## H2\nContent B\n### H3\nContent C"
    results = heading_chunk_text(text, size=500, overlap=0)
    h3_paths = [path for _, path in results if len(path) == 3]
    assert len(h3_paths) > 0
    assert h3_paths[0] == ["H1", "H2", "H3"]


def test_heading_chunk_text_sibling_headings():
    """Sibling headings don't nest into each other."""
    text = "# H1\nContent A\n# H2\nContent B"
    results = heading_chunk_text(text, size=500, overlap=0)
    paths = [path for _, path in results]
    assert ["H1"] in paths
    assert ["H2"] in paths


def test_heading_chunk_text_empty_sections_skipped():
    """Sections with no content produce no chunks."""
    text = "# H1\n# H2\nContent"
    results = heading_chunk_text(text, size=500, overlap=0)
    # H1 has no content so only H2 chunk
    assert len(results) == 1
    assert results[0][1] == ["H2"]


def test_heading_chunk_text_large_section_split():
    """A large section is split by paragraph chunking."""
    long_content = "Word " * 200
    text = f"# Title\n{long_content}"
    results = heading_chunk_text(text, size=100, overlap=0)
    assert len(results) > 1
    for _, path in results:
        assert path == ["Title"]


def test_heading_chunk_text_fallback_on_all_empty_sections():
    """Falls back to paragraph chunking when all sections are empty."""
    # All headings have no content
    text = "# H1\n# H2\n# H3"
    # Since all sections are empty, heading_chunk_text falls back
    results = heading_chunk_text(text, size=500, overlap=0)
    # The fallback will produce chunks from the original text
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_chunk_text_with_heading_metadata_heading_strategy():
    """Returns (chunk, metadata) pairs with heading_path when using HEADING strategy."""
    text = "# Title\nSome content here.\n## Section\nMore content."
    import src.utils as _utils

    with patch.object(_utils.settings, "CHUNK_STRATEGY", ChunkStrategy.HEADING):
        with patch.object(_utils.settings, "CHUNK_SIZE", 500):
            with patch.object(_utils.settings, "CHUNK_OVERLAP", 0):
                results = await chunk_text_with_heading_metadata(text)
    assert len(results) > 0
    for chunk, meta in results:
        assert "heading_path" in meta
        assert "heading_level" in meta
        assert isinstance(meta["heading_path"], list)


@pytest.mark.asyncio
async def test_chunk_text_with_heading_metadata_heading_strategy_value():
    """Accepts ChunkStrategy.HEADING.value (string) as strategy param."""
    text = "# Title\nContent."
    import src.utils as _utils

    with patch.object(_utils.settings, "CHUNK_SIZE", 500):
        with patch.object(_utils.settings, "CHUNK_OVERLAP", 0):
            results = await chunk_text_with_heading_metadata(text, strategy=ChunkStrategy.HEADING.value)
    assert len(results) > 0
    for _, meta in results:
        assert "heading_path" in meta


@pytest.mark.asyncio
async def test_chunk_text_with_heading_metadata_other_strategy():
    """Falls back to chunk_text_according_to_settings for non-HEADING strategies."""
    text = "Some content without headings."
    with patch(
        "src.services.web_crawler.chunk_text_according_to_settings",
        new_callable=AsyncMock,
        return_value=["chunk1", "chunk2"],
    ):
        results = await chunk_text_with_heading_metadata(text, strategy=ChunkStrategy.PARAGRAPH)
    assert results == [
        ("chunk1", {"heading_path": [], "heading_level": 0}),
        ("chunk2", {"heading_path": [], "heading_level": 0}),
    ]


@pytest.mark.asyncio
async def test_chunk_text_with_heading_metadata_none_strategy_uses_settings():
    """When strategy is None, uses settings.CHUNK_STRATEGY."""
    text = "Some content."
    import src.utils as _utils

    with patch.object(_utils.settings, "CHUNK_STRATEGY", ChunkStrategy.PARAGRAPH):
        with patch(
            "src.services.web_crawler.chunk_text_according_to_settings",
            new_callable=AsyncMock,
            return_value=["chunk"],
        ):
            results = await chunk_text_with_heading_metadata(text)
    assert results == [("chunk", {"heading_path": [], "heading_level": 0})]


@pytest.mark.asyncio
async def test_chunk_text_with_heading_metadata_heading_level_reflects_path():
    """heading_level equals len(heading_path) for each chunk."""
    text = "# H1\nContent A\n## H2\nContent B\n### H3\nContent C"
    import src.utils as _utils

    with patch.object(_utils.settings, "CHUNK_STRATEGY", ChunkStrategy.HEADING):
        with patch.object(_utils.settings, "CHUNK_SIZE", 500):
            with patch.object(_utils.settings, "CHUNK_OVERLAP", 0):
                results = await chunk_text_with_heading_metadata(text)
    for _, meta in results:
        assert meta["heading_level"] == len(meta["heading_path"])
