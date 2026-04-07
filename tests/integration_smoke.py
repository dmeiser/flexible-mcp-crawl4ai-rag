"""
Integration smoke test — must be run against a live Docker stack.

Usage:
    MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py

Flags (environment variables):
    EXPECT_NEW_TOOLS=true   Enforce crawl_url / crawl_deep / crawl_adaptive are
                            present (strict rollout check).
    EXPECT_ALL_TOOLS=true   Enforce all optional tools (including
                            search_code_examples) are present and exercise them.
    USE_AGENTIC_RAG=true    Alias for EXPECT_ALL_TOOLS regarding search_code_examples.

Structure:
    Each tool / concern has its own test function.  The coordinator
    run_smoke_tests() executes all of them, reports [PASS]/[FAIL] per
    test, and exits with code 1 if any test failed.  A failure in one test
    does NOT prevent subsequent tests from running.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import pytest
from fastmcp import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MCP_URL = os.getenv("MCP_URL", "http://localhost:8051/sse")
TEST_URL = "https://example.com"

EXPECT_NEW_TOOLS = os.getenv("EXPECT_NEW_TOOLS", "false").lower() in {"1", "true", "yes", "on"}
EXPECT_ALL_TOOLS = os.getenv("EXPECT_ALL_TOOLS", "false").lower() in {"1", "true", "yes", "on"}
USE_AGENTIC_RAG = os.getenv("USE_AGENTIC_RAG", "false").lower() in {"1", "true", "yes", "on"}

# Known-stable synthetic URLs used throughout indexing/retrieval tests.
SMOKE_BASE_URL = "https://smoke-test.internal"
SMOKE_INDEX_URL = f"{SMOKE_BASE_URL}/index-markdown-test"
SMOKE_FIT_URL = f"{SMOKE_BASE_URL}/index-fit-markdown-test"
SMOKE_STRUCT_URL = f"{SMOKE_BASE_URL}/index-structured-test"
SMOKE_CODE_URL = f"{SMOKE_BASE_URL}/index-code-test"
SMOKE_PROVENANCE_URL = f"{SMOKE_BASE_URL}/provenance-test"

CODE_MD = "# Code Example\n\n```python\ndef hello_world():\n    return 'Hello, World!'\n```\n"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_PYTEST_FIRST_CHUNK_ID: Optional[int] = None


@dataclass
class TestContext:
    """State shared across test functions within a single run."""

    __test__ = False

    client: Client
    tool_names: List[str]
    # Populated by test_index_markdown; used by test_get_document_by_id_round_trip.
    first_chunk_id: Optional[int] = None


# Pytest mode support (while preserving script mode below).
pytestmark = pytest.mark.asyncio


@pytest.fixture
async def ctx() -> TestContext:
    """Pytest fixture for integration smoke tests against a live MCP endpoint.

    Skips the suite when the endpoint is unavailable rather than failing fixture
    setup with connection errors.
    """
    try:
        async with Client(MCP_URL, timeout=120) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            yield TestContext(client=client, tool_names=tool_names, first_chunk_id=_PYTEST_FIRST_CHUNK_ID)
    except Exception as exc:
        pytest.skip(f"Integration smoke tests require a live MCP server at {MCP_URL}: {exc}")


# ---------------------------------------------------------------------------
# Test runner helper
# ---------------------------------------------------------------------------


async def _run_test(name: str, coro) -> bool:
    """Execute *coro*, print result, return True on pass."""
    try:
        await coro
        print(f"  [PASS] {name}")
        return True
    except AssertionError as exc:
        print(f"  [FAIL] {name}: {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


async def test_tool_registration(ctx: TestContext) -> None:
    """Assert all required tools are present; disallowed tools are absent."""
    required_tools = {
        # Acquisition
        "crawl_url",
        "crawl_to_markdown",
        "crawl_many_urls",
        "crawl_local_file",
        "crawl_raw_html",
        "crawl_deep",
        "crawl_adaptive",
        "crawl_with_session",
        "crawl_with_auth_hooks",
        "crawl_login_required",
        "crawl_paginated",
        "crawl_with_browser_config",
        "ingest_content_directory",
        # Session lifecycle
        "create_session",
        "inspect_session",
        "kill_session",
        # Extraction / transformation
        "extract_fit_markdown",
        "extract_markdown_variants",
        "extract_structured_json",
        "generate_extraction_schema",
        "validate_extraction_schema",
        "extract_regex_entities",
        "extract_knowledge_graph",
        "extract_code_examples",
        # Indexing
        "index_markdown",
        "index_fit_markdown",
        "index_structured_content",
        "index_code_examples",
        # Retrieval
        "search_documents",
        "search_raw_markdown",
        "search_fit_markdown",
        "search_structured_content",
        "get_document_by_id",
        "get_markdown_by_url",
        "get_fit_markdown_by_url",
    }
    if EXPECT_ALL_TOOLS or USE_AGENTIC_RAG:
        required_tools.add("search_code_examples")

    missing = sorted(required_tools - set(ctx.tool_names))
    assert not missing, f"Missing required tools: {missing}"

    disallowed_ops_tools = {
        "compute_value_scores",
        "preview_eviction_plan",
        "enforce_storage_budget",
        "pin_records",
        "unpin_records",
        "index_storage_report",
        "restore_tombstoned_records",
        "recrawl_due_sources",
        "prune_stale_content",
        "hard_delete_tombstones",
        "detect_content_drift",
    }
    exposed_ops = sorted(disallowed_ops_tools.intersection(set(ctx.tool_names)))
    assert not exposed_ops, f"Admin/ops tools must not be exposed via MCP: {exposed_ops}"

    disallowed_legacy_tools = {
        "crawl_single_page",
        "smart_crawl_url",
        "get_available_sources",
        "perform_rag_query",
        "search_documents_tool",
    }
    exposed_legacy = sorted(disallowed_legacy_tools.intersection(set(ctx.tool_names)))
    assert not exposed_legacy, f"Legacy tools must not be exposed via MCP: {exposed_legacy}"

    if EXPECT_NEW_TOOLS:
        required_new = {"crawl_url", "crawl_deep", "crawl_adaptive"}
        missing_new = sorted(required_new - set(ctx.tool_names))
        assert not missing_new, f"EXPECT_NEW_TOOLS=true but missing: {missing_new}; available={ctx.tool_names}"
        print(f"    strict rollout mode: verified new tools {sorted(required_new)}")


# ---------------------------------------------------------------------------
# Crawl write-path tests (all assert index_result=True writes >= 1 chunk)
# ---------------------------------------------------------------------------


async def test_crawl_url(ctx: TestContext) -> None:
    """crawl_url with index_result=True must report success and write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_url",
        {"url": TEST_URL, "mode": "markdown", "markdown_variant": "raw", "index_result": True},
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_url failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_url wrote 0 chunks: {data}"


async def test_crawl_deep(ctx: TestContext) -> None:
    """crawl_deep BFS with index_result=True must crawl pages and write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_deep",
        {
            "url": TEST_URL,
            "strategy": "bfs",
            "max_depth": 1,
            "max_pages": 3,
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_deep failed: {data}"
    assert data["pages_crawled"] >= 1, "Expected >=1 page from crawl_deep"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_deep indexed 0 chunks: {data}"


async def test_crawl_adaptive(ctx: TestContext) -> None:
    """crawl_adaptive must report pages crawled and write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_adaptive",
        {
            "url": TEST_URL,
            "query": "what is example.com about",
            "strategy": "statistical",
            "max_depth": 1,
            "max_pages": 3,
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_adaptive failed: {data}"
    assert data["pages_crawled"] >= 1, "Expected >=1 page from crawl_adaptive"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_adaptive indexed 0 chunks: {data}"

    # Export + answer workflow
    result2 = await ctx.client.call_tool(
        "crawl_adaptive",
        {
            "url": TEST_URL,
            "query": "what is example.com about",
            "strategy": "statistical",
            "max_depth": 1,
            "max_pages": 3,
            "index_result": True,
            "export_knowledge_base": True,
            "knowledge_base_format": "jsonl",
            "answer_query": "summarize this site",
            "answer_match_count": 2,
        },
    )
    data2 = json.loads(result2.data)
    assert data2["success"], f"crawl_adaptive(export+answer) failed: {data2}"
    assert data2["knowledge_base_export"]["format"] in {
        "json",
        "jsonl",
    }, f"Unexpected knowledge_base_export.format: {data2}"
    assert "adaptive_answer" in data2
    assert data2["adaptive_answer"]["query"] == "summarize this site"


async def test_crawl_to_markdown(ctx: TestContext) -> None:
    """crawl_to_markdown must write >=1 chunk (not >=0) when index_result=True."""
    result = await ctx.client.call_tool(
        "crawl_to_markdown",
        {"url": TEST_URL, "markdown_variant": "raw", "index_result": True},
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_to_markdown failed: {data}"
    assert data.get("pages_crawled", 0) >= 1, "Expected crawl_to_markdown to crawl >=1 page"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_to_markdown wrote 0 chunks: {data}"

    # index_variants='both' variant
    result2 = await ctx.client.call_tool(
        "crawl_to_markdown",
        {
            "url": TEST_URL,
            "markdown_variant": "raw",
            "index_result": True,
            "index_variants": "both",
        },
    )
    data2 = json.loads(result2.data)
    assert data2["success"], f"crawl_to_markdown(index_variants=both) failed: {data2}"
    assert data2.get("index_variants_override") == "both"
    assert "indexed_variants" in data2, "Expected indexed_variants in response"
    assert "raw_markdown" in data2.get("indexed_variants", []), f"Expected raw_markdown to be indexed: {data2}"


async def test_crawl_many_urls(ctx: TestContext) -> None:
    """crawl_many_urls with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_many_urls",
        {"urls": [TEST_URL], "markdown_variant": "raw", "index_result": True},
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_many_urls failed: {data}"
    assert data["pages_crawled"] >= 1, "Expected crawl_many_urls to crawl >=1 page"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_many_urls indexed 0 chunks: {data}"


async def test_crawl_raw_html(ctx: TestContext) -> None:
    """crawl_raw_html with index_result=True must return content and write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_raw_html",
        {
            "html": "<html><body><h1>Raw HTML Smoke Test</h1><p>Content for e2e testing.</p></body></html>",
            "markdown_variant": "raw",
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_raw_html failed: {data}"
    raw_content = data.get("selected_markdown", "")
    assert raw_content, "Expected non-empty content from crawl_raw_html"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_raw_html indexed 0 chunks: {data}"


async def test_crawl_local_file(ctx: TestContext) -> None:
    """crawl_local_file with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_local_file",
        {
            "file_path": "/app/pyproject.toml",
            "markdown_variant": "raw",
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_local_file failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_local_file indexed 0 chunks: {data}"


async def test_crawl_with_session(ctx: TestContext) -> None:
    """crawl_with_session with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_with_session",
        {
            "url": TEST_URL,
            "session_id": "smoke-e2e-session",
            "action": "reuse",
            "markdown_variant": "raw",
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_with_session failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_with_session indexed 0 chunks: {data}"


async def test_crawl_with_browser_config(ctx: TestContext) -> None:
    """crawl_with_browser_config with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_with_browser_config",
        {
            "url": TEST_URL,
            "browser_config": {"text_mode": True},
            "markdown_variant": "raw",
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_with_browser_config failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_with_browser_config indexed 0 chunks: {data}"


async def test_crawl_with_auth_hooks(ctx: TestContext) -> None:
    """crawl_with_auth_hooks with index_result=True must report correct workflow and write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_with_auth_hooks",
        {
            "url": TEST_URL,
            "session_id": "smoke-auth-session",
            "markdown_variant": "raw",
            "index_result": True,
            "custom_headers": {"X-Smoke-Test": "true"},
            "final_scroll": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_with_auth_hooks failed: {data}"
    assert data["workflow_mode"] == "direct_auth_hooks"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_with_auth_hooks indexed 0 chunks: {data}"


async def test_crawl_login_required(ctx: TestContext) -> None:
    """crawl_login_required with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_login_required",
        {
            "url": TEST_URL,
            "session_id": "smoke-auth-session",
            "login_script": "console.log('smoke login preset')",
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_login_required failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_login_required indexed 0 chunks: {data}"


async def test_crawl_paginated(ctx: TestContext) -> None:
    """crawl_paginated with index_result=True must write >=1 chunk."""
    result = await ctx.client.call_tool(
        "crawl_paginated",
        {
            "start_url": TEST_URL,
            "session_id": "smoke-auth-session",
            "additional_urls": [TEST_URL],
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"crawl_paginated failed: {data}"
    assert data.get("chunks_stored", 0) >= 1, f"crawl_paginated indexed 0 chunks: {data}"


async def test_ingest_content_directory(ctx: TestContext) -> None:
    """ingest_content_directory must discover files and index >=1 file."""
    result = await ctx.client.call_tool(
        "ingest_content_directory",
        {
            "directory_path": "/app",
            "include_patterns": ["pyproject.toml", "README.md"],
            "index_result": True,
        },
    )
    data = json.loads(result.data)
    assert data["success"], f"ingest_content_directory failed: {data}"
    assert data.get("files_discovered", 0) >= 1, "Expected >=1 file discovered"
    assert data.get("indexed_count", 0) >= 1, f"ingest_content_directory indexed 0 files: {data}"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


async def test_session_lifecycle(ctx: TestContext) -> None:
    """create_session -> inspect_session (active=True) -> kill_session."""
    create_data = json.loads((await ctx.client.call_tool("create_session", {"session_id": "smoke-test-session"})).data)
    assert create_data["success"], f"create_session failed: {create_data}"
    assert create_data["session_id"] == "smoke-test-session"

    inspect_data = json.loads(
        (await ctx.client.call_tool("inspect_session", {"session_id": "smoke-test-session"})).data
    )
    assert inspect_data["success"], f"inspect_session failed: {inspect_data}"
    assert inspect_data["active"] is True

    kill_data = json.loads((await ctx.client.call_tool("kill_session", {"session_id": "smoke-test-session"})).data)
    assert kill_data["success"], f"kill_session failed: {kill_data}"


# ---------------------------------------------------------------------------
# Extraction tools
# ---------------------------------------------------------------------------


async def test_extract_markdown_variants(ctx: TestContext) -> None:
    """extract_markdown_variants must return raw_markdown + fit_markdown and store >=1 chunk."""
    data = json.loads(
        (await ctx.client.call_tool("extract_markdown_variants", {"url": TEST_URL, "index_result": False})).data
    )
    assert data["success"], f"extract_markdown_variants failed: {data}"
    assert "raw_markdown" in data, "Expected raw_markdown in extract_markdown_variants output"
    assert "fit_markdown" in data, "Expected fit_markdown in extract_markdown_variants output"

    data2 = json.loads(
        (
            await ctx.client.call_tool(
                "extract_markdown_variants",
                {"url": TEST_URL, "index_result": True, "index_variants": "both"},
            )
        ).data
    )
    assert data2["success"], f"extract_markdown_variants(index_result=True) failed: {data2}"
    assert data2.get("index_variants_override") == "both"
    assert "indexed_variants" in data2
    assert data2.get("chunks_stored", 0) >= 1, f"extract_markdown_variants wrote 0 chunks: {data2}"


async def test_extract_fit_markdown(ctx: TestContext) -> None:
    """extract_fit_markdown must succeed and return fit_markdown variant."""
    data = json.loads((await ctx.client.call_tool("extract_fit_markdown", {"url": TEST_URL})).data)
    assert data["success"], f"extract_fit_markdown failed: {data}"
    assert data.get("selected_variant") == "fit_markdown", "Expected fit_markdown variant"


async def test_extract_structured_json(ctx: TestContext) -> None:
    """extract_structured_json and schema helpers must succeed."""
    struct_data = json.loads(
        (
            await ctx.client.call_tool(
                "extract_structured_json",
                {
                    "url": TEST_URL,
                    "extraction_strategy": "css",
                    "extraction_schema": {
                        "baseSelector": "body",
                        "fields": [
                            {"name": "title", "selector": "h1", "type": "text"},
                            {"name": "paragraph", "selector": "p", "type": "text"},
                        ],
                    },
                },
            )
        ).data
    )
    assert struct_data["success"], f"extract_structured_json failed: {struct_data}"

    schema_data = json.loads(
        (
            await ctx.client.call_tool(
                "generate_extraction_schema",
                {
                    "sample_html": (
                        "<html><head><title>Smoke</title></head>"
                        "<body><h1>Smoke H1</h1><p>Smoke paragraph</p></body></html>"
                    ),
                    "strategy": "css",
                    "cache_schema": False,
                },
            )
        ).data
    )
    assert schema_data["success"], f"generate_extraction_schema failed: {schema_data}"

    validate_data = json.loads(
        (
            await ctx.client.call_tool(
                "validate_extraction_schema",
                {"schema": schema_data["schema"], "strategy": "css"},
            )
        ).data
    )
    assert validate_data["success"], f"validate_extraction_schema failed: {validate_data}"

    struct_fit_data = json.loads(
        (
            await ctx.client.call_tool(
                "extract_structured_json",
                {
                    "url": TEST_URL,
                    "extraction_strategy": "css",
                    "extraction_schema": schema_data["schema"],
                    "fit_source": True,
                },
            )
        ).data
    )
    assert struct_fit_data["success"], f"extract_structured_json(fit_source=True) failed: {struct_fit_data}"
    assert "normalized_output" in struct_fit_data, "Expected normalized_output in structured extraction response"


async def test_extract_regex_entities(ctx: TestContext) -> None:
    """extract_regex_entities must succeed."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "extract_regex_entities",
                {
                    "url": TEST_URL,
                    "extraction_patterns": {"links": r"https?://[^\s\"']+"},
                },
            )
        ).data
    )
    assert data["success"], f"extract_regex_entities failed: {data}"


async def test_extract_code_examples(ctx: TestContext) -> None:
    """extract_code_examples must succeed and return the code_examples key."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "extract_code_examples",
                {"url": TEST_URL, "markdown_variant": "raw"},
            )
        ).data
    )
    assert data["success"], f"extract_code_examples failed: {data}"
    assert "code_examples" in data, "Expected code_examples key"


async def test_extract_knowledge_graph(ctx: TestContext) -> None:
    """extract_knowledge_graph (LLM-backed) -- skipped unless EXPECT_NEW_TOOLS=true."""
    if not EXPECT_NEW_TOOLS:
        print("    skipped -- set EXPECT_NEW_TOOLS=true to test LLM-backed extraction")
        return
    data = json.loads((await ctx.client.call_tool("extract_knowledge_graph", {"url": TEST_URL})).data)
    assert data["success"], f"extract_knowledge_graph failed: {data}"


# ---------------------------------------------------------------------------
# Indexing tools
# ---------------------------------------------------------------------------


async def test_index_markdown(ctx: TestContext) -> None:
    """index_markdown must store >=1 chunk and return a positive first_chunk_id."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_markdown",
                {
                    "url": SMOKE_INDEX_URL,
                    "markdown": ("# Smoke Test Document\n\nThis is e2e smoke test content for retrieval verification."),
                    "metadata": {"smoke_test": True},
                },
            )
        ).data
    )
    assert data["success"], f"index_markdown failed: {data}"
    assert data["chunks_stored"] >= 1, "Expected >=1 chunk from index_markdown"
    assert "first_chunk_id" in data, "Expected first_chunk_id in index_markdown response"
    first_id = data.get("first_chunk_id")
    assert first_id is not None and first_id > 0, f"Expected a positive integer first_chunk_id; got {first_id}"
    ctx.first_chunk_id = first_id
    global _PYTEST_FIRST_CHUNK_ID
    _PYTEST_FIRST_CHUNK_ID = first_id


async def test_index_fit_markdown(ctx: TestContext) -> None:
    """index_fit_markdown must store >=1 chunk."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_fit_markdown",
                {
                    "url": SMOKE_FIT_URL,
                    "fit_markdown": ("# Fit Markdown Smoke Test\n\nIndexed fit markdown for retrieval testing."),
                    "metadata": {"smoke_test": True},
                },
            )
        ).data
    )
    assert data["success"], f"index_fit_markdown failed: {data}"
    assert data["chunks_stored"] >= 1, "Expected >=1 chunk from index_fit_markdown"


async def test_index_structured_content(ctx: TestContext) -> None:
    """index_structured_content must store >=1 chunk."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_structured_content",
                {
                    "url": SMOKE_STRUCT_URL,
                    "structured_content": {
                        "title": "Smoke Test Record",
                        "tags": ["e2e", "structured"],
                    },
                    "metadata": {"smoke_test": True},
                    "projection_mode": "hybrid",
                },
            )
        ).data
    )
    assert data["success"], f"index_structured_content failed: {data}"
    assert data["chunks_stored"] >= 1, "Expected >=1 chunk from index_structured_content"


async def test_index_code_examples(ctx: TestContext) -> None:
    """index_code_examples must index >=1 code example (not just report success)."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_code_examples",
                {"url": SMOKE_CODE_URL, "markdown": CODE_MD},
            )
        ).data
    )
    assert data["success"], f"index_code_examples failed: {data}"
    code_count = data.get("code_examples_indexed", 0)
    assert code_count >= 1, f"Expected code_examples_indexed >= 1; got {code_count}: {data}"


async def test_index_provenance(ctx: TestContext) -> None:
    """index_markdown with provenance metadata must store >=1 chunk."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_markdown",
                {
                    "url": SMOKE_PROVENANCE_URL,
                    "markdown": ("# Provenance Smoke Test\n\nThis document proves provenance retrieval works."),
                    "metadata": {
                        "smoke_test": True,
                        "markdown_variant": "raw_markdown",
                        "references_markdown": ("[1]: https://example.com/reference Example reference"),
                        "has_citations": True,
                    },
                },
            )
        ).data
    )
    assert data["success"], f"provenance index_markdown failed: {data}"
    assert data["chunks_stored"] >= 1, "Expected >=1 chunk from provenance index_markdown"


async def test_chunking_strategies(ctx: TestContext) -> None:
    """index_markdown must respect chunking_strategy overrides (sentence/fixed/paragraph)."""
    sample_md = (
        "# Section One\n\nThis is the first paragraph with some content.\n\n"
        "# Section Two\n\nThis is the second paragraph with different content.\n\n"
        "# Section Three\n\nAnd a third paragraph to ensure there are multiple chunks.\n"
    )
    for strategy in ("sentence", "fixed", "paragraph"):
        url = f"{SMOKE_BASE_URL}/chunking-{strategy}-test"
        data = json.loads(
            (
                await ctx.client.call_tool(
                    "index_markdown",
                    {
                        "url": url,
                        "markdown": sample_md,
                        "chunking_strategy": strategy,
                    },
                )
            ).data
        )
        assert data["success"], f"index_markdown(chunking_strategy={strategy!r}) failed: {data}"
        assert data["chunks_stored"] >= 1, f"Expected >=1 chunk for strategy={strategy!r}: {data}"
        assert data.get("chunking_strategy_applied") == strategy, (
            f"Expected chunking_strategy_applied={strategy!r} in response: {data}"
        )


# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------


async def test_search_documents(ctx: TestContext) -> None:
    """search_documents must return >=1 result containing 'example', and >=1 post-index result."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_documents",
                {"query": "what is example.com about"},
            )
        ).data
    )
    assert data["success"], f"search_documents failed: {data}"
    assert len(data["results"]) >= 1, "Expected >=1 result from search_documents (initial)"
    top = data["results"][0]
    assert "example" in top["content"].lower(), f"Expected 'example' in content: {top['content']}"

    data2 = json.loads(
        (
            await ctx.client.call_tool(
                "search_documents",
                {"query": "smoke test retrieval verification", "match_count": 3},
            )
        ).data
    )
    assert data2["success"], f"search_documents(post-index) failed: {data2}"
    assert len(data2["results"]) >= 1, "Expected >=1 result from search_documents after indexing"


async def test_search_raw_markdown(ctx: TestContext) -> None:
    """search_raw_markdown must return >=1 result after raw markdown indexing."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_raw_markdown",
                {"query": "smoke test retrieval verification", "match_count": 3},
            )
        ).data
    )
    assert data["success"], f"search_raw_markdown failed: {data}"
    assert len(data.get("results", [])) >= 1, (
        f"Expected >=1 result from search_raw_markdown after indexing; got: {data}"
    )


async def test_search_fit_markdown(ctx: TestContext) -> None:
    """search_fit_markdown must return >=1 result after fit markdown indexing."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_fit_markdown",
                {"query": "fit markdown retrieval testing", "match_count": 3},
            )
        ).data
    )
    assert data["success"], f"search_fit_markdown failed: {data}"
    assert len(data.get("results", [])) >= 1, (
        f"Expected >=1 result from search_fit_markdown after indexing; got: {data}"
    )


async def test_search_structured_content(ctx: TestContext) -> None:
    """search_structured_content must return >=1 result after structured indexing."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_structured_content",
                {"query": "smoke test tags structured", "match_count": 3},
            )
        ).data
    )
    assert data["success"], f"search_structured_content failed: {data}"
    assert len(data.get("results", [])) >= 1, (
        f"Expected >=1 result from search_structured_content after indexing; got: {data}"
    )


async def test_search_code_examples(ctx: TestContext) -> None:
    """search_code_examples must return >=1 result after code indexing.

    Skipped when search_code_examples is absent (USE_AGENTIC_RAG not enabled).
    """
    if "search_code_examples" not in ctx.tool_names:
        print(
            "    skipped -- search_code_examples not in tool list "
            "(set USE_AGENTIC_RAG=true or EXPECT_ALL_TOOLS=true to enforce it)"
        )
        return
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_code_examples",
                {"query": "hello world python function", "match_count": 3},
            )
        ).data
    )
    assert data["success"], f"search_code_examples failed: {data}"
    assert len(data.get("results", [])) >= 1, (
        f"Expected >=1 result from search_code_examples after indexing; got: {data}"
    )


async def test_search_provenance(ctx: TestContext) -> None:
    """search_documents with include_provenance must surface citation fields."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_documents",
                {
                    "query": "proves provenance retrieval works",
                    "match_count": 3,
                    "include_provenance": True,
                },
            )
        ).data
    )
    assert data["success"], f"search_documents(include_provenance) failed: {data}"
    provenance_hits = [h for h in data["results"] if h.get("url") == SMOKE_PROVENANCE_URL]
    assert provenance_hits, f"Expected provenance smoke URL '{SMOKE_PROVENANCE_URL}' in results: {data}"
    assert provenance_hits[0]["provenance"]["has_citations"] is True
    assert provenance_hits[0]["provenance"]["link_references"][0]["url"] == "https://example.com/reference"


async def test_search_freshness_controls(ctx: TestContext) -> None:
    """search_documents with freshness controls must return expected response fields."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "search_documents",
                {
                    "query": "smoke test retrieval verification",
                    "match_count": 3,
                    "fresh_only": True,
                    "as_of": "2999-01-01T00:00:00+00:00",
                    "recency_bias": 0.4,
                },
            )
        ).data
    )
    assert data["success"], f"search_documents(freshness controls) failed: {data}"
    assert data["fresh_only"] is True
    assert data["as_of"] is not None
    if data["results"]:
        first = data["results"][0]
        assert "final_score" in first, f"Expected final_score with recency_bias>0: {data}"
        assert "freshness_score" in first, f"Expected freshness_score with recency_bias>0: {data}"


async def test_get_document_by_id_round_trip(ctx: TestContext) -> None:
    """get_document_by_id must retrieve the exact document stored by test_index_markdown."""
    first_chunk_id = ctx.first_chunk_id or _PYTEST_FIRST_CHUNK_ID
    assert first_chunk_id is not None, "Cannot run round-trip test: test_index_markdown did not provide first_chunk_id"
    data = json.loads(
        (
            await ctx.client.call_tool(
                "get_document_by_id",
                {"document_id": first_chunk_id},
            )
        ).data
    )
    assert data["success"], f"get_document_by_id({first_chunk_id}) returned failure: {data}"
    assert "document" in data, "Expected 'document' key in get_document_by_id response"
    doc = data["document"]
    assert doc["url"] == SMOKE_INDEX_URL, f"Expected url={SMOKE_INDEX_URL!r}, got {doc['url']!r}"
    assert doc["id"] == first_chunk_id


async def test_get_markdown_by_url(ctx: TestContext) -> None:
    """get_markdown_by_url must return >=1 chunk for a previously indexed URL."""
    data = json.loads((await ctx.client.call_tool("get_markdown_by_url", {"url": TEST_URL})).data)
    assert data["success"], f"get_markdown_by_url failed: {data}"
    assert data["chunk_count"] >= 1, "Expected >=1 stored chunk from get_markdown_by_url"


async def test_get_fit_markdown_by_url(ctx: TestContext) -> None:
    """get_fit_markdown_by_url must return >=1 chunk for the fit-markdown smoke URL."""
    data = json.loads((await ctx.client.call_tool("get_fit_markdown_by_url", {"url": SMOKE_FIT_URL})).data)
    assert data["success"], f"get_fit_markdown_by_url failed: {data}"
    assert data["chunk_count"] >= 1, "Expected >=1 fit markdown chunk"


async def test_get_markdown_provenance(ctx: TestContext) -> None:
    """get_markdown_by_url with include_provenance must surface citation fields."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "get_markdown_by_url",
                {"url": SMOKE_PROVENANCE_URL, "include_provenance": True},
            )
        ).data
    )
    assert data["success"], f"get_markdown_by_url(include_provenance) failed: {data}"
    assert data["provenance"]["has_citations"] is True
    assert data["provenance"]["link_references"][0]["url"] == "https://example.com/reference"


# ---------------------------------------------------------------------------
# Error / failure-mode tests
# ---------------------------------------------------------------------------


async def test_error_index_empty_content(ctx: TestContext) -> None:
    """index_markdown with empty markdown must return success=False with an error field."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "index_markdown",
                {"url": f"{SMOKE_BASE_URL}/error-empty", "markdown": ""},
            )
        ).data
    )
    assert data["success"] is False, f"Expected failure when indexing empty markdown; got: {data}"
    assert "error" in data, "Expected 'error' key in failure response"


async def test_error_crawl_raw_html_empty(ctx: TestContext) -> None:
    """crawl_raw_html with empty html must return success=False with an error field."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "crawl_raw_html",
                {"html": "", "markdown_variant": "raw", "index_result": False},
            )
        ).data
    )
    assert data["success"] is False, f"Expected failure when crawling empty HTML; got: {data}"
    assert "error" in data, "Expected 'error' key in failure response"


async def test_error_get_document_nonexistent(ctx: TestContext) -> None:
    """get_document_by_id with a non-existent ID must return success=False."""
    data = json.loads(
        (
            await ctx.client.call_tool(
                "get_document_by_id",
                {"document_id": 999_999_999},
            )
        ).data
    )
    assert data["success"] is False, f"Expected success=False for non-existent document ID; got: {data}"
    assert "error" in data, "Expected 'error' key in not-found response"


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

# Tests are ordered so indexing always runs before retrieval.
# Shared state (first_chunk_id) is populated by test_index_markdown.
_TEST_SEQUENCE = [
    # Tool surface sanity
    ("tool_registration", test_tool_registration),
    # Crawl write-path
    ("crawl_url [index+write]", test_crawl_url),
    ("crawl_deep [index+write]", test_crawl_deep),
    ("crawl_adaptive [index+write]", test_crawl_adaptive),
    ("crawl_to_markdown [index+write]", test_crawl_to_markdown),
    ("crawl_many_urls [index+write]", test_crawl_many_urls),
    ("crawl_raw_html [index+write]", test_crawl_raw_html),
    ("crawl_local_file [index+write]", test_crawl_local_file),
    ("session_lifecycle", test_session_lifecycle),
    ("crawl_with_session [index+write]", test_crawl_with_session),
    ("crawl_with_browser_config [index+write]", test_crawl_with_browser_config),
    ("crawl_with_auth_hooks [index+write]", test_crawl_with_auth_hooks),
    ("crawl_login_required [index+write]", test_crawl_login_required),
    ("crawl_paginated [index+write]", test_crawl_paginated),
    ("ingest_content_directory [index+write]", test_ingest_content_directory),
    # Extraction (no writes required)
    ("extract_markdown_variants", test_extract_markdown_variants),
    ("extract_fit_markdown", test_extract_fit_markdown),
    ("extract_structured_json + schema helpers", test_extract_structured_json),
    ("extract_regex_entities", test_extract_regex_entities),
    ("extract_code_examples", test_extract_code_examples),
    ("extract_knowledge_graph", test_extract_knowledge_graph),
    # Indexing (must run before retrieval)
    ("index_markdown [-> first_chunk_id]", test_index_markdown),
    ("index_fit_markdown", test_index_fit_markdown),
    ("index_structured_content", test_index_structured_content),
    ("index_code_examples [write verified]", test_index_code_examples),
    ("index_markdown [provenance]", test_index_provenance),
    ("index_markdown [chunking strategies]", test_chunking_strategies),
    # Retrieval (depend on previous indexing)
    ("search_documents [initial + post-index]", test_search_documents),
    ("search_raw_markdown [result count]", test_search_raw_markdown),
    ("search_fit_markdown [result count]", test_search_fit_markdown),
    ("search_structured_content [result count]", test_search_structured_content),
    ("search_code_examples [result count]", test_search_code_examples),
    ("search_documents [provenance]", test_search_provenance),
    ("search_documents [freshness controls]", test_search_freshness_controls),
    ("get_document_by_id [round-trip]", test_get_document_by_id_round_trip),
    ("get_markdown_by_url", test_get_markdown_by_url),
    ("get_fit_markdown_by_url", test_get_fit_markdown_by_url),
    ("get_markdown_by_url [provenance]", test_get_markdown_provenance),
    # Error / failure-mode paths
    ("error: index empty content", test_error_index_empty_content),
    ("error: crawl_raw_html empty html", test_error_crawl_raw_html_empty),
    ("error: get_document_by_id nonexistent", test_error_get_document_nonexistent),
]


async def run_smoke_tests() -> None:
    print(f"Connecting to MCP server at {MCP_URL} ...")
    async with Client(MCP_URL, timeout=120) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"  tools registered: {len(tool_names)}")

        ctx = TestContext(client=client, tool_names=tool_names)

        results: dict = {}
        for name, fn in _TEST_SEQUENCE:
            print(f"\n[{name}]")
            passed = await _run_test(name, fn(ctx))
            results[name] = passed

    passed_count = sum(results.values())
    total = len(results)
    failed = [n for n, ok in results.items() if not ok]

    print(f"\n{'=' * 60}")
    print(f"Results: {passed_count}/{total} passed")
    if failed:
        print(f"\nFailed tests ({len(failed)}):")
        for n in failed:
            print(f"  - {n}")
        print("\n=== Smoke tests FAILED ===")
        sys.exit(1)
    else:
        print(f"\n=== All {total} smoke tests PASSED ===")


if __name__ == "__main__":
    try:
        asyncio.run(run_smoke_tests())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n=== Smoke test FAILED (unhandled): {exc} ===", file=sys.stderr)
        sys.exit(1)
