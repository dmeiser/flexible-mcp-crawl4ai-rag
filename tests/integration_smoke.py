"""
Integration smoke test — must be run against a live Docker stack.

Usage:
    MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py

Strict rollout check (require newly introduced tools):
    EXPECT_NEW_TOOLS=true MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py
"""
import asyncio
import json
import os
import sys

from fastmcp import Client

MCP_URL = os.getenv("MCP_URL", "http://localhost:8051/sse")
TEST_URL = "https://example.com"
EXPECT_NEW_TOOLS = os.getenv("EXPECT_NEW_TOOLS", "false").lower() in {"1", "true", "yes", "on"}


async def run_smoke_tests() -> None:
    print(f"Connecting to MCP server at {MCP_URL} …")
    async with Client(MCP_URL, timeout=120) as c:
        # --- list tools ---
        tools = await c.list_tools()
        tool_names = [t.name for t in tools]
        required_tools = {
            # Acquisition
            "crawl_single_page",
            "crawl_url",
            "crawl_to_markdown",
            "crawl_many_urls",
            "crawl_local_file",
            "crawl_raw_html",
            "smart_crawl_url",
            "crawl_deep",
            "crawl_adaptive",
            "crawl_with_session",
            "crawl_with_browser_config",
            # Session lifecycle
            "create_session",
            "inspect_session",
            "kill_session",
            # Source registry
            "get_available_sources",
            # Extraction / transformation
            "extract_fit_markdown",
            "extract_markdown_variants",
            "extract_structured_json",
            "extract_regex_entities",
            "extract_knowledge_graph",
            "extract_code_examples",
            # Indexing
            "index_markdown",
            "index_fit_markdown",
            "index_structured_content",
            "index_code_examples",
            # Retrieval
            "search_documents",        # registered name for search_documents_v2
            "search_structured_content",
            "search_documents_tool",
            "perform_rag_query",
            "get_document_by_id",
            "get_markdown_by_url",
            "get_fit_markdown_by_url",
        }
        assert set(tool_names) >= required_tools, f"Unexpected tool list: {tool_names}"
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
        }
        exposed_disallowed = sorted(disallowed_ops_tools.intersection(set(tool_names)))
        assert not exposed_disallowed, (
            f"Admin/ops tools must not be exposed via MCP: {exposed_disallowed}"
        )
        if EXPECT_NEW_TOOLS:
            required_new_tools = {"crawl_url", "crawl_deep", "crawl_adaptive"}
            assert required_new_tools.issubset(set(tool_names)), (
                f"EXPECT_NEW_TOOLS=true but missing: {sorted(required_new_tools - set(tool_names))}; "
                f"available={tool_names}"
            )
            print(f"  strict rollout mode enabled; verified new tools: {sorted(required_new_tools)}")
        print(f"  tools: {tool_names}")

        # --- crawl a page ---
        print(f"\nCrawling {TEST_URL} …")
        crawl_result = await c.call_tool("crawl_single_page", {"url": TEST_URL})
        crawl_data = json.loads(crawl_result.data)
        assert crawl_data["success"], f"Crawl failed: {crawl_data}"
        assert crawl_data["chunks_stored"] >= 1, "Expected ≥1 chunk stored"
        print(f"  chunks_stored={crawl_data['chunks_stored']}")

        # --- unified crawl entrypoint (optional during rollout) ---
        if "crawl_url" in tool_names:
            print("\nCrawling via crawl_url (markdown mode) …")
            crawl_url_result = await c.call_tool(
                "crawl_url",
                {"url": TEST_URL, "mode": "markdown", "markdown_variant": "raw", "index_result": False},
            )
            crawl_url_data = json.loads(crawl_url_result.data)
            assert crawl_url_data["success"], f"crawl_url failed: {crawl_url_data}"
        else:
            print("\nSkipping crawl_url check (tool not present in running stack image).")

        # --- crawl_deep (optional during rollout) ---
        if "crawl_deep" in tool_names:
            print("\nDeep-crawling via crawl_deep (bfs, depth=1, 3 pages) …")
            deep_result = await c.call_tool(
                "crawl_deep",
                {
                    "url": TEST_URL,
                    "strategy": "bfs",
                    "max_depth": 1,
                    "max_pages": 3,
                    "index_result": False,
                },
            )
            deep_data = json.loads(deep_result.data)
            assert deep_data["success"], f"crawl_deep failed: {deep_data}"
            assert deep_data["pages_crawled"] >= 1, "Expected ≥1 page from crawl_deep"
        else:
            print("\nSkipping crawl_deep check (tool not present in running stack image).")

        # --- crawl_adaptive (optional during rollout) ---
        if "crawl_adaptive" in tool_names:
            print("\nAdaptive-crawling via crawl_adaptive …")
            adaptive_result = await c.call_tool(
                "crawl_adaptive",
                {
                    "url": TEST_URL,
                    "query": "what is example.com about",
                    "strategy": "statistical",
                    "max_depth": 1,
                    "max_pages": 3,
                    "index_result": False,
                },
            )
            adaptive_data = json.loads(adaptive_result.data)
            assert adaptive_data["success"], f"crawl_adaptive failed: {adaptive_data}"
            assert adaptive_data["pages_crawled"] >= 1, "Expected ≥1 page from crawl_adaptive"
        else:
            print("\nSkipping crawl_adaptive check (tool not present in running stack image).")

        # --- sources ---
        print("\nListing sources …")
        src_result = await c.call_tool("get_available_sources", {})
        src_data = json.loads(src_result.data)
        assert src_data["success"], f"get_available_sources failed: {src_data}"
        assert any("example.com" in s for s in src_data["sources"]), (
            f"example.com not in sources: {src_data['sources']}"
        )
        print(f"  sources={src_data['sources']}")

        # --- markdown-only crawl with optional indexing ---
        print("\nCrawling via crawl_to_markdown …")
        md_result = await c.call_tool(
            "crawl_to_markdown",
            {"url": TEST_URL, "markdown_variant": "raw", "index_result": True},
        )
        md_data = json.loads(md_result.data)
        assert md_data["success"], f"crawl_to_markdown failed: {md_data}"
        assert md_data["chunks_stored"] >= 1, "Expected crawl_to_markdown to store chunks"

        # --- fit markdown extraction (optional during rollout) ---
        if "extract_fit_markdown" in tool_names:
            print("\nExtracting fit markdown …")
            fit_result = await c.call_tool("extract_fit_markdown", {"url": TEST_URL})
            fit_data = json.loads(fit_result.data)
            assert fit_data["success"], f"extract_fit_markdown failed: {fit_data}"
            assert fit_data.get("selected_variant") == "fit_markdown", "Expected fit_markdown variant"
        else:
            print("\nSkipping extract_fit_markdown check (tool not present in running stack image).")

        # --- structured extraction (optional during rollout) ---
        if "extract_structured_json" in tool_names:
            print("\nExtracting structured JSON …")
            struct_result = await c.call_tool(
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
            struct_data = json.loads(struct_result.data)
            assert struct_data["success"], f"extract_structured_json failed: {struct_data}"
        else:
            print("\nSkipping extract_structured_json check (tool not present in running stack image).")

        # --- batch crawl tool ---
        print("\nCrawling via crawl_many_urls …")
        many_result = await c.call_tool(
            "crawl_many_urls",
            {"urls": [TEST_URL], "markdown_variant": "raw", "index_result": False},
        )
        many_data = json.loads(many_result.data)
        assert many_data["success"], f"crawl_many_urls failed: {many_data}"
        assert many_data["pages_crawled"] >= 1, "Expected crawl_many_urls to crawl at least one page"

        # --- retrieve stored markdown by URL ---
        print("\nRetrieving markdown by URL …")
        get_md_result = await c.call_tool("get_markdown_by_url", {"url": TEST_URL})
        get_md_data = json.loads(get_md_result.data)
        assert get_md_data["success"], f"get_markdown_by_url failed: {get_md_data}"
        assert get_md_data["chunk_count"] >= 1, "Expected at least one stored chunk"

        # --- RAG retrieval (legacy + new alias) ---
        print("\nQuerying RAG …")
        rag_result = await c.call_tool(
            "perform_rag_query", {"query": "what is example.com about"}
        )
        rag_data = json.loads(rag_result.data)
        assert rag_data["success"], f"RAG query failed: {rag_data}"
        assert len(rag_data["results"]) >= 1, "Expected ≥1 result"
        top = rag_data["results"][0]
        assert "example" in top["content"].lower(), (
            f"Expected 'example' in content: {top['content']}"
        )
        print(f"  top result similarity={top['similarity']:.4f}")

        print("\nQuerying RAG via search_documents_tool …")
        search_result = await c.call_tool(
            "search_documents_tool", {"query": "what is example.com about", "match_count": 3}
        )
        search_data = json.loads(search_result.data)
        assert search_data["success"], f"search_documents_tool failed: {search_data}"
        assert len(search_data["results"]) >= 1, "Expected ≥1 result from search_documents_tool"

        # ================================================================
        # Phase 4-9 tool coverage
        # ================================================================
        SMOKE_BASE_URL = "https://smoke-test.internal"
        SMOKE_INDEX_URL = f"{SMOKE_BASE_URL}/index-markdown-test"
        SMOKE_FIT_URL = f"{SMOKE_BASE_URL}/index-fit-markdown-test"
        SMOKE_STRUCT_URL = f"{SMOKE_BASE_URL}/index-structured-test"
        SMOKE_CODE_URL = f"{SMOKE_BASE_URL}/index-code-test"
        CODE_MD = (
            "# Code Example\n\n"
            "```python\n"
            "def hello_world():\n"
            "    return 'Hello, World!'\n"
            "```\n"
        )

        # --- crawl_raw_html ---
        print("\nCrawling raw HTML via crawl_raw_html …")
        raw_html_result = await c.call_tool(
            "crawl_raw_html",
            {
                "html": "<html><body><h1>Raw HTML Smoke Test</h1><p>Content for e2e testing.</p></body></html>",
                "markdown_variant": "raw",
                "index_result": False,
            },
        )
        raw_html_data = json.loads(raw_html_result.data)
        assert raw_html_data["success"], f"crawl_raw_html failed: {raw_html_data}"
        raw_content = raw_html_data.get("selected_markdown", "")
        assert raw_content, "Expected non-empty content from crawl_raw_html"
        print(f"  crawl_raw_html: {len(raw_content)} chars")

        # --- crawl_local_file ---
        # Use a file that is guaranteed to exist in the runtime image.
        print("\nCrawling local file via crawl_local_file …")
        local_file_result = await c.call_tool(
            "crawl_local_file",
            {"file_path": "/app/pyproject.toml", "markdown_variant": "raw", "index_result": False},
        )
        local_file_data = json.loads(local_file_result.data)
        assert local_file_data["success"], f"crawl_local_file failed: {local_file_data}"
        print(f"  crawl_local_file: success")

        # --- smart_crawl_url ---
        print("\nCrawling via smart_crawl_url (single mode) …")
        smart_result = await c.call_tool(
            "smart_crawl_url",
            {"url": TEST_URL, "crawl_mode": "single"},
        )
        smart_data = json.loads(smart_result.data)
        assert smart_data["success"], f"smart_crawl_url failed: {smart_data}"
        assert smart_data.get("deprecated") is True, "smart_crawl_url should carry deprecated=true"
        print(f"  smart_crawl_url: pages_crawled={smart_data.get('pages_crawled')}")

        # --- session lifecycle: create / inspect / kill ---
        print("\nTesting session lifecycle (create / inspect / kill) …")
        create_sess_data = json.loads(
            (await c.call_tool("create_session", {"session_id": "smoke-test-session"})).data
        )
        assert create_sess_data["success"], f"create_session failed: {create_sess_data}"
        assert create_sess_data["session_id"] == "smoke-test-session"

        inspect_data = json.loads(
            (await c.call_tool("inspect_session", {"session_id": "smoke-test-session"})).data
        )
        assert inspect_data["success"], f"inspect_session failed: {inspect_data}"
        assert inspect_data["active"] is True

        kill_data = json.loads(
            (await c.call_tool("kill_session", {"session_id": "smoke-test-session"})).data
        )
        assert kill_data["success"], f"kill_session failed: {kill_data}"
        print("  session lifecycle: create → inspect → kill all passed")

        # --- crawl_with_session ---
        print("\nCrawling with session via crawl_with_session …")
        session_crawl_data = json.loads(
            (
                await c.call_tool(
                    "crawl_with_session",
                    {
                        "url": TEST_URL,
                        "session_id": "smoke-e2e-session",
                        "action": "reuse",
                        "markdown_variant": "raw",
                        "index_result": False,
                    },
                )
            ).data
        )
        assert session_crawl_data["success"], f"crawl_with_session failed: {session_crawl_data}"
        print(f"  crawl_with_session: success")

        # --- crawl_with_browser_config ---
        print("\nCrawling with browser config via crawl_with_browser_config …")
        browser_crawl_data = json.loads(
            (
                await c.call_tool(
                    "crawl_with_browser_config",
                    {
                        "url": TEST_URL,
                        "browser_config": {"text_mode": True},
                        "markdown_variant": "raw",
                        "index_result": False,
                    },
                )
            ).data
        )
        assert browser_crawl_data["success"], f"crawl_with_browser_config failed: {browser_crawl_data}"
        print(f"  crawl_with_browser_config: selected_variant={browser_crawl_data.get('selected_variant')}")

        # --- extract_markdown_variants ---
        print("\nExtracting all markdown variants …")
        variants_data = json.loads(
            (await c.call_tool("extract_markdown_variants", {"url": TEST_URL, "index_result": False})).data
        )
        assert variants_data["success"], f"extract_markdown_variants failed: {variants_data}"
        assert "raw_markdown" in variants_data, "Expected raw_markdown key in extract_markdown_variants output"
        assert "fit_markdown" in variants_data, "Expected fit_markdown key in extract_markdown_variants output"
        print(f"  extract_markdown_variants: raw={len(variants_data.get('raw_markdown', ''))} chars")

        # --- extract_regex_entities ---
        print("\nExtracting regex entities via extract_regex_entities …")
        regex_data = json.loads(
            (
                await c.call_tool(
                    "extract_regex_entities",
                    {
                        "url": TEST_URL,
                        "extraction_patterns": {"links": r"https?://[^\s\"']+"},
                    },
                )
            ).data
        )
        assert regex_data["success"], f"extract_regex_entities failed: {regex_data}"
        print(f"  extract_regex_entities: success")

        # --- extract_code_examples ---
        print("\nExtracting code examples via extract_code_examples …")
        code_examples_data = json.loads(
            (await c.call_tool("extract_code_examples", {"url": TEST_URL, "markdown_variant": "raw"})).data
        )
        assert code_examples_data["success"], f"extract_code_examples failed: {code_examples_data}"
        assert "code_examples" in code_examples_data, "Expected code_examples key"
        print(f"  extract_code_examples: count={code_examples_data.get('count', 0)}")

        # --- extract_knowledge_graph (optional — needs LLM) ---
        if EXPECT_NEW_TOOLS:
            print("\nExtracting knowledge graph via extract_knowledge_graph (LLM) …")
            kg_data = json.loads(
                (await c.call_tool("extract_knowledge_graph", {"url": TEST_URL})).data
            )
            assert kg_data["success"], f"extract_knowledge_graph failed: {kg_data}"
            print("  extract_knowledge_graph: success")
        else:
            print("\nSkipping extract_knowledge_graph (EXPECT_NEW_TOOLS not set — LLM may be unavailable).")

        # ================================================================
        # Indexing phase
        # ================================================================

        # --- index_markdown ---
        print(f"\nIndexing markdown via index_markdown → {SMOKE_INDEX_URL} …")
        index_md_data = json.loads(
            (
                await c.call_tool(
                    "index_markdown",
                    {
                        "url": SMOKE_INDEX_URL,
                        "markdown": "# Smoke Test Document\n\nThis is e2e smoke test content for retrieval verification.",
                        "metadata": {"smoke_test": True},
                    },
                )
            ).data
        )
        assert index_md_data["success"], f"index_markdown failed: {index_md_data}"
        assert index_md_data["chunks_stored"] >= 1, "Expected ≥1 chunk from index_markdown"
        print(f"  index_markdown: chunks_stored={index_md_data['chunks_stored']}")

        # --- index_fit_markdown ---
        print(f"\nIndexing fit markdown via index_fit_markdown → {SMOKE_FIT_URL} …")
        index_fit_data = json.loads(
            (
                await c.call_tool(
                    "index_fit_markdown",
                    {
                        "url": SMOKE_FIT_URL,
                        "fit_markdown": "# Fit Markdown Smoke Test\n\nIndexed fit markdown for retrieval testing.",
                        "metadata": {"smoke_test": True},
                    },
                )
            ).data
        )
        assert index_fit_data["success"], f"index_fit_markdown failed: {index_fit_data}"
        assert index_fit_data["chunks_stored"] >= 1, "Expected ≥1 chunk from index_fit_markdown"
        print(f"  index_fit_markdown: chunks_stored={index_fit_data['chunks_stored']}")

        # --- index_structured_content ---
        print(f"\nIndexing structured content via index_structured_content → {SMOKE_STRUCT_URL} …")
        index_struct_data = json.loads(
            (
                await c.call_tool(
                    "index_structured_content",
                    {
                        "url": SMOKE_STRUCT_URL,
                        "structured_content": {"title": "Smoke Test Record", "tags": ["e2e", "structured"]},
                        "metadata": {"smoke_test": True},
                    },
                )
            ).data
        )
        assert index_struct_data["success"], f"index_structured_content failed: {index_struct_data}"
        assert index_struct_data["chunks_stored"] >= 1, "Expected ≥1 chunk from index_structured_content"
        print(f"  index_structured_content: chunks_stored={index_struct_data['chunks_stored']}")

        # --- index_code_examples ---
        print(f"\nIndexing code blocks via index_code_examples → {SMOKE_CODE_URL} …")
        index_code_data = json.loads(
            (await c.call_tool("index_code_examples", {"url": SMOKE_CODE_URL, "markdown": CODE_MD})).data
        )
        assert index_code_data["success"], f"index_code_examples failed: {index_code_data}"
        print(f"  index_code_examples: code_examples_indexed={index_code_data.get('code_examples_indexed', 0)}")

        # ================================================================
        # Retrieval phase (after indexing)
        # ================================================================

        # --- search_documents (taxonomy-native alias) ---
        print("\nQuerying via search_documents …")
        sd_data = json.loads(
            (
                await c.call_tool(
                    "search_documents",
                    {"query": "smoke test retrieval verification", "match_count": 3},
                )
            ).data
        )
        assert sd_data["success"], f"search_documents failed: {sd_data}"
        assert len(sd_data["results"]) >= 1, "Expected ≥1 result from search_documents"
        print(f"  search_documents: top similarity={sd_data['results'][0]['similarity']:.4f}")

        # --- search_structured_content ---
        print("\nQuerying structured content via search_structured_content …")
        struct_search_data = json.loads(
            (
                await c.call_tool(
                    "search_structured_content",
                    {"query": "smoke test tags structured", "match_count": 3},
                )
            ).data
        )
        assert struct_search_data["success"], f"search_structured_content failed: {struct_search_data}"
        print(f"  search_structured_content: {len(struct_search_data.get('results', []))} results")

        # --- get_document_by_id ---
        print("\nFetching document by ID via get_document_by_id …")
        doc_by_id_data = json.loads(
            (await c.call_tool("get_document_by_id", {"document_id": 1})).data
        )
        # Either found or not found depending on DB state; both are valid smoke outcomes
        assert "success" in doc_by_id_data, f"get_document_by_id missing 'success' key: {doc_by_id_data}"
        if doc_by_id_data["success"]:
            assert "document" in doc_by_id_data, "Expected 'document' in successful get_document_by_id response"
            print(f"  get_document_by_id: found id=1, url={doc_by_id_data['document'].get('url')}")
        else:
            print("  get_document_by_id: not found (id=1) — acceptable for fresh DB")

        # --- get_fit_markdown_by_url ---
        print(f"\nRetrieving fit markdown via get_fit_markdown_by_url → {SMOKE_FIT_URL} …")
        fit_md_by_url_data = json.loads(
            (await c.call_tool("get_fit_markdown_by_url", {"url": SMOKE_FIT_URL})).data
        )
        assert fit_md_by_url_data["success"], f"get_fit_markdown_by_url failed: {fit_md_by_url_data}"
        assert fit_md_by_url_data["chunk_count"] >= 1, "Expected ≥1 fit markdown chunk"
        print(f"  get_fit_markdown_by_url: chunk_count={fit_md_by_url_data['chunk_count']}")

        # --- search_code_examples (conditional on USE_AGENTIC_RAG) ---
        if "search_code_examples" in tool_names:
            print("\nSearching code examples via search_code_examples …")
            search_code_data = json.loads(
                (
                    await c.call_tool(
                        "search_code_examples",
                        {"query": "hello world python function", "match_count": 3},
                    )
                ).data
            )
            assert search_code_data["success"], f"search_code_examples failed: {search_code_data}"
            print(f"  search_code_examples: {len(search_code_data.get('results', []))} results")
        else:
            print("\nSkipping search_code_examples (USE_AGENTIC_RAG not enabled).")

        print("\nVerified: admin/ops maintenance endpoints are not exposed via MCP.")

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    try:
        asyncio.run(run_smoke_tests())
    except Exception as exc:
        print(f"\n=== Smoke test FAILED: {exc} ===", file=sys.stderr)
        sys.exit(1)
