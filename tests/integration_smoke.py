"""
Integration smoke test — must be run against a live Docker stack.

Usage:
    MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py
"""
import asyncio
import json
import os
import sys

from fastmcp import Client

MCP_URL = os.getenv("MCP_URL", "http://localhost:8051/sse")
TEST_URL = "https://example.com"


async def run_smoke_tests() -> None:
    print(f"Connecting to MCP server at {MCP_URL} …")
    async with Client(MCP_URL, timeout=120) as c:
        # --- list tools ---
        tools = await c.list_tools()
        tool_names = [t.name for t in tools]
        assert set(tool_names) >= {
            "crawl_single_page",
            "get_available_sources",
            "perform_rag_query",
        }, f"Unexpected tool list: {tool_names}"
        print(f"  tools: {tool_names}")

        # --- crawl a page ---
        print(f"\nCrawling {TEST_URL} …")
        crawl_result = await c.call_tool("crawl_single_page", {"url": TEST_URL})
        crawl_data = json.loads(crawl_result.data)
        assert crawl_data["success"], f"Crawl failed: {crawl_data}"
        assert crawl_data["chunks_stored"] >= 1, "Expected ≥1 chunk stored"
        print(f"  chunks_stored={crawl_data['chunks_stored']}")

        # --- sources ---
        print("\nListing sources …")
        src_result = await c.call_tool("get_available_sources", {})
        src_data = json.loads(src_result.data)
        assert src_data["success"], f"get_available_sources failed: {src_data}"
        assert any("example.com" in s for s in src_data["sources"]), (
            f"example.com not in sources: {src_data['sources']}"
        )
        print(f"  sources={src_data['sources']}")

        # --- RAG retrieval ---
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

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    try:
        asyncio.run(run_smoke_tests())
    except Exception as exc:
        print(f"\n=== Smoke test FAILED: {exc} ===", file=sys.stderr)
        sys.exit(1)
