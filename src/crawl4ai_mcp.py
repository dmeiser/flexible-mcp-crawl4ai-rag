"""
MCP server for web crawling with Crawl4AI.

Provides tools to crawl websites, store content using pgvector, and perform
RAG queries. Uses FastMCP 3.2.0 with async lifespan.
"""
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from fastmcp import FastMCP, Context
from crawl4ai import AsyncWebCrawler, BrowserConfig

from src.utils import get_session, engine, settings
from sqlmodel import text
from sqlalchemy.orm import Session

# Load .env from project root before any settings are imported
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=True)

log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    crawler: AsyncWebCrawler


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Start crawler and verify DB connection, then yield context."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    logger.info("Starting AsyncWebCrawler...")
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    logger.info("AsyncWebCrawler ready.")

    logger.info("Checking DB connection...")
    try:
        with next(get_session()) as session:
            session.exec(text("SELECT 1")).first()
        logger.info("DB connection OK.")
    except Exception as exc:
        logger.error(f"DB connection failed: {exc}")
        await crawler.__aexit__(None, None, None)
        raise

    try:
        yield AppContext(crawler=crawler)
    finally:
        logger.info("Shutting down AsyncWebCrawler...")
        await crawler.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Create server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "mcp-docs-rag",
    instructions="Web crawling and RAG server built on Crawl4AI and pgvector.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Register tools
# ---------------------------------------------------------------------------
from src.crawler import tool_definitions  # noqa: E402

mcp.tool()(tool_definitions.crawl_single_page)
mcp.tool()(tool_definitions.smart_crawl_url)
mcp.tool()(tool_definitions.get_available_sources)
mcp.tool()(tool_definitions.perform_rag_query)

if settings.USE_AGENTIC_RAG:
    mcp.tool()(tool_definitions.search_code_examples)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    transport = os.getenv("TRANSPORT", "sse")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))
    logger.info(f"Starting MCP server [{transport}] on {host}:{port}")
    if transport == "stdio":
        await mcp.run_stdio_async()
    else:
        await mcp.run_http_async(transport="sse", host=host, port=port)


if __name__ == "__main__":
    asyncio.run(main())
