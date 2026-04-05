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
from types import SimpleNamespace
from pathlib import Path
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

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
    scheduler: AsyncIOScheduler | None = None


def _make_scheduler_ctx(crawler: AsyncWebCrawler) -> Context:
    """Create a minimal context object compatible with tool_definitions._get_crawler."""
    # fastmcp.Context is only used by our tools via `lifespan_context.crawler`.
    return SimpleNamespace(lifespan_context=SimpleNamespace(crawler=crawler))  # type: ignore[return-value]


async def _job_compute_value_scores(crawler: AsyncWebCrawler) -> None:
    try:
        from src.crawler import tool_definitions as _td
        await _td.compute_value_scores(_make_scheduler_ctx(crawler), limit=2000)
    except Exception as exc:
        logger.error(f"scheduled job compute_value_scores failed: {exc}", exc_info=True)


async def _job_recrawl_due_sources(crawler: AsyncWebCrawler) -> None:
    try:
        from src.crawler import tool_definitions as _td
        await _td.recrawl_due_sources(_make_scheduler_ctx(crawler), max_concurrent=5)
    except Exception as exc:
        logger.error(f"scheduled job recrawl_due_sources failed: {exc}", exc_info=True)


async def _job_prune_stale_content(crawler: AsyncWebCrawler) -> None:
    try:
        from src.crawler import tool_definitions as _td
        await _td.prune_stale_content(_make_scheduler_ctx(crawler), force=False)
    except Exception as exc:
        logger.error(f"scheduled job prune_stale_content failed: {exc}", exc_info=True)


async def _job_enforce_storage_budget(crawler: AsyncWebCrawler) -> None:
    try:
        from src.crawler import tool_definitions as _td
        await _td.enforce_storage_budget(_make_scheduler_ctx(crawler), force=False)
    except Exception as exc:
        logger.error(f"scheduled job enforce_storage_budget failed: {exc}", exc_info=True)


async def _job_hard_delete_tombstones(crawler: AsyncWebCrawler) -> None:
    try:
        from src.crawler import tool_definitions as _td
        await _td.hard_delete_tombstones(_make_scheduler_ctx(crawler), max_age_hours=24)
    except Exception as exc:
        logger.error(f"scheduled job hard_delete_tombstones failed: {exc}", exc_info=True)


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

    scheduler: AsyncIOScheduler | None = None
    try:
        # Schedule background freshness/eviction maintenance jobs.
        scheduler = AsyncIOScheduler(timezone="UTC")
        scheduler.add_job(
            _job_compute_value_scores,
            IntervalTrigger(hours=1, jitter=60),
            kwargs={"crawler": crawler},
            id="compute_value_scores_hourly",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            _job_recrawl_due_sources,
            IntervalTrigger(hours=1, jitter=60),
            kwargs={"crawler": crawler},
            id="recrawl_due_sources_hourly",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            _job_prune_stale_content,
            IntervalTrigger(hours=6, jitter=90),
            kwargs={"crawler": crawler},
            id="prune_stale_content_6h",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            _job_enforce_storage_budget,
            IntervalTrigger(minutes=15, jitter=30),
            kwargs={"crawler": crawler},
            id="enforce_storage_budget_15m",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            _job_hard_delete_tombstones,
            IntervalTrigger(hours=1, jitter=60),
            kwargs={"crawler": crawler},
            id="hard_delete_tombstones_hourly",
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        logger.info("APScheduler started with 5 maintenance jobs.")

        yield AppContext(crawler=crawler, scheduler=scheduler)
    finally:
        if scheduler is not None:
            logger.info("Shutting down APScheduler...")
            scheduler.shutdown(wait=False)
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
mcp.tool()(tool_definitions.crawl_url)
mcp.tool()(tool_definitions.crawl_to_markdown)
mcp.tool()(tool_definitions.crawl_many_urls)
mcp.tool()(tool_definitions.crawl_deep)
mcp.tool()(tool_definitions.crawl_adaptive)
mcp.tool()(tool_definitions.crawl_with_session)
mcp.tool()(tool_definitions.create_session)
mcp.tool()(tool_definitions.kill_session)
mcp.tool()(tool_definitions.inspect_session)
mcp.tool()(tool_definitions.crawl_with_browser_config)
mcp.tool()(tool_definitions.crawl_local_file)
mcp.tool()(tool_definitions.crawl_raw_html)
mcp.tool()(tool_definitions.extract_markdown_variants)
mcp.tool()(tool_definitions.extract_fit_markdown)
mcp.tool()(tool_definitions.extract_structured_json)
mcp.tool()(tool_definitions.extract_regex_entities)
mcp.tool()(tool_definitions.extract_knowledge_graph)
mcp.tool()(tool_definitions.extract_code_examples)
mcp.tool()(tool_definitions.index_markdown)
mcp.tool()(tool_definitions.index_fit_markdown)
mcp.tool()(tool_definitions.index_structured_content)
mcp.tool()(tool_definitions.index_code_examples)
mcp.tool(name="search_documents")(tool_definitions.search_documents_v2)
mcp.tool()(tool_definitions.search_structured_content)
mcp.tool()(tool_definitions.get_fit_markdown_by_url)
mcp.tool()(tool_definitions.smart_crawl_url)
mcp.tool()(tool_definitions.get_available_sources)
mcp.tool()(tool_definitions.perform_rag_query)
mcp.tool()(tool_definitions.search_documents_tool)
mcp.tool()(tool_definitions.get_document_by_id)
mcp.tool()(tool_definitions.get_markdown_by_url)

if settings.USE_AGENTIC_RAG:
    mcp.tool()(tool_definitions.search_code_examples)

# Freshness/eviction maintenance remains scheduler-driven and is intentionally
# not exposed as MCP admin/ops endpoints.


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
