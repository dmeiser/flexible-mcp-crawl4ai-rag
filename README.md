[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# flexible-mcp-crawl4ai-rag

A **Crawl4AI-backed RAG wrapper** exposed via MCP.

This project is not just a crawler. It is a composable platform for:
- acquisition (crawl/deep/adaptive/session/local/raw)
- transformation (raw/fit/cited markdown + structured extraction)
- indexing (pgvector + metadata-rich variants)
- retrieval (representation-aware search + freshness controls)

## What this project is (and is not)

- ✅ It is a production-minded MCP layer around Crawl4AI primitives with RAG indexing/retrieval.
- ✅ It is scheduler-driven for freshness/eviction maintenance.
- ❌ It does **not** expose MCP admin/ops endpoints for storage maintenance.

Freshness and storage lifecycle operations run in background scheduler jobs only.

## Quick start

### Docker compose (recommended)

1. Copy env template and edit values:
   - `cp .env.example .env`
2. Start stack:
   - `docker compose up -d`
3. MCP SSE endpoint:
   - `http://localhost:8051/sse`

### Local run (uv)

1. Create venv and install:
   - `uv venv`
   - `source .venv/bin/activate`
   - `uv pip install -e .`
2. Run server:
   - `uv run src/crawl4ai_mcp.py`

## Tool categories

### Acquisition
- `crawl_url`
- `crawl_to_markdown`
- `crawl_many_urls`
- `crawl_deep`
- `crawl_adaptive`
- `crawl_with_session`, `create_session`, `inspect_session`, `kill_session`
- `crawl_with_auth_hooks`, `crawl_login_required`, `crawl_paginated`
- `crawl_local_file`, `crawl_raw_html`, `ingest_content_directory`

### Transformation / extraction
- `extract_markdown_variants`
- `extract_fit_markdown`
- `extract_structured_json`
- `extract_regex_entities`
- `extract_knowledge_graph`
- `extract_code_examples`
- `generate_extraction_schema`, `validate_extraction_schema`

### Indexing
- `index_markdown`
- `index_fit_markdown`
- `index_structured_content`
- `index_code_examples`

### Retrieval
- `search_documents`
- `search_raw_markdown`
- `search_fit_markdown`
- `search_structured_content`
- `search_code_examples` (feature-flagged)
- `get_document_by_id`
- `get_markdown_by_url`
- `get_fit_markdown_by_url`

## Safe defaults vs advanced options

### Safe defaults
- headless crawling
- no admin/ops lifecycle endpoints
- scheduler-driven freshness/eviction
- deterministic extraction schema validation
- offline deterministic unit test suite by default

### Advanced options
- crawl run config allowlist (`cache_mode`, JS hooks, screenshot/pdf/mhtml, etc.)
- browser config allowlist (`headers`, `cookies`, stealth, viewport, etc.)
- deep crawl strategy and filter chains
- adaptive crawl strategy selection and export modes
- representation filters at retrieval time

## Markdown vs fit markdown vs structured extraction

- **raw markdown**: best for completeness and archival provenance.
- **fit markdown**: best for tighter RAG context density.
- **structured extraction**: best for schema-driven data access and hybrid JSON+vector use cases.

Recommended approach:
- index both raw + fit when feasible
- use structured extraction when queries depend on stable fields/records

## Storage model and retrieval modes

Primary persisted entities:
- `crawled_pages`
- `code_examples`
- `source_policies`
- `storage_policies`
- `eviction_audit_log`

Freshness metadata includes:
- `first_seen_at`, `last_seen_at`, `last_crawled_at`
- `expires_at`, `staleness_score`
- `content_hash`, `source_change_id`
- `is_active`, `tombstoned_at`

Retrieval supports:
- vector search with optional hybrid behavior
- freshness-aware filters/reranking
- representation-specific search wrappers

## Testing

### Unit tests
- `make test`
- targeted: `pytest -q tests/crawler/test_tool_definitions.py`

### Integration / e2e smoke
- `MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py`
- strict rollout: `EXPECT_NEW_TOOLS=true MCP_URL=http://localhost:8051/sse uv run python tests/integration_smoke.py`

## Migration and cookbook

- Migration guidance: `docs/MIGRATION.md`
- Cookbook workflows: `docs/COOKBOOK.md`

## Project identity summary

When using this repo, think:

> “Crawl4AI acquisition + transformation primitives wrapped by MCP, with pgvector-backed variant-aware RAG retrieval and scheduler-driven freshness lifecycle.”
