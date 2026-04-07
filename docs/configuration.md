# Configuration Reference

All runtime behaviour is controlled through environment variables, read at startup from the `.env` file in the project root (or from the process environment).

Copy the provided template and edit it before the first run:

```bash
cp .env.example .env
```

---

## Table of contents

1. [Quick reference table](#quick-reference-table)
2. [Database](#database)
   - [Bundled pgvector (Mode A)](#bundled-pgvector-mode-a)
   - [External PostgreSQL (Mode B)](#external-postgresql-mode-b)
   - [Bare-metal / uv local run](#bare-metal--uv-local-run)
3. [Embedding provider](#embedding-provider)
   - [Bundled Ollama (default)](#bundled-ollama-default)
   - [External Ollama instance](#external-ollama-instance)
   - [Changing the Ollama model](#changing-the-ollama-model)
   - [OpenAI](#openai)
   - [OpenAI-compatible endpoint (Ollama v0.1.28+)](#openai-compatible-endpoint-ollama-v0128)
4. [Chunking](#chunking)
5. [Markdown indexing policy](#markdown-indexing-policy)
6. [RAG feature flags](#rag-feature-flags)
7. [Default and per-feature LLM settings](#default-and-per-feature-llm-settings)
8. [Server](#server)
9. [Re-embedding after a model change](#re-embedding-after-a-model-change)

---

## Quick reference table

| Variable | Default | Required | Description |
|---|---|---|---|
| `POSTGRES_URL` | — | **yes** | Full SQLAlchemy DSN for the PostgreSQL+pgvector database |
| `EMBEDDING_DIM` | `768` | no | Vector dimension; must match the chosen model |
| `EMBEDDING_BASE_URL` | — | no | Base URL for the OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | — | no | API key for the embedding provider (any non-empty value for Ollama) |
| `EMBEDDING_MODEL_NAME` | provider-dependent | no | Embedding model name |
| `EMBEDDING_MAX_RETRIES` | `3` | no | Retry attempts on embedding request failure |
| `EMBEDDING_RETRY_DELAY_SECONDS` | `1.0` | no | Seconds between embedding retries |
| `BATCH_SIZE` | `50` | no | Chunks embedded per batch |
| `CHUNK_SIZE` | `1000` | no | Target tokens per chunk |
| `CHUNK_OVERLAP` | `200` | no | Overlap tokens between adjacent chunks |
| `CHUNK_STRATEGY` | `paragraph` | no | `paragraph` or `fixed` |
| `MARKDOWN_INDEX_POLICY` | `both-by-default` | no | `both-by-default`, `raw-only`, or `fit-only` |
| `MARKDOWN_FALLBACK_ENABLED` | `true` | no | Fall back to raw markdown when fit markdown is empty |
| `USE_CONTEXTUAL_EMBEDDINGS` | `false` | no | Prepend LLM-generated context to each chunk before embedding |
| `USE_HYBRID_SEARCH` | `false` | no | Combine vector + full-text search (BM25/tsvector) |
| `USE_AGENTIC_RAG` | `false` | no | Expose the feature-flagged `search_code_examples` tool |
| `USE_RERANKING` | `false` | no | Cross-encoder re-ranking pass on retrieved results |
| `USE_WEB_SEARCH` | `false` | no | Expose the feature-flagged `search_web` tool backed by OpenRouter web search |
| `DEFAULT_LLM_PROVIDER` | `openai` | no | Shared fallback provider for LLM-powered features |
| `DEFAULT_LLM_BASE_URL` | — | no | Shared fallback base URL for OpenAI-compatible LLM endpoints |
| `DEFAULT_LLM_API_KEY` | — | no | Shared fallback API key for LLM-powered features |
| `DEFAULT_LLM_MODEL_NAME` | — | no | Shared fallback model name for LLM-powered features |
| `CONTEXTUAL_LLM_*` | — | no | Optional overrides used by contextual embeddings |
| `AGENTIC_LLM_*` | — | no | Preferred shared override bucket used by LLM-based filtering/extraction helpers and agentic-style features |
| `RERANK_LLM_*` | see `.env.example` | no | Overrides for reranking; may target a local cross-encoder or an OpenAI-compatible scorer |
| `WEB_SEARCH_*` | see `.env.example` | no | Provider-dispatched web search configuration, request defaults, and optional short-TTL cache settings |
| `TRANSPORT` | `sse` | no | MCP transport: `sse` or `stdio` |
| `HOST` | `0.0.0.0` | no | Bind address for the SSE server |
| `PORT` | `8051` | no | Port for the SSE server |
| `LOG_LEVEL` | `INFO` | no | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

> **`EMBEDDING_DIM` must match the model.** The value is baked into the pgvector column when the database is first initialised. Changing it afterwards requires a database migration or a fresh database. See [Re-embedding after a model change](#re-embedding-after-a-model-change).

---

## Database

### Bundled pgvector (Mode A)

The default `docker compose up` starts a pgvector container alongside the app. No extra configuration is needed beyond the values already in `.env.example`. The app container receives `POSTGRES_URL` automatically via the `environment:` block in `docker-compose.yml`.

```bash
docker compose up -d
```

The pgvector container exposes `5434` on the host (mapped to `5432` inside) for direct inspection:

```bash
psql -h localhost -p 5434 -U raguser -d ragdb
```

Schema initialisation is handled automatically by `initdb/init.sh`, which substitutes the `EMBEDDING_DIM` placeholder at first startup.

### External PostgreSQL (Mode B)

Use an existing PostgreSQL 14+ instance with the `pgvector` and `pg_trgm` extensions installed.

**Requirements:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**Steps:**

1. Apply the schema manually (substitute `768` or your chosen dimension):

   ```bash
   EMBEDDING_DIM=768 envsubst < initdb/setup.sql.in | psql "$POSTGRES_URL"
   ```

2. Set `POSTGRES_URL` in `.env`:

   ```env
   POSTGRES_URL=postgresql://user:password@host:5432/dbname
   ```

3. Start with the external profile — this uses the `docs-rag-mcp-ext` service, which has no `depends_on` the local DB:

   ```bash
   docker compose --profile external up -d
   ```

   Or for a local `uv` run, the `POSTGRES_URL` in `.env` is used directly:

   ```bash
   uv run src/crawl4ai_mcp.py
   ```

> **Cloud databases:** `POSTGRES_URL` accepts any SQLAlchemy-compatible DSN, including managed services such as Supabase, Neon, AlloyDB, and RDS — as long as the `vector` and `pg_trgm` extensions are available.

### Bare-metal / uv local run

No Docker required. Point `POSTGRES_URL` at any accessible PostgreSQL instance and run:

```bash
uv run src/crawl4ai_mcp.py
```

---

## Embedding provider

All embedding providers are accessed through the OpenAI-compatible client. Set `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` to point at any compatible endpoint.

### Bundled Ollama (default)

```env
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_DIM=768
```

The `docker-compose.yml` starts an `ollama` container, pulls `nomic-embed-text` automatically on first run (~274 MB, cached in the `ollama_data` volume), and overrides `EMBEDDING_BASE_URL` to `http://ollama:11434/v1` inside the container. Expect up to two minutes on the first `make up`.

### External Ollama instance

To use an Ollama server running on the host machine or elsewhere, override `EMBEDDING_BASE_URL`:

```env
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1   # from inside a container
# or
EMBEDDING_BASE_URL=http://192.168.1.50:11434/v1           # remote host
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_DIM=768
```

When using an external Ollama, the `ollama` service in `docker-compose.yml` is still started by default. To skip it, use the `external` profile (which starts `docs-rag-mcp-ext` with no bundled services):

```bash
docker compose --profile external up -d
```

Or comment out / remove the `ollama` service from `docker-compose.yml` for a permanent change.

### Changing the Ollama model

1. Pull the new model on your Ollama instance:

   ```bash
   ollama pull mxbai-embed-large
   ```

2. Update `.env` with the new model name and its vector dimension:

   ```env
   EMBEDDING_MODEL_NAME=mxbai-embed-large
   EMBEDDING_DIM=1024
   ```

3. **Re-initialise the database** if `EMBEDDING_DIM` changed (the vector column dimension is fixed at schema creation time). Either:
   - Drop and recreate the database, then re-apply `initdb/setup.sql.in`; or
   - Keep the existing database and re-embed all stored documents after restarting the app (see [Re-embedding after a model change](#re-embedding-after-a-model-change)).

   > If the dimension does **not** change (e.g. switching between two 768-d models), no schema change is needed — only a re-embedding run is required so that stored vectors are consistent with the new model.

**Common model dimensions:**

| Model | Dimension |
|---|---|
| `nomic-embed-text` (default) | 768 |
| `mxbai-embed-large` | 1024 |
| `snowflake-arctic-embed` | 1024 |
| `all-minilm` | 384 |
| `bge-m3` | 1024 |

### OpenAI

```env
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_DIM=1536
```

`EMBEDDING_DIM` must match the model output dimension:

| Model | Dimension |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |

### Other OpenAI-compatible endpoints

`EMBEDDING_BASE_URL` overrides the base URL used by the `openai` Python client, so any OpenAI-compatible server (LM Studio, vLLM, Azure OpenAI, etc.) works the same way.

---

## Chunking

```env
CHUNK_STRATEGY=paragraph   # paragraph | fixed
CHUNK_SIZE=1000             # tokens per chunk (target for paragraph; exact for fixed)
CHUNK_OVERLAP=200           # overlap tokens between consecutive chunks
BATCH_SIZE=50               # chunks embedded per batch (tune for memory / throughput)
```

**Strategies:**

- `paragraph` — splits on paragraph boundaries; `CHUNK_SIZE` is a soft target. Recommended for most content.
- `fixed` — hard token splits at exactly `CHUNK_SIZE` with `CHUNK_OVERLAP` overlap. Use for content without clear paragraph structure.

Per-request overrides of `chunk_strategy` are accepted by crawl and index tools when the value is in the server allowlist.

---

## Markdown indexing policy

Controls which markdown representation is stored when using `index_markdown`:

```env
MARKDOWN_INDEX_POLICY=both-by-default   # both-by-default | raw-only | fit-only
MARKDOWN_FALLBACK_ENABLED=true          # fall back to raw when fit markdown is empty
```

| Policy | Behaviour |
|---|---|
| `both-by-default` | Indexes both raw and fit markdown. Recommended; enables all search tools. |
| `raw-only` | Indexes raw markdown only. Reduces storage. `search_fit_markdown` returns nothing. |
| `fit-only` | Indexes fit markdown only. `search_raw_markdown` returns nothing. |

---

## RAG feature flags

All flags default to `false`.

```env
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_WEB_SEARCH=false
```

| Flag | Effect when `true` |
|---|---|
| `USE_CONTEXTUAL_EMBEDDINGS` | A short LLM-generated context summary is prepended to each chunk before embedding. Requires `CONTEXTUAL_LLM_MODEL_NAME` or `DEFAULT_LLM_MODEL_NAME` to be configured. Improves recall at the cost of additional LLM calls per chunk at index time. |
| `USE_HYBRID_SEARCH` | Vector similarity search is combined with full-text (tsvector/BM25) search. Improves recall for keyword-heavy queries. The `fts` column is generated automatically; no extra setup required. |
| `USE_AGENTIC_RAG` | Registers the feature-flagged MCP tool `search_code_examples`. Off by default to avoid exposing an extra retrieval surface unintentionally. |
| `USE_RERANKING` | Applies a cross-encoder re-ranking pass to returned results. Improves precision. Requires a compatible re-ranker model to be available (currently uses the embedding provider). |
| `USE_WEB_SEARCH` | Registers the feature-flagged MCP tool `search_web`, which calls OpenRouter's `openrouter:web_search` server tool. Optional caching stores short-lived web-lead records for crawl/recrawl workflows, not normal document retrieval. |

---

## Default and per-feature LLM settings

LLM-powered features use a shared fallback plus optional per-feature overrides.

- `DEFAULT_LLM_*` provides the common provider/base URL/API key/model/retry settings.
- `CONTEXTUAL_LLM_*` overrides are used by contextual embeddings.
- `AGENTIC_LLM_*` is the preferred shared override bucket currently used by LLM-based content filtering and extraction helpers, and is also available to agentic-style features.
- `RERANK_LLM_*` overrides are used only by reranking.
- `WEB_SEARCH_*` configures the separate provider-dispatched live web search tool and its optional ephemeral cache.

There is no separate `HYBRID_LLM_*` setting family; hybrid retrieval is vector search + PostgreSQL full-text search.

### Shared/default LLM example

Use this when you want one OpenAI-compatible LLM configuration reused across features:

```env
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_LLM_API_KEY=<key>
DEFAULT_LLM_MODEL_NAME=anthropic/claude-3-haiku
DEFAULT_LLM_MAX_RETRIES=3
DEFAULT_LLM_RETRY_DELAY_SECONDS=1.0
```

### Contextual embeddings override example

Required only when `USE_CONTEXTUAL_EMBEDDINGS=true` and you do not want to rely on `DEFAULT_LLM_*`.

```env
USE_CONTEXTUAL_EMBEDDINGS=true
CONTEXTUAL_LLM_PROVIDER=openai
CONTEXTUAL_LLM_BASE_URL=https://api.openai.com/v1
CONTEXTUAL_LLM_API_KEY=<key>
CONTEXTUAL_LLM_MODEL_NAME=gpt-4o-mini
```

### Agentic/shared override example

`AGENTIC_LLM_*` does not drive hybrid retrieval and does not power `search_code_examples` directly. Today it is the preferred override bucket for LLM-based content filtering/extraction helpers, and can also be used for agentic-style features.

```env
AGENTIC_LLM_PROVIDER=ollama
AGENTIC_LLM_BASE_URL=http://ollama:11434/v1
AGENTIC_LLM_API_KEY=ollama
AGENTIC_LLM_MODEL_NAME=llama3.1:8b
```

### Reranking override example

When `RERANK_LLM_BASE_URL` is unset, `RERANK_LLM_MODEL_NAME` is treated as a local sentence-transformers `CrossEncoder` model name. When `RERANK_LLM_BASE_URL` is set, reranking uses an OpenAI-compatible scoring endpoint.

```env
USE_RERANKING=true
RERANK_LLM_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
```

`*_BASE_URL` accepts any OpenAI-compatible API endpoint, including:

- [OpenRouter](https://openrouter.ai) — `https://openrouter.ai/api/v1`
- Direct OpenAI — `https://api.openai.com/v1`
- Local LM Studio — `http://localhost:1234/v1`
- Ollama OpenAI endpoint — `http://localhost:11434/v1`

### Web search

`search_web` is a separate MCP tool with a pluggable provider model. Today the supported provider is `openrouter`; the settings surface is structured so future providers such as Brave or DuckDuckGo can be added behind the same tool contract.

Current provider selection:

```env
WEB_SEARCH_PROVIDER=openrouter
```

With `WEB_SEARCH_PROVIDER=openrouter`, the tool performs live web queries through OpenRouter's current server-tool interface:

```json
{
   "tools": [
      {
         "type": "openrouter:web_search",
         "parameters": {
            "engine": "auto",
            "max_results": 5
         }
      }
   ]
}
```

Example configuration:

```env
USE_WEB_SEARCH=true
WEB_SEARCH_PROVIDER=openrouter
WEB_SEARCH_BASE_URL=https://openrouter.ai/api/v1
WEB_SEARCH_API_KEY=<key>
WEB_SEARCH_MODEL_NAME=openrouter/perplexity/sonar
WEB_SEARCH_DEFAULT_ENGINE=auto
WEB_SEARCH_DEFAULT_MAX_RESULTS=5
```

Optional ephemeral cache:

```env
WEB_SEARCH_CACHE_ENABLED=true
WEB_SEARCH_CACHE_TTL_HOURS=24
WEB_SEARCH_CACHE_SOURCE=openrouter_web_search
```

Cached web-search rows are intentionally stored outside normal retrieval flow:

- they are persisted as short-lived crawl/recrawl leads,
- they are marked inactive so `search_documents` does not return them by default,
- they carry explicit expiry metadata and are pruned automatically on later cache writes.

As additional providers are implemented later, they should plug into the same `search_web` tool and `WEB_SEARCH_PROVIDER` switch rather than creating parallel MCP tools.

---

## Server

```env
TRANSPORT=sse      # sse | stdio
HOST=0.0.0.0
PORT=8051
LOG_LEVEL=INFO     # DEBUG | INFO | WARNING | ERROR
```

- **`TRANSPORT=sse`** — exposes the MCP endpoint at `http://<HOST>:<PORT>/sse`. This is the standard mode for use with MCP clients (Claude Desktop, Cursor, etc.).
- **`TRANSPORT=stdio`** — reads/writes MCP messages over stdin/stdout. Used for subprocess-based MCP integration.

The healthcheck in `docker-compose.yml` uses the SSE endpoint; if you change `TRANSPORT`, update the healthcheck accordingly.

---

## Re-embedding after a model change

If you switch embedding models (or change `EMBEDDING_DIM`), existing stored vectors will be mismatched and search quality will degrade significantly. Re-embed all stored documents with the new model:

```bash
# With Docker Compose running:
docker compose exec docs-rag-mcp uv run python -m src.reembed_documents

# Or locally:
uv run python -m src.reembed_documents
```

The script re-embeds all rows in `crawled_pages` and `code_examples` and commits the updated vectors. It logs progress to stdout. Re-run it after every model change.

> **Schema constraint:** if `EMBEDDING_DIM` changed, you must reinitialise the database schema before running the re-embed script, because the `VECTOR(n)` column type is fixed at creation time. Easiest approach is `make clean` (removes volumes) followed by `make up` to recreate the schema with the new dimension, then re-crawl or re-index content.
