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
7. [LLM for contextual embeddings](#llm-for-contextual-embeddings)
8. [Server](#server)
9. [Re-embedding after a model change](#re-embedding-after-a-model-change)

---

## Quick reference table

| Variable | Default | Required | Description |
|---|---|---|---|
| `POSTGRES_URL` | — | **yes** | Full SQLAlchemy DSN for the PostgreSQL+pgvector database |
| `EMBEDDING_PROVIDER` | `ollama` | no | `ollama` or `openai` |
| `EMBEDDING_DIM` | `768` | no | Vector dimension; must match the chosen model |
| `OLLAMA_API_URL` | `http://localhost:11434/api/embeddings` | no | Ollama `/api/embeddings` endpoint |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | no | Ollama model name |
| `OLLAMA_MAX_RETRIES` | `3` | no | Retry attempts on Ollama network error |
| `OLLAMA_RETRY_DELAY_SECONDS` | `1.0` | no | Seconds between retry attempts |
| `OPENAI_API_KEY` | — | if `EMBEDDING_PROVIDER=openai` | OpenAI API key |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | no | OpenAI embeddings model |
| `OPENAI_BASE_URL` | — | no | Override base URL for OpenAI-compatible endpoints |
| `BATCH_SIZE` | `50` | no | Chunks embedded per batch |
| `CHUNK_SIZE` | `1000` | no | Target tokens per chunk |
| `CHUNK_OVERLAP` | `200` | no | Overlap tokens between adjacent chunks |
| `CHUNK_STRATEGY` | `paragraph` | no | `paragraph` or `fixed` |
| `MARKDOWN_INDEX_POLICY` | `both-by-default` | no | `both-by-default`, `raw-only`, or `fit-only` |
| `MARKDOWN_FALLBACK_ENABLED` | `true` | no | Fall back to raw markdown when fit markdown is empty |
| `USE_CONTEXTUAL_EMBEDDINGS` | `false` | no | Prepend LLM-generated context to each chunk before embedding |
| `USE_HYBRID_SEARCH` | `false` | no | Combine vector + full-text search (BM25/tsvector) |
| `USE_AGENTIC_RAG` | `false` | no | Expose the agentic RAG tool |
| `USE_RERANKING` | `false` | no | Cross-encoder re-ranking pass on retrieved results |
| `LLM_ENABLED` | `false` | no | Enable LLM backend (required for contextual embeddings) |
| `LLM_API_KEY` | — | if `LLM_ENABLED=true` | API key for the contextual-embedding LLM |
| `LLM_BASE_URL` | — | if `LLM_ENABLED=true` | Base URL of the LLM API (e.g. OpenRouter) |
| `LLM_MODEL_NAME` | — | if `LLM_ENABLED=true` | Model identifier (e.g. `anthropic/claude-3-haiku`) |
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

### Bundled Ollama (default)

```env
EMBEDDING_PROVIDER=ollama
OLLAMA_API_URL=http://ollama:11434/api/embeddings
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

The `docker-compose.yml` starts an `ollama` container and pulls `nomic-embed-text` automatically on first run (~274 MB, cached in the `ollama_data` volume). Expect up to two minutes on the first `make up`.

### External Ollama instance

To use an Ollama server running on the host machine or elsewhere, override `OLLAMA_API_URL`:

```env
EMBEDDING_PROVIDER=ollama
OLLAMA_API_URL=http://host.docker.internal:11434/api/embeddings   # from inside a container
# or
OLLAMA_API_URL=http://192.168.1.50:11434/api/embeddings           # remote host
OLLAMA_EMBED_MODEL=nomic-embed-text
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
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_DIM=1536
```

`EMBEDDING_API_KEY` is validated at startup when `EMBEDDING_PROVIDER=openai`. `EMBEDDING_DIM` must match the model output dimension:

| Model | Dimension |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |

### OpenAI-compatible endpoint (Ollama v0.1.28+)

Ollama exposes an OpenAI-compatible endpoint at `/v1` alongside the native `/api/embeddings` endpoint. This is useful to use the OpenAI client library with a local model:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=ollama          # Ollama ignores the key; any non-empty value works
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_DIM=768
```

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
```

| Flag | Effect when `true` |
|---|---|
| `USE_CONTEXTUAL_EMBEDDINGS` | A short LLM-generated context summary is prepended to each chunk before embedding. Requires `LLM_ENABLED=true`. Improves recall at the cost of additional LLM calls per chunk at index time. |
| `USE_HYBRID_SEARCH` | Vector similarity search is combined with full-text (tsvector/BM25) search. Improves recall for keyword-heavy queries. The `fts` column is generated automatically; no extra setup required. |
| `USE_AGENTIC_RAG` | Registers the agentic RAG MCP tool (`search_agentic_rag`). Off by default to avoid exposing a broad-access tool unintentionally. |
| `USE_RERANKING` | Applies a cross-encoder re-ranking pass to returned results. Improves precision. Requires a compatible re-ranker model to be available (currently uses the embedding provider). |

---

## LLM for contextual embeddings

Required only when `USE_CONTEXTUAL_EMBEDDINGS=true`.

```env
LLM_ENABLED=true
LLM_API_KEY=<key>
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=anthropic/claude-3-haiku
```

All three `LLM_*` fields are validated at startup when `LLM_ENABLED=true`; the server will refuse to start if any is missing.

`LLM_BASE_URL` accepts any OpenAI-compatible API endpoint, including:

- [OpenRouter](https://openrouter.ai) — `https://openrouter.ai/api/v1`
- Direct OpenAI — `https://api.openai.com/v1`
- Local LM Studio — `http://localhost:1234/v1`
- Ollama OpenAI endpoint — `http://localhost:11434/v1`

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
