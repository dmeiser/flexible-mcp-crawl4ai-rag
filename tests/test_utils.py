"""Unit tests for src/utils.py — 100% coverage, all offline."""
import asyncio
import json
import builtins
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

import os

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")
os.environ.setdefault("OLLAMA_MAX_RETRIES", "2")
os.environ.setdefault("OLLAMA_RETRY_DELAY_SECONDS", "0.01")
os.environ.setdefault("BATCH_SIZE", "10")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("USE_CONTEXTUAL_EMBEDDINGS", "false")
os.environ.setdefault("USE_HYBRID_SEARCH", "false")
os.environ.setdefault("USE_AGENTIC_RAG", "false")
os.environ.setdefault("USE_RERANKING", "false")

from src.utils import (
    Settings,
    get_session,
    EmbeddingProvider,
    EmbeddingError,
    OllamaError,
    ChunkStrategy,
    _normalize,
    create_embedding,
    _create_ollama_embedding,
    _create_openai_embedding,
    create_embeddings_batch,
    generate_contextual_text,
    upsert_source,
    add_documents_to_db,
    add_code_examples_to_db,
    search_documents,
    _python_side_vector_search,
    search_code_examples,
    rerank_results,
    extract_code_blocks,
    settings,
    CrawledPage,
    CodeExample,
    Source,
)

DIM = 4


def _vec(*values):
    v = list(values) + [0.0] * (DIM - len(values))
    return _normalize(v[:DIM])


EMBED_A = _vec(1.0, 0.0, 0.0, 0.0)
EMBED_B = _vec(0.0, 1.0, 0.0, 0.0)
EMBED_Q = _vec(1.0, 0.0, 0.0, 0.0)


def _fake_settings(**kw):
    defaults = dict(
        POSTGRES_URL="postgresql://u:p@localhost:5432/testdb",
        EMBEDDING_PROVIDER=EmbeddingProvider.OLLAMA,
        EMBEDDING_DIM=DIM,
        OLLAMA_API_URL="http://localhost:11434/api/embeddings",
        OLLAMA_EMBED_MODEL="nomic-embed-text",
        OLLAMA_MAX_RETRIES=2,
        OLLAMA_RETRY_DELAY_SECONDS=0.001,
        OPENAI_API_KEY=None,
        OPENAI_EMBED_MODEL="text-embedding-3-small",
        OPENAI_BASE_URL=None,
        BATCH_SIZE=10,
        CHUNK_SIZE=100,
        CHUNK_OVERLAP=10,
        CHUNK_STRATEGY=ChunkStrategy.PARAGRAPH,
        USE_CONTEXTUAL_EMBEDDINGS=False,
        USE_HYBRID_SEARCH=False,
        USE_AGENTIC_RAG=False,
        USE_RERANKING=False,
        LLM_ENABLED=False,
        LLM_API_KEY=None,
        LLM_BASE_URL=None,
        LLM_MODEL_NAME=None,
    )
    defaults.update(kw)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# Tests: EmbeddingError / OllamaError alias
# ---------------------------------------------------------------------------

def test_ollama_error_is_embedding_error():
    assert OllamaError is EmbeddingError


def test_embedding_error_raise():
    with pytest.raises(EmbeddingError):
        raise OllamaError("test")


# ---------------------------------------------------------------------------
# Tests: ChunkStrategy / EmbeddingProvider enums
# ---------------------------------------------------------------------------

def test_chunk_strategy_values():
    assert ChunkStrategy.PARAGRAPH == "paragraph"
    assert ChunkStrategy.SENTENCE == "sentence"
    assert ChunkStrategy.FIXED == "fixed"
    assert ChunkStrategy.SEMANTIC == "semantic"


def test_embedding_provider_values():
    assert EmbeddingProvider.OPENAI == "openai"
    assert EmbeddingProvider.OLLAMA == "ollama"


# ---------------------------------------------------------------------------
# Tests: Settings validation
# ---------------------------------------------------------------------------

class TestSettingsValidation:
    def test_llm_enabled_missing_api_key_raises(self):
        with pytest.raises(Exception, match="LLM_API_KEY"):
            Settings(
                POSTGRES_URL="postgresql://u:p@h/db",
                EMBEDDING_PROVIDER="ollama",
                OLLAMA_API_URL="http://localhost:11434/api/embeddings",
                OLLAMA_EMBED_MODEL="m",
                LLM_ENABLED=True,
                LLM_API_KEY=None,
                LLM_BASE_URL="http://llm",
                LLM_MODEL_NAME="model",
            )

    def test_llm_enabled_missing_base_url_raises(self):
        with pytest.raises(Exception, match="LLM_BASE_URL"):
            Settings(
                POSTGRES_URL="postgresql://u:p@h/db",
                EMBEDDING_PROVIDER="ollama",
                OLLAMA_API_URL="http://localhost:11434/api/embeddings",
                OLLAMA_EMBED_MODEL="m",
                LLM_ENABLED=True,
                LLM_API_KEY="key",
                LLM_BASE_URL=None,
                LLM_MODEL_NAME="model",
            )

    def test_llm_enabled_missing_model_name_raises(self):
        with pytest.raises(Exception, match="LLM_MODEL_NAME"):
            Settings(
                POSTGRES_URL="postgresql://u:p@h/db",
                EMBEDDING_PROVIDER="ollama",
                OLLAMA_API_URL="http://localhost:11434/api/embeddings",
                OLLAMA_EMBED_MODEL="m",
                LLM_ENABLED=True,
                LLM_API_KEY="key",
                LLM_BASE_URL="http://llm",
                LLM_MODEL_NAME=None,
            )

    def test_openai_missing_key_raises(self):
        with pytest.raises(Exception, match="OPENAI_API_KEY"):
            Settings(
                POSTGRES_URL="postgresql://u:p@h/db",
                EMBEDDING_PROVIDER="openai",
                OPENAI_API_KEY=None,
                OLLAMA_API_URL="http://localhost:11434/api/embeddings",
                OLLAMA_EMBED_MODEL="m",
            )

    def test_valid_settings_ok(self):
        s = Settings(
            POSTGRES_URL="postgresql://u:p@h/db",
            EMBEDDING_PROVIDER="ollama",
            OLLAMA_API_URL="http://localhost:11434/api/embeddings",
            OLLAMA_EMBED_MODEL="m",
        )
        assert s.EMBEDDING_PROVIDER == EmbeddingProvider.OLLAMA


# ---------------------------------------------------------------------------
# Tests: _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_unit_vector(self):
        v = [3.0, 4.0, 0.0, 0.0]
        n = _normalize(v)
        assert abs(n[0] - 0.6) < 1e-5
        assert abs(n[1] - 0.8) < 1e-5

    def test_zero_vector_returns_as_is(self):
        v = [0.0, 0.0, 0.0, 0.0]
        assert _normalize(v) == v


# ---------------------------------------------------------------------------
# Tests: create_embedding dispatch
# ---------------------------------------------------------------------------

class TestCreateEmbedding:
    @pytest.mark.asyncio
    async def test_empty_string_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            await create_embedding("")

    @pytest.mark.asyncio
    async def test_whitespace_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            await create_embedding("   ")

    @pytest.mark.asyncio
    async def test_routes_to_ollama(self):
        with patch("src.utils.settings", _fake_settings(EMBEDDING_PROVIDER=EmbeddingProvider.OLLAMA)), \
             patch("src.utils._create_ollama_embedding", new_callable=AsyncMock, return_value=EMBED_A) as mock_o:
            result = await create_embedding("hello")
        mock_o.assert_called_once_with("hello")
        assert result == EMBED_A

    @pytest.mark.asyncio
    async def test_routes_to_openai(self):
        with patch("src.utils.settings", _fake_settings(
            EMBEDDING_PROVIDER=EmbeddingProvider.OPENAI,
            OPENAI_API_KEY="sk-test",
        )), \
             patch("src.utils._create_openai_embedding", new_callable=AsyncMock, return_value=EMBED_A) as mock_oa:
            result = await create_embedding("hello")
        mock_oa.assert_called_once_with("hello")
        assert result == EMBED_A


# ---------------------------------------------------------------------------
# Tests: _create_ollama_embedding
# ---------------------------------------------------------------------------

class TestOllamaEmbedding:
    def _make_client(self, side_effect=None, return_value=None):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embedding": [1.0, 0.0, 0.0, 0.0]}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        if side_effect is not None:
            mock_client.post = AsyncMock(side_effect=side_effect)
        else:
            mock_client.post = AsyncMock(return_value=return_value or mock_resp)
        return mock_client

    @pytest.mark.asyncio
    async def test_success(self):
        raw = [1.0, 0.0, 0.0, 0.0]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embedding": raw}
        mc = self._make_client(return_value=mock_resp)

        with patch("src.utils.settings", _fake_settings()), \
             patch("src.utils.httpx.AsyncClient", return_value=mc):
            result = await _create_ollama_embedding("test")
        assert result == _normalize(raw)

    @pytest.mark.asyncio
    async def test_retry_on_timeout_then_success(self):
        import httpx as httpx_lib
        raw = [1.0, 0.0, 0.0, 0.0]
        success_resp = MagicMock()
        success_resp.raise_for_status = MagicMock()
        success_resp.json.return_value = {"embedding": raw}
        mc = self._make_client(side_effect=[
            httpx_lib.TimeoutException("timeout"),
            success_resp,
        ])

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=2, OLLAMA_RETRY_DELAY_SECONDS=0.001)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc), \
             patch("src.utils.asyncio.sleep", new_callable=AsyncMock):
            result = await _create_ollama_embedding("test")
        assert result == _normalize(raw)

    @pytest.mark.asyncio
    async def test_connect_error_retry_then_success(self):
        import httpx as httpx_lib
        raw = [0.0, 1.0, 0.0, 0.0]
        success_resp = MagicMock()
        success_resp.raise_for_status = MagicMock()
        success_resp.json.return_value = {"embedding": raw}
        mc = self._make_client(side_effect=[
            httpx_lib.ConnectError("conn refused"),
            success_resp,
        ])

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=2, OLLAMA_RETRY_DELAY_SECONDS=0.001)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc), \
             patch("src.utils.asyncio.sleep", new_callable=AsyncMock):
            result = await _create_ollama_embedding("test")
        assert result == _normalize(raw)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self):
        import httpx as httpx_lib
        mc = self._make_client(side_effect=httpx_lib.TimeoutException("timeout"))

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=2, OLLAMA_RETRY_DELAY_SECONDS=0.001)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc), \
             patch("src.utils.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(EmbeddingError):
                await _create_ollama_embedding("test")

    @pytest.mark.asyncio
    async def test_http_status_error_raises(self):
        import httpx as httpx_lib
        error_resp = MagicMock()
        error_resp.status_code = 500
        mc = self._make_client(side_effect=httpx_lib.HTTPStatusError(
            "error", request=MagicMock(), response=error_resp
        ))

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=1)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc):
            with pytest.raises(EmbeddingError):
                await _create_ollama_embedding("test")

    @pytest.mark.asyncio
    async def test_invalid_embedding_none_raises(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embedding": None}
        mc = self._make_client(return_value=mock_resp)

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=1)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc):
            with pytest.raises(EmbeddingError):
                await _create_ollama_embedding("test")

    @pytest.mark.asyncio
    async def test_invalid_embedding_non_list_raises(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embedding": "not-a-list"}
        mc = self._make_client(return_value=mock_resp)

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=1)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc):
            with pytest.raises(EmbeddingError):
                await _create_ollama_embedding("test")

    @pytest.mark.asyncio
    async def test_unexpected_exception_raises(self):
        mc = self._make_client(side_effect=RuntimeError("unexpected"))

        with patch("src.utils.settings", _fake_settings(OLLAMA_MAX_RETRIES=1)), \
             patch("src.utils.httpx.AsyncClient", return_value=mc):
            with pytest.raises(EmbeddingError):
                await _create_ollama_embedding("test")


# ---------------------------------------------------------------------------
# Tests: _create_openai_embedding
# ---------------------------------------------------------------------------

class TestOpenAIEmbedding:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=[1.0, 0.0, 0.0, 0.0])]
        ))
        mock_client.close = AsyncMock()

        with patch("src.utils.settings", _fake_settings(
            EMBEDDING_PROVIDER=EmbeddingProvider.OPENAI,
            OPENAI_API_KEY="sk-test",
            OPENAI_EMBED_MODEL="text-embedding-3-small",
            OPENAI_BASE_URL=None,
        )), \
             patch("src.utils.AsyncOpenAI", return_value=mock_client):
            result = await _create_openai_embedding("hello")
        assert result == _normalize([1.0, 0.0, 0.0, 0.0])

    @pytest.mark.asyncio
    async def test_with_custom_base_url(self):
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=[0.0, 1.0, 0.0, 0.0])]
        ))
        mock_client.close = AsyncMock()

        with patch("src.utils.settings", _fake_settings(
            EMBEDDING_PROVIDER=EmbeddingProvider.OPENAI,
            OPENAI_API_KEY="sk-test",
            OPENAI_BASE_URL="http://custom:11434",
        )), \
             patch("src.utils.AsyncOpenAI", return_value=mock_client) as MockOA:
            await _create_openai_embedding("hello")
        call_kwargs = MockOA.call_args[1]
        assert call_kwargs.get("base_url") == "http://custom:11434"

    @pytest.mark.asyncio
    async def test_api_failure_raises_embedding_error(self):
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("API down"))
        mock_client.close = AsyncMock()

        with patch("src.utils.settings", _fake_settings(
            EMBEDDING_PROVIDER=EmbeddingProvider.OPENAI,
            OPENAI_API_KEY="sk-test",
        )), \
             patch("src.utils.AsyncOpenAI", return_value=mock_client):
            with pytest.raises(EmbeddingError):
                await _create_openai_embedding("hello")


# ---------------------------------------------------------------------------
# Tests: create_embeddings_batch
# ---------------------------------------------------------------------------

class TestBatchEmbedding:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        result = await create_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_texts(self):
        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_A):
            result = await create_embeddings_batch(["a", "b", "c"])
        assert result == [EMBED_A, EMBED_A, EMBED_A]


# ---------------------------------------------------------------------------
# Tests: generate_contextual_text
# ---------------------------------------------------------------------------

class TestContextualText:
    @pytest.mark.asyncio
    async def test_disabled_returns_original(self):
        with patch("src.utils.settings", _fake_settings(
            USE_CONTEXTUAL_EMBEDDINGS=False, LLM_ENABLED=False
        )):
            text, enriched = await generate_contextual_text("doc", "chunk")
        assert text == "chunk"
        assert enriched is False

    @pytest.mark.asyncio
    async def test_llm_disabled_returns_original(self):
        with patch("src.utils.settings", _fake_settings(
            USE_CONTEXTUAL_EMBEDDINGS=True, LLM_ENABLED=False
        )):
            text, enriched = await generate_contextual_text("doc", "chunk")
        assert text == "chunk"
        assert enriched is False

    @pytest.mark.asyncio
    async def test_llm_enriches_text(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="great context"))]
        ))
        mock_client.close = AsyncMock()

        with patch("src.utils.settings", _fake_settings(
            USE_CONTEXTUAL_EMBEDDINGS=True,
            LLM_ENABLED=True,
            LLM_API_KEY="key",
            LLM_BASE_URL="http://llm",
            LLM_MODEL_NAME="model",
            CHUNK_SIZE=1000,
        )), \
             patch("src.utils.AsyncOpenAI", return_value=mock_client):
            text, enriched = await generate_contextual_text("full doc", "chunk text")
        assert enriched is True
        assert "great context" in text
        assert "chunk text" in text

    @pytest.mark.asyncio
    async def test_llm_empty_response_returns_original(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="  "))]
        ))
        mock_client.close = AsyncMock()

        with patch("src.utils.settings", _fake_settings(
            USE_CONTEXTUAL_EMBEDDINGS=True,
            LLM_ENABLED=True,
            LLM_API_KEY="key",
            LLM_BASE_URL="http://llm",
            LLM_MODEL_NAME="model",
        )), \
             patch("src.utils.AsyncOpenAI", return_value=mock_client):
            text, enriched = await generate_contextual_text("doc", "chunk")
        assert text == "chunk"
        assert enriched is False

    @pytest.mark.asyncio
    async def test_llm_exception_returns_original(self):
        with patch("src.utils.settings", _fake_settings(
            USE_CONTEXTUAL_EMBEDDINGS=True,
            LLM_ENABLED=True,
            LLM_API_KEY="key",
            LLM_BASE_URL="http://llm",
            LLM_MODEL_NAME="model",
        )), \
             patch("src.utils.AsyncOpenAI", side_effect=Exception("LLM down")):
            text, enriched = await generate_contextual_text("doc", "chunk")
        assert text == "chunk"
        assert enriched is False


# ---------------------------------------------------------------------------
# Tests: upsert_source
# ---------------------------------------------------------------------------

class TestUpsertSource:
    def test_existing_source_returns_id(self):
        session = MagicMock()
        existing = MagicMock(id=42)
        session.exec.return_value.first.return_value = existing
        assert upsert_source(session, "example.com") == 42
        session.add.assert_not_called()

    def test_new_source_is_created(self):
        session = MagicMock()
        session.exec.return_value.first.return_value = None

        def fake_refresh(obj):
            obj.id = 99

        session.refresh.side_effect = fake_refresh
        result = upsert_source(session, "new.com")
        session.add.assert_called_once()
        session.commit.assert_called_once()
        assert result == 99


# ---------------------------------------------------------------------------
# Tests: add_documents_to_db
# ---------------------------------------------------------------------------

class TestAddDocumentsToDb:
    @pytest.mark.asyncio
    async def test_empty_urls_returns_zero(self):
        session = MagicMock()
        result = await add_documents_to_db(session, [], [], [], [])
        assert result == 0
        session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_length_mismatch_returns_zero(self):
        session = MagicMock()
        result = await add_documents_to_db(
            session, ["url1", "url2"], ["content1"], [{}], [0]
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_whitespace_content_skipped(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings(USE_CONTEXTUAL_EMBEDDINGS=False, LLM_ENABLED=False)):
            result = await add_documents_to_db(
                session,
                ["url1", "url2"],
                ["valid content", "   "],
                [{"source": "x.com"}, {"source": "x.com"}],
                [0, 1],
            )
        assert result == 1

    @pytest.mark.asyncio
    async def test_normal_flow_stores_rows(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings(USE_CONTEXTUAL_EMBEDDINGS=False, LLM_ENABLED=False)):
            result = await add_documents_to_db(
                session, ["http://x.com/page"], ["some content"], [{"source": "x.com"}], [0]
            )
        assert result == 1
        session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_source_in_metadata_uses_none(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        added_rows = []

        def capture_add_all(rows):
            added_rows.extend(rows)

        session.add_all.side_effect = capture_add_all

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.settings", _fake_settings(USE_CONTEXTUAL_EMBEDDINGS=False, LLM_ENABLED=False)):
            result = await add_documents_to_db(
                session, ["url"], ["content"], [{}], [0]
            )
        assert result == 1
        assert added_rows[0].source_id is None

    @pytest.mark.asyncio
    async def test_embedding_error_skips_batch(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, side_effect=EmbeddingError("fail")), \
             patch("src.utils.settings", _fake_settings()):
            result = await add_documents_to_db(
                session, ["url"], ["content"], [{"source": "x"}], [0]
            )
        assert result == 0
        session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_insert_error_rolls_back(self):
        from sqlalchemy.exc import SQLAlchemyError
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        session.add_all.side_effect = SQLAlchemyError("insert fail")

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings()):
            result = await add_documents_to_db(
                session, ["url"], ["content"], [{"source": "x"}], [0]
            )
        assert result == 0
        session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_contextual_enrichment_applied(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.generate_contextual_text", new_callable=AsyncMock,
                   return_value=("enriched content", True)) as mock_ctx, \
             patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings(
                 USE_CONTEXTUAL_EMBEDDINGS=True, LLM_ENABLED=True,
                 LLM_API_KEY="k", LLM_BASE_URL="u", LLM_MODEL_NAME="m",
             )):
            result = await add_documents_to_db(
                session, ["url"], ["content"], [{"source": "x"}], [0],
                full_documents=["full doc"],
            )
        mock_ctx.assert_called_once()
        assert result == 1

    @pytest.mark.asyncio
    async def test_delete_existing_sql_error_logs_and_continues(self):
        from sqlalchemy.exc import SQLAlchemyError
        session = MagicMock()
        existing_page = MagicMock()
        call_count = {"n": 0}

        def mock_exec(stmt):
            call_count["n"] += 1
            m = MagicMock()
            m.all.return_value = [existing_page]
            return m

        session.exec.side_effect = mock_exec
        session.delete.side_effect = SQLAlchemyError("delete fail")

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings()):
            await add_documents_to_db(
                session, ["url"], ["content"], [{"source": "x"}], [0]
            )
        session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_batches(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        urls = [f"http://x.com/{i}" for i in range(5)]
        contents = [f"content {i}" for i in range(5)]
        metas = [{"source": "x.com"}] * 5
        chunks = list(range(5))
        embed_calls = []

        async def mock_batch(texts):
            embed_calls.append(len(texts))
            return [EMBED_A] * len(texts)

        with patch("src.utils.create_embeddings_batch", side_effect=mock_batch), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings(BATCH_SIZE=2)):
            result = await add_documents_to_db(session, urls, contents, metas, chunks)
        assert result == 5
        assert len(embed_calls) == 3  # 2+2+1


# ---------------------------------------------------------------------------
# Tests: add_code_examples_to_db
# ---------------------------------------------------------------------------

class TestAddCodeExamplesToDb:
    @pytest.mark.asyncio
    async def test_empty_returns_zero(self):
        session = MagicMock()
        result = await add_code_examples_to_db(session, [], [], [], [], [], [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_stores_examples(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1):
            result = await add_code_examples_to_db(
                session,
                urls=["http://x.com"],
                contents=["print('hi')"],
                languages=["python"],
                summaries=["hello world"],
                metadatas=[{"source": "x.com"}],
                chunk_numbers=[0],
            )
        assert result == 1
        session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_source_in_metadata(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        added_rows = []
        session.add_all.side_effect = lambda rows: added_rows.extend(rows)

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]):
            result = await add_code_examples_to_db(
                session, ["url"], ["code"], [None], [None], [{}], [0]
            )
        assert result == 1
        assert added_rows[0].source_id is None

    @pytest.mark.asyncio
    async def test_embedding_error_returns_zero(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, side_effect=EmbeddingError("bad")):
            result = await add_code_examples_to_db(
                session, ["url"], ["code"], [None], [None], [{}], [0]
            )
        assert result == 0

    @pytest.mark.asyncio
    async def test_db_insert_error_rolls_back(self):
        from sqlalchemy.exc import SQLAlchemyError
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        session.add_all.side_effect = SQLAlchemyError("insert fail")

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1):
            result = await add_code_examples_to_db(
                session, ["url"], ["code"], [None], [None], [{"source": "x"}], [0]
            )
        assert result == 0
        session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_delete_sql_error_rolls_back(self):
        from sqlalchemy.exc import SQLAlchemyError
        session = MagicMock()
        existing = MagicMock()
        session.exec.return_value.all.return_value = [existing]
        session.delete.side_effect = SQLAlchemyError("delete fail")

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1):
            await add_code_examples_to_db(
                session, ["url"], ["code"], [None], [None], [{"source": "x"}], [0]
            )
        session.rollback.assert_called()


# ---------------------------------------------------------------------------
# Tests: search_documents
# ---------------------------------------------------------------------------

class TestSearchDocuments:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        session = MagicMock()
        result = await search_documents(session, "")
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self):
        session = MagicMock()
        result = await search_documents(session, "   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_error_returns_empty(self):
        session = MagicMock()
        with patch("src.utils.create_embedding", new_callable=AsyncMock, side_effect=EmbeddingError("fail")):
            result = await search_documents(session, "query")
        assert result == []



# ---------------------------------------------------------------------------
# Tests: _python_side_vector_search
# ---------------------------------------------------------------------------

class TestPythonSideVectorSearch:
    def _make_page(self, pid, url, embedding, metadata=None):
        p = MagicMock()
        p.id = pid
        p.url = url
        p.chunk_number = 0
        p.content = "content"
        p.page_metadata = metadata or {}
        p.embedding = embedding
        return p

    def test_returns_sorted_by_similarity(self):
        session = MagicMock()
        page_a = self._make_page(1, "url_a", EMBED_A)
        page_b = self._make_page(2, "url_b", EMBED_B)
        session.exec.return_value.all.return_value = [page_a, page_b]
        results = _python_side_vector_search(session, EMBED_Q, limit=5, filter_metadata=None)
        assert results[0][0] == 1
        assert results[0][5] > results[1][5]

    def test_skips_page_with_no_embedding(self):
        session = MagicMock()
        page = self._make_page(1, "url", None)
        session.exec.return_value.all.return_value = [page]
        results = _python_side_vector_search(session, EMBED_Q, limit=5, filter_metadata=None)
        assert results == []

    def test_filter_metadata(self):
        session = MagicMock()
        page_match = self._make_page(1, "url", EMBED_A, {"source": "x.com"})
        page_no_match = self._make_page(2, "url2", EMBED_B, {"source": "y.com"})
        session.exec.return_value.all.return_value = [page_match, page_no_match]
        results = _python_side_vector_search(session, EMBED_Q, limit=5, filter_metadata={"source": "x.com"})
        assert len(results) == 1
        assert results[0][0] == 1

    def test_filter_metadata_non_dict_page_metadata(self):
        session = MagicMock()
        page = self._make_page(1, "url", EMBED_A, "bad-type")
        session.exec.return_value.all.return_value = [page]
        results = _python_side_vector_search(session, EMBED_Q, limit=5, filter_metadata={"source": "x.com"})
        assert results == []

    def test_zero_norm_query_yields_zero_similarity(self):
        session = MagicMock()
        page = self._make_page(1, "url", EMBED_A)
        session.exec.return_value.all.return_value = [page]
        results = _python_side_vector_search(session, [0.0, 0.0, 0.0, 0.0], limit=5, filter_metadata=None)
        assert results[0][5] == 0.0

    def test_zero_norm_page_yields_zero_similarity(self):
        session = MagicMock()
        page = self._make_page(1, "url", [0.0, 0.0, 0.0, 0.0])
        session.exec.return_value.all.return_value = [page]
        results = _python_side_vector_search(session, EMBED_Q, limit=5, filter_metadata=None)
        assert results[0][5] == 0.0

    def test_limit_applied(self):
        session = MagicMock()
        pages = [self._make_page(i, f"url_{i}", EMBED_A) for i in range(10)]
        session.exec.return_value.all.return_value = pages
        results = _python_side_vector_search(session, EMBED_Q, limit=3, filter_metadata=None)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests: search_code_examples
# ---------------------------------------------------------------------------

class TestSearchCodeExamples:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        session = MagicMock()
        result = await search_code_examples(session, "")
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self):
        session = MagicMock()
        result = await search_code_examples(session, "  ")
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_error_returns_empty(self):
        session = MagicMock()
        with patch("src.utils.create_embedding", new_callable=AsyncMock, side_effect=EmbeddingError("bad")):
            result = await search_code_examples(session, "query")
        assert result == []

    @pytest.mark.asyncio
    async def test_language_filter_accepted(self):
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            result = await search_code_examples(session, "query", language="python")
        assert result == []

    @pytest.mark.asyncio
    async def test_db_exception_returns_empty(self):
        session = MagicMock()
        session.exec.side_effect = Exception("DB down")
        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            result = await search_code_examples(session, "query")
        assert result == []



# ---------------------------------------------------------------------------
# Tests: rerank_results
# ---------------------------------------------------------------------------

class TestRerankResults:
    def test_disabled_returns_top_k_original_order(self):
        results = [{"content": f"r{i}", "id": i} for i in range(10)]
        with patch("src.utils.settings", _fake_settings(USE_RERANKING=False)):
            out = rerank_results("query", results, top_k=3)
        assert len(out) == 3
        assert out[0]["id"] == 0

    def test_empty_input_returns_empty(self):
        # Empty list short-circuits regardless of USE_RERANKING
        with patch("src.utils.settings", _fake_settings(USE_RERANKING=True)):
            out = rerank_results("q", [], top_k=5)
        assert out == []

    def test_enabled_uses_cross_encoder(self):
        results = [{"content": "relevant", "id": 1}, {"content": "unrelated", "id": 2}]
        mock_ce_instance = MagicMock()
        mock_ce_instance.predict.return_value = [0.9, 0.1]
        mock_ce_class = MagicMock(return_value=mock_ce_instance)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                m = MagicMock()
                m.CrossEncoder = mock_ce_class
                return m
            return original_import(name, *args, **kwargs)

        with patch("src.utils.settings", _fake_settings(USE_RERANKING=True)), \
             patch.object(builtins, "__import__", side_effect=mock_import):
            out = rerank_results("query", results, top_k=2)
        assert len(out) == 2
        mock_ce_instance.predict.assert_called_once()

    def test_cross_encoder_exception_fallback(self):
        results = [{"content": f"r{i}", "id": i} for i in range(5)]
        original_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("src.utils.settings", _fake_settings(USE_RERANKING=True)), \
             patch.object(builtins, "__import__", side_effect=bad_import):
            out = rerank_results("query", results, top_k=3)
        assert len(out) == 3

    def test_rerank_sorts_by_score(self):
        results = [{"content": "a", "id": 1}, {"content": "b", "id": 2}]
        mock_ce_instance = MagicMock()
        mock_ce_instance.predict.return_value = [0.3, 0.8]
        mock_ce_class = MagicMock(return_value=mock_ce_instance)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                m = MagicMock()
                m.CrossEncoder = mock_ce_class
                return m
            return original_import(name, *args, **kwargs)

        with patch("src.utils.settings", _fake_settings(USE_RERANKING=True)), \
             patch.object(builtins, "__import__", side_effect=mock_import):
            out = rerank_results("q", results, top_k=2)
        assert out[0]["id"] == 2
        assert out[0]["rerank_score"] == 0.8


# ---------------------------------------------------------------------------
# Tests: extract_code_blocks
# ---------------------------------------------------------------------------

class TestExtractCodeBlocks:
    def test_simple_python_block(self):
        md = "before\n```python\nprint('hi')\n```\nafter"
        blocks = extract_code_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert "print" in blocks[0]["content"]

    def test_no_language_tag_yields_none(self):
        md = "```\nsome code\n```"
        blocks = extract_code_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["language"] is None

    def test_multiple_blocks(self):
        md = "```python\na=1\n```\n\n```javascript\nconsole.log()\n```"
        blocks = extract_code_blocks(md)
        assert len(blocks) == 2

    def test_empty_code_block_skipped(self):
        md = "```python\n\n```"
        blocks = extract_code_blocks(md)
        assert blocks == []

    def test_no_code_blocks(self):
        assert extract_code_blocks("just text") == []


# ---------------------------------------------------------------------------
# Tests: get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_yields_session(self):
        mock_session = MagicMock()
        with patch("src.utils.Session") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            gen = get_session()
            sess = next(gen)
            assert sess is mock_session
            try:
                next(gen)
            except StopIteration:
                pass


# ---------------------------------------------------------------------------
# Extra coverage: add_documents_to_db — all-whitespace branch (line 326)
# and delete-commit branch (line 337)
# ---------------------------------------------------------------------------

class TestAddDocumentsExtraCoverage:
    @pytest.mark.asyncio
    async def test_all_whitespace_contents_returns_zero(self):
        """All contents are whitespace → valid list is empty → return 0 early."""
        session = MagicMock()
        result = await add_documents_to_db(
            session,
            urls=["url1", "url2"],
            contents=["  ", "\t"],
            metadatas=[{}, {}],
            chunk_numbers=[0, 1],
        )
        assert result == 0
        session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_commit_when_existing_rows_found(self):
        """When to_delete is non-empty the delete commit is issued."""
        session = MagicMock()
        existing_page = MagicMock()
        session.exec.return_value.all.return_value = [existing_page]

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1), \
             patch("src.utils.settings", _fake_settings(USE_CONTEXTUAL_EMBEDDINGS=False, LLM_ENABLED=False)):
            result = await add_documents_to_db(
                session, ["url"], ["content"], [{"source": "x"}], [0]
            )
        assert result == 1
        session.delete.assert_called_once_with(existing_page)
        # Two commits: delete + insert
        assert session.commit.call_count == 2


# ---------------------------------------------------------------------------
# Extra coverage: add_code_examples_to_db — delete-commit branch (line 413)
# ---------------------------------------------------------------------------

class TestAddCodeExamplesExtraCoverage:
    @pytest.mark.asyncio
    async def test_delete_commit_when_existing_rows_found(self):
        """When to_delete is non-empty the delete commit is issued."""
        session = MagicMock()
        existing_example = MagicMock()
        session.exec.return_value.all.return_value = [existing_example]

        with patch("src.utils.create_embeddings_batch", new_callable=AsyncMock, return_value=[EMBED_A]), \
             patch("src.utils.upsert_source", return_value=1):
            result = await add_code_examples_to_db(
                session,
                urls=["url"],
                contents=["code"],
                languages=[None],
                summaries=[None],
                metadatas=[{"source": "x"}],
                chunk_numbers=[0],
            )
        assert result == 1
        session.delete.assert_called_once_with(existing_example)
        assert session.commit.call_count == 2


# ---------------------------------------------------------------------------
# Extra coverage: search_documents — core search logic (lines 476-557)
# ---------------------------------------------------------------------------

class TestSearchDocumentsCore:
    @pytest.mark.asyncio
    async def test_non_hybrid_vector_search_returns_results(self):
        """Non-hybrid path: vector results returned directly."""
        session = MagicMock()
        # Row tuple: (id, url, chunk_number, content, metadata, similarity)
        row = (1, "http://x.com", 0, "content", {"source": "x"}, 0.9)
        session.exec.return_value.all.return_value = [row]

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            results = await search_documents(session, "query text", use_hybrid=False)

        assert len(results) == 1
        assert results[0]["url"] == "http://x.com"
        assert results[0]["similarity_score"] == pytest.approx(0.9)
        assert results[0]["fts_score"] == 0.0

    @pytest.mark.asyncio
    async def test_non_hybrid_metadata_non_dict_normalized(self):
        """Non-dict metadata in row is coerced to empty dict."""
        session = MagicMock()
        row = (1, "http://x.com", 0, "content", "not-a-dict", 0.8)
        session.exec.return_value.all.return_value = [row]

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            results = await search_documents(session, "query", use_hybrid=False)

        assert results[0]["page_metadata"] == {}

    @pytest.mark.asyncio
    async def test_db_function_failure_falls_back_to_python(self):
        """When DB function raises, falls back to _python_side_vector_search."""
        session = MagicMock()
        session.exec.side_effect = Exception("function not found")

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q), \
             patch("src.utils._python_side_vector_search", return_value=[]) as mock_fallback:
            results = await search_documents(session, "query", use_hybrid=False)

        mock_fallback.assert_called_once()
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_merges_vector_and_fts_results(self):
        """Hybrid path: vector + FTS results merged via RRF."""
        session = MagicMock()
        vector_row = (1, "http://v.com", 0, "vector content", {"source": "x"}, 0.9)
        fts_row = (2, "http://f.com", 0, "fts content", {"source": "x"}, 0.7)

        call_count = {"n": 0}

        def mock_exec(stmt):
            call_count["n"] += 1
            m = MagicMock()
            if call_count["n"] == 1:
                m.all.return_value = [vector_row]
            else:
                m.all.return_value = [fts_row]
            return m

        session.exec.side_effect = mock_exec

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            results = await search_documents(session, "query text", match_count=5, use_hybrid=True)

        urls = [r["url"] for r in results]
        assert "http://v.com" in urls
        assert "http://f.com" in urls

    @pytest.mark.asyncio
    async def test_hybrid_fts_exception_falls_back_to_vector_only(self):
        """When FTS query raises, fts_raw is empty; only vector results returned."""
        session = MagicMock()
        vector_row = (1, "http://v.com", 0, "content", {"source": "x"}, 0.9)

        call_count = {"n": 0}

        def mock_exec(stmt):
            call_count["n"] += 1
            m = MagicMock()
            if call_count["n"] == 1:
                m.all.return_value = [vector_row]
            else:
                raise Exception("FTS not available")
            return m

        session.exec.side_effect = mock_exec

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q):
            results = await search_documents(session, "query text", match_count=5, use_hybrid=True)

        assert len(results) >= 1
        assert results[0]["url"] == "http://v.com"

    @pytest.mark.asyncio
    async def test_use_hybrid_param_overrides_settings(self):
        """use_hybrid parameter takes precedence over settings.USE_HYBRID_SEARCH."""
        session = MagicMock()
        row = (1, "http://x.com", 0, "content", {}, 0.9)
        session.exec.return_value.all.return_value = [row]

        with patch("src.utils.create_embedding", new_callable=AsyncMock, return_value=EMBED_Q), \
             patch("src.utils.settings", _fake_settings(USE_HYBRID_SEARCH=True)):
            # Explicitly pass use_hybrid=False to override the settings
            results = await search_documents(session, "query", use_hybrid=False)

        # Only one exec call (no FTS pass)
        assert session.exec.call_count == 1
        assert len(results) == 1
