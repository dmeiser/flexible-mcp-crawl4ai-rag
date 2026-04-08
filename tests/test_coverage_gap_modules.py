import logging
import os
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "test")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.config import Settings
from src.providers import openai_stack as oas
from src.providers.openai_stack import EmbeddingsProvider, OpenAICompatibleEndpoint, OpenAIConfiguration
from src.services.contextual_enrichment_service import ContextualEnrichmentService
from src.services.embedding_service import EmbeddingService
from src.services.retrieval import search_documents_with_embedding
from src.tools import reembed_documents as reembed
from src.utils import _context_prompt


class _DummyAsyncClient:
    def __init__(self, **_kwargs):
        self.embeddings = SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])]))
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value={"ok": True})))
        self.close = AsyncMock()


class _DummySyncClient:
    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=MagicMock(return_value={"ok": True})))
        self.close = MagicMock()


@pytest.mark.asyncio
async def test_openai_endpoint_chat_completion_paths_and_close(caplog):
    endpoint = OpenAICompatibleEndpoint(
        OpenAIConfiguration(api_key="k", base_url="http://x"),
        async_openai_cls=_DummyAsyncClient,
        openai_cls=_DummySyncClient,
    )
    with caplog.at_level(logging.DEBUG, logger="src.providers.openai_stack"):
        out_async = await endpoint.chat_completion(
            request_kwargs={"model": "m", "messages": []},
            max_retries=1,
            retry_delay_seconds=0.0,
            call_name="chat",
        )
    assert out_async["ok"] is True

    with caplog.at_level(logging.DEBUG, logger="src.providers.openai_stack"):
        out_sync = endpoint.chat_completion_sync(
            request_kwargs={"model": "m", "messages": []},
            max_retries=1,
            retry_delay_seconds=0.0,
            call_name="chat-sync",
        )
    assert out_sync["ok"] is True
    assert "chat completion request:" in caplog.text
    assert "chat completion response:" in caplog.text
    assert "chat completion sync request:" in caplog.text
    assert "chat completion sync response:" in caplog.text


@pytest.mark.asyncio
async def test_openai_endpoint_chat_completion_raw_and_no_base_url_error(caplog):
    endpoint_no_base = OpenAICompatibleEndpoint(OpenAIConfiguration(api_key="k", base_url=None))
    with pytest.raises(ValueError):
        await endpoint_no_base.chat_completion_raw(
            payload={},
            max_retries=1,
            retry_delay_seconds=0.0,
            call_name="raw",
        )

    endpoint = OpenAICompatibleEndpoint(OpenAIConfiguration(api_key="k", base_url="http://x"))
    with (
        caplog.at_level(logging.DEBUG, logger="src.providers.openai_stack"),
        patch("src.providers.openai_stack._raw_chat_completion_with_retries", new=AsyncMock(return_value={"ok": 1})),
    ):
        raw = await endpoint.chat_completion_raw(
            payload={"a": 1},
            max_retries=1,
            retry_delay_seconds=0.0,
            call_name="raw",
        )
    assert raw == {"ok": 1}
    assert "raw chat completion request:" in caplog.text
    assert "raw chat completion headers:" in caplog.text
    assert "***REDACTED***" in caplog.text


@pytest.mark.asyncio
async def test_embeddings_provider_without_normalize_returns_raw():
    p = EmbeddingsProvider(
        configuration=OpenAIConfiguration(api_key="k", base_url="http://x"),
        model_name="m",
        normalize_fn=None,
        async_openai_cls=_DummyAsyncClient,
        openai_cls=_DummySyncClient,
    )
    vec = await p.create_embedding("hello")
    assert vec == [1.0]


@pytest.mark.asyncio
async def test_retry_strategies_and_raw_helpers():
    aclient = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=[RuntimeError("x"), {"ok": 1}])))
    )
    with patch("src.providers.openai_stack.asyncio.sleep", new=AsyncMock()):
        out = await oas.ChatCompletionRetryStrategy.async_chat_completion_with_retries(
            client=aclient,
            request_kwargs={"model": "m"},
            max_retries=2,
            retry_delay_seconds=0.0,
            call_name="c",
        )
    assert out["ok"] == 1

    sclient = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock(side_effect=[RuntimeError("x"), {"ok": 2}])))
    )
    with patch("time.sleep"):
        out2 = oas.ChatCompletionRetryStrategy.sync_chat_completion_with_retries(
            client=sclient,
            request_kwargs={"model": "m"},
            max_retries=2,
            retry_delay_seconds=0.0,
            call_name="c",
        )
    assert out2["ok"] == 2

    assert oas._raw_chat_headers("k")["Authorization"].startswith("Bearer")
    assert "Authorization" not in oas._raw_chat_headers(None)
    assert oas._redacted_headers({"Authorization": "Bearer secret"})["Authorization"].endswith("***REDACTED***")


def test_raw_response_body_preview_noop_when_not_debug(caplog):
    with caplog.at_level(logging.INFO, logger="src.providers.openai_stack"):
        oas._maybe_log_raw_response_body_preview(b"abc", "abc", 1)
    assert "raw chat completion response body preview" not in caplog.text


@pytest.mark.asyncio
async def test_raw_chat_attempt_and_completion_retry_exhaustion():
    resp = SimpleNamespace(raise_for_status=lambda: None, json=lambda: [1, 2, 3])
    client = SimpleNamespace(post=AsyncMock(return_value=resp))
    got = await oas._raw_chat_attempt(
        client=client,
        url="http://x",
        headers={},
        payload={},
        attempt=1,
        attempts=1,
        retry_delay_seconds=0.0,
        timeout_seconds=1.0,
    )
    assert got == {}

    client_fail = SimpleNamespace(post=AsyncMock(side_effect=RuntimeError("boom")))
    with patch("src.providers.openai_stack.asyncio.sleep", new=AsyncMock()):
        got2 = await oas._raw_chat_attempt(
            client=client_fail,
            url="http://x",
            headers={},
            payload={},
            attempt=1,
            attempts=2,
            retry_delay_seconds=0.0,
            timeout_seconds=1.0,
        )
    assert got2 is None

    with patch("src.providers.openai_stack._raw_chat_attempt", new=AsyncMock(return_value=None)):
        with pytest.raises(RuntimeError):
            await oas._raw_chat_completion_with_retries(
                url="http://x",
                headers={},
                payload={},
                max_retries=1,
                retry_delay_seconds=0.0,
                timeout_seconds=1.0,
                call_name="raw",
            )

    with patch("src.providers.openai_stack._raw_chat_attempt", new=AsyncMock(return_value={"ok": 7})):
        out = await oas._raw_chat_completion_with_retries(
            url="http://x",
            headers={},
            payload={},
            max_retries=2,
            retry_delay_seconds=0.0,
            timeout_seconds=1.0,
            call_name="raw",
        )
    assert out == {"ok": 7}

    with pytest.raises(RuntimeError):
        await oas._raw_chat_attempt(
            client=client_fail,
            url="http://x",
            headers={},
            payload={},
            attempt=1,
            attempts=1,
            retry_delay_seconds=0.0,
            timeout_seconds=1.0,
        )


@pytest.mark.asyncio
async def test_raw_chat_attempt_reads_body_bytes_and_logs_debug(caplog):
    class _Resp:
        status_code = 200
        headers = {"content-type": "application/json"}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        async def aread():
            return b'{"ok": 42}'

    client = SimpleNamespace(post=AsyncMock(return_value=_Resp()))
    with caplog.at_level(logging.DEBUG, logger="src.providers.openai_stack"):
        got = await oas._raw_chat_attempt(
            client=client,
            url="http://x",
            headers={},
            payload={},
            attempt=1,
            attempts=1,
            retry_delay_seconds=0.0,
            timeout_seconds=1.0,
        )

    assert got == {"ok": 42}
    assert "raw chat completion response headers" in caplog.text
    assert "raw chat completion response body preview" in caplog.text


@pytest.mark.asyncio
async def test_contextual_and_embedding_services_and_utils_prompt():
    logger = MagicMock()
    endpoint = SimpleNamespace(
        chat_completion=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" summary "))])
        )
    )
    svc = ContextualEnrichmentService(endpoint_factory=lambda **_kw: endpoint, logger=logger)
    settings = SimpleNamespace(
        USE_CONTEXTUAL_EMBEDDINGS=True,
        effective_contextual_model_name="m",
        effective_contextual_api_key="k",
        effective_contextual_base_url="http://x",
        effective_contextual_max_retries=1,
        effective_contextual_retry_delay_seconds=0.0,
        CHUNK_SIZE=10,
    )
    enriched, ok = await svc.generate_contextual_text(settings, "doc", "chunk")
    assert ok is True and enriched.startswith("Context:")

    settings_missing = SimpleNamespace(USE_CONTEXTUAL_EMBEDDINGS=False, effective_contextual_model_name=None)
    out, ok2 = await svc.generate_contextual_text(settings_missing, "doc", "chunk")
    assert out == "chunk" and ok2 is False

    bad_endpoint = SimpleNamespace(
        chat_completion=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=123))])
        )
    )
    svc2 = ContextualEnrichmentService(endpoint_factory=lambda **_kw: bad_endpoint, logger=logger)
    txt = await svc2.request_contextual_summary(settings, "doc", "chunk")
    assert txt == ""
    assert svc.combine_context_and_chunk(settings, "", "chunk") is None

    emb = EmbeddingService(
        provider_factory=lambda: SimpleNamespace(create_embedding=AsyncMock(return_value=[1.0, 0.0])),
        error_cls=ValueError,
    )
    assert EmbeddingService.normalize([0.0, 0.0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        await emb.create_embedding(" ")
    assert await emb.create_embeddings_batch([]) == []

    emb_fail = EmbeddingService(
        provider_factory=lambda: SimpleNamespace(create_embedding=AsyncMock(side_effect=RuntimeError("x"))),
        error_cls=ValueError,
    )
    with pytest.raises(ValueError):
        await emb_fail.create_openai_embedding("x")

    ok_vec = await emb.create_embedding("hello")
    assert ok_vec == [1.0, 0.0]
    batch_vecs = await emb.create_embeddings_batch(["a", "b"])
    assert len(batch_vecs) == 2

    prompt = _context_prompt("doc", "chunk")
    assert "<document>" in prompt and "<chunk>" in prompt


@pytest.mark.asyncio
async def test_contextual_generate_handles_exception_branch():
    logger = MagicMock()
    svc = ContextualEnrichmentService(endpoint_factory=lambda **_kw: None, logger=logger)
    settings = SimpleNamespace(USE_CONTEXTUAL_EMBEDDINGS=True, effective_contextual_model_name="m")
    with patch.object(svc, "request_contextual_summary", new=AsyncMock(side_effect=RuntimeError("boom"))):
        out, ok = await svc.generate_contextual_text(settings, "doc", "chunk")
    assert out == "chunk"
    assert ok is False
    assert logger.warning.called


@pytest.mark.asyncio
async def test_contextual_generate_returns_chunk_when_summary_empty():
    logger = MagicMock()
    svc = ContextualEnrichmentService(endpoint_factory=lambda **_kw: None, logger=logger)
    settings = SimpleNamespace(USE_CONTEXTUAL_EMBEDDINGS=True, effective_contextual_model_name="m", CHUNK_SIZE=10)
    with patch.object(svc, "request_contextual_summary", new=AsyncMock(return_value="")):
        out, ok = await svc.generate_contextual_text(settings, "doc", "chunk")
    assert out == "chunk"
    assert ok is False


def test_config_and_retrieval_wrapper():
    s = Settings(
        POSTGRES_URL="postgresql://u:p@h/db",
        EMBEDDING_BASE_URL="http://localhost:11434/v1",
        EMBEDDING_MODEL_NAME="m",
    )
    assert s.effective_embedding_base_url == "http://localhost:11434/v1"
    assert s.effective_embedding_model_name == "m"
    assert s.effective_embedding_max_retries == 3
    assert s.effective_embedding_retry_delay_seconds == 1.0

    with patch("src.services.retrieval._search_documents_with_embedding", return_value=[]) as m:
        out = search_documents_with_embedding(
            session=MagicMock(),
            query="q",
            query_embedding=[0.1],
            match_count=1,
            filter_metadata=None,
            filter_json="{}",
            hybrid=False,
            crawled_page_cls=MagicMock(),
        )
    assert out == []
    assert m.called


@pytest.mark.asyncio
async def test_reembed_documents_module_paths_and_main():
    assert isinstance(reembed._current_model(), str)

    page_ok = SimpleNamespace(id=1, content="a")
    page_bad = SimpleNamespace(id=2, content="b")
    code_ok = SimpleNamespace(id=3, content="c")
    code_bad = SimpleNamespace(id=4, content="d")

    session = MagicMock()
    first = MagicMock()
    first.all.return_value = [page_ok, page_bad]
    second = MagicMock()
    second.all.return_value = [code_ok, code_bad]
    session.exec.side_effect = [first, second]

    async def _embed(text):
        if text in {"b", "d"}:
            raise RuntimeError("fail")
        return [1.0]

    def _gen():
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            yield session

        yield _ctx()

    with (
        patch("src.tools.reembed_documents.get_session", side_effect=_gen),
        patch("src.tools.reembed_documents.create_embedding", new=AsyncMock(side_effect=_embed)),
    ):
        await reembed.reembed_all()

    assert session.commit.call_count == 2

    with patch("asyncio.run", side_effect=lambda c: c.close()):
        runpy.run_module("src.tools.reembed_documents", run_name="__main__")
