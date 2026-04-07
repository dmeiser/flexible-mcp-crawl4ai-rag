from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import AsyncOpenAI, OpenAI


@dataclass(frozen=True)
class OpenAIConfiguration:
    """Configuration for OpenAI-compatible endpoints."""

    api_key: Optional[str]
    base_url: Optional[str] = None
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 90.0

    @property
    def resolved_api_key(self) -> Optional[str]:
        if isinstance(self.api_key, str):
            value = self.api_key.strip()
            return value or None
        return self.api_key


class OpenAICompatibleEndpoint:
    """Shared client wrapper for OpenAI-compatible endpoints."""

    def __init__(
        self,
        configuration: OpenAIConfiguration,
        *,
        async_openai_cls: type[AsyncOpenAI] = AsyncOpenAI,
        openai_cls: type[OpenAI] = OpenAI,
        async_chat_retry_fn: Optional[Callable[..., Any]] = None,
        sync_chat_retry_fn: Optional[Callable[..., Any]] = None,
    ):
        self.configuration = configuration
        self.api_key = configuration.resolved_api_key
        self.base_url = configuration.base_url
        self._async_openai_cls = async_openai_cls
        self._openai_cls = openai_cls
        self._async_chat_retry_fn = async_chat_retry_fn
        self._sync_chat_retry_fn = sync_chat_retry_fn

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    async def create_embedding(self, *, model: str, text: str) -> List[float]:
        client = self._async_openai_cls(**self._client_kwargs())
        try:
            resp = await client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        finally:
            await client.close()

    async def chat_completion(
        self,
        *,
        request_kwargs: Dict[str, Any],
        max_retries: int,
        retry_delay_seconds: float,
        call_name: str,
    ) -> Any:
        client = self._async_openai_cls(**self._client_kwargs())
        try:
            if self._async_chat_retry_fn is not None:
                return await self._async_chat_retry_fn(
                    client=client,
                    request_kwargs=request_kwargs,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                    call_name=call_name,
                )
            return await _async_chat_completion_with_retries(
                client=client,
                request_kwargs=request_kwargs,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        finally:
            await client.close()

    def chat_completion_sync(
        self,
        *,
        request_kwargs: Dict[str, Any],
        max_retries: int,
        retry_delay_seconds: float,
        call_name: str,
    ) -> Any:
        client = self._openai_cls(**self._client_kwargs())
        try:
            if self._sync_chat_retry_fn is not None:
                return self._sync_chat_retry_fn(
                    client=client,
                    request_kwargs=request_kwargs,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                    call_name=call_name,
                )
            return _sync_chat_completion_with_retries(
                client=client,
                request_kwargs=request_kwargs,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    async def chat_completion_raw(
        self,
        *,
        payload: Dict[str, Any],
        max_retries: int,
        retry_delay_seconds: float,
        call_name: str,
    ) -> Dict[str, Any]:
        if not self.base_url:
            raise ValueError(f"{call_name} requires base_url for raw chat completions")
        return await _raw_chat_completion_with_retries(
            url=f"{str(self.base_url).rstrip('/')}/chat/completions",
            headers=_raw_chat_headers(self.api_key),
            payload=payload,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=self.configuration.timeout_seconds,
            call_name=call_name,
        )


class EmbeddingsProvider:
    """Embedding provider backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        configuration: OpenAIConfiguration,
        model_name: str,
        *,
        normalize_fn: Optional[Callable[[List[float]], List[float]]] = None,
        async_openai_cls: type[AsyncOpenAI] = AsyncOpenAI,
        openai_cls: type[OpenAI] = OpenAI,
        async_chat_retry_fn: Optional[Callable[..., Any]] = None,
        sync_chat_retry_fn: Optional[Callable[..., Any]] = None,
    ):
        self.endpoint = OpenAICompatibleEndpoint(
            configuration,
            async_openai_cls=async_openai_cls,
            openai_cls=openai_cls,
            async_chat_retry_fn=async_chat_retry_fn,
            sync_chat_retry_fn=sync_chat_retry_fn,
        )
        self.model_name = model_name
        self._normalize_fn = normalize_fn

    async def create_embedding(self, text: str) -> List[float]:
        raw = await self.endpoint.create_embedding(model=self.model_name, text=text)
        if self._normalize_fn is None:
            return raw
        return self._normalize_fn(raw)


class ChatCompletionRetryStrategy:
    """Class-based retry strategy for chat completion calls."""

    @staticmethod
    def retry_backoff_seconds(base_delay_seconds: float, attempt: int) -> float:
        return base_delay_seconds * (2 ** (attempt - 1))

    @classmethod
    async def async_chat_completion_with_retries(
        cls,
        *,
        client: AsyncOpenAI,
        request_kwargs: Dict[str, Any],
        max_retries: int,
        retry_delay_seconds: float,
        call_name: str,
    ) -> Any:
        _ = call_name
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            try:
                return await client.chat.completions.create(**request_kwargs)
            except Exception:
                if attempt >= attempts:
                    raise
                await asyncio.sleep(cls.retry_backoff_seconds(float(retry_delay_seconds), attempt))

    @classmethod
    def sync_chat_completion_with_retries(
        cls,
        *,
        client: OpenAI,
        request_kwargs: Dict[str, Any],
        max_retries: int,
        retry_delay_seconds: float,
        call_name: str,
    ) -> Any:
        _ = call_name
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            try:
                return client.chat.completions.create(**request_kwargs)
            except Exception:
                if attempt >= attempts:
                    raise
                import time

                time.sleep(cls.retry_backoff_seconds(float(retry_delay_seconds), attempt))


def _raw_chat_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _raw_chat_completion_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int,
    retry_delay_seconds: float,
    timeout_seconds: float,
    call_name: str,
) -> Dict[str, Any]:
    attempts = max(1, int(max_retries))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(1, attempts + 1):
            maybe_raw = await _raw_chat_attempt(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                attempt=attempt,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            if maybe_raw is not None:
                return maybe_raw
    raise RuntimeError(f"{call_name} exhausted retries")


async def _raw_chat_attempt(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    attempt: int,
    attempts: int,
    retry_delay_seconds: float,
) -> Optional[Dict[str, Any]]:
    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()
        return raw if isinstance(raw, dict) else {}
    except Exception:
        if attempt >= attempts:
            raise
        await asyncio.sleep(_retry_backoff_seconds(float(retry_delay_seconds), attempt))
        return None


def _retry_backoff_seconds(base_delay_seconds: float, attempt: int) -> float:
    return ChatCompletionRetryStrategy.retry_backoff_seconds(base_delay_seconds, attempt)


async def _async_chat_completion_with_retries(
    *,
    client: AsyncOpenAI,
    request_kwargs: Dict[str, Any],
    max_retries: int,
    retry_delay_seconds: float,
) -> Any:
    return await ChatCompletionRetryStrategy.async_chat_completion_with_retries(
        client=client,
        request_kwargs=request_kwargs,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        call_name="chat completion",
    )


def _sync_chat_completion_with_retries(
    *,
    client: OpenAI,
    request_kwargs: Dict[str, Any],
    max_retries: int,
    retry_delay_seconds: float,
) -> Any:
    return ChatCompletionRetryStrategy.sync_chat_completion_with_retries(
        client=client,
        request_kwargs=request_kwargs,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        call_name="chat completion",
    )
