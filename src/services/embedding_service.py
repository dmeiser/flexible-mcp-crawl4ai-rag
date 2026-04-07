import asyncio
from typing import Any, Callable, List, Type

import numpy as np


class EmbeddingService:
    """Service for embedding creation and normalization."""

    def __init__(self, provider_factory: Callable[[], Any], error_cls: Type[Exception]) -> None:
        self._provider_factory = provider_factory
        self._error_cls = error_cls

    @staticmethod
    def normalize(vec: List[float]) -> List[float]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vec
        return (arr / norm).tolist()

    async def create_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise self._error_cls("Attempted to create embedding for empty string.")

        return await self.create_openai_embedding(text)

    async def create_openai_embedding(self, text: str) -> List[float]:
        provider = self._provider_factory()
        try:
            return await provider.create_embedding(text)
        except Exception as exc:
            raise self._error_cls(f"OpenAI embedding failed: {exc}") from exc

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.gather(*[self.create_embedding(t) for t in texts])
