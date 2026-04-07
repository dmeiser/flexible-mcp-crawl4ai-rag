"""Provider abstractions and implementations."""

from .openai_stack import EmbeddingsProvider, OpenAICompatibleEndpoint, OpenAIConfiguration
from .openrouter_web_search import OpenRouterWebSearchAdapter, WebSearchModel

__all__ = [
    "OpenAIConfiguration",
    "OpenAICompatibleEndpoint",
    "EmbeddingsProvider",
    "OpenRouterWebSearchAdapter",
    "WebSearchModel",
]
