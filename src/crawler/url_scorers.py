"""Extensible URL scorer factory for deep crawl strategies."""
from typing import Any, Callable, Dict, List, Optional

from crawl4ai import KeywordRelevanceScorer

ScorerBuilder = Callable[..., Any]


class UrlScorerFactory:
    """Registry-backed scorer factory to allow safe extensibility."""

    def __init__(self) -> None:
        self._builders: Dict[str, ScorerBuilder] = {}

    def register(self, name: str, builder: ScorerBuilder) -> None:
        normalized = (name or "").strip().lower()
        if not normalized:
            raise ValueError("Scorer name cannot be empty.")
        self._builders[normalized] = builder

    def supported(self) -> List[str]:
        return sorted(self._builders.keys())

    def build(
        self,
        scorer_type: str,
        *,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        normalized = (scorer_type or "").strip().lower()
        if not normalized or normalized == "none":
            return None

        builder = self._builders.get(normalized)
        if builder is None:
            # Safe fallback to keyword scorer type for unknown values.
            builder = self._builders.get("keyword")
            normalized = "keyword"

        if builder is None:
            return None
        return builder(keywords=keywords, **kwargs)


def _build_keyword_scorer(*, keywords: Optional[List[str]] = None, **_: Any) -> Any:
    if not keywords:
        return None
    return KeywordRelevanceScorer(keywords=keywords)


_factory = UrlScorerFactory()
_factory.register("keyword", _build_keyword_scorer)


def build_url_scorer(
    scorer_type: str,
    *,
    keywords: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Create a URL scorer using a registry-backed, extensible abstraction."""
    return _factory.build(scorer_type, keywords=keywords, **kwargs)


def get_supported_scorer_types() -> List[str]:
    """Return supported scorer names from the registry."""
    return _factory.supported()


def register_url_scorer(name: str, builder: ScorerBuilder) -> None:
    """Register a custom scorer builder for future extensions."""
    _factory.register(name, builder)
