"""Unit tests for extensible URL scorer factory."""

import os

import pytest

os.environ.setdefault("POSTGRES_URL", "postgresql://u:p@localhost:5432/testdb")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434/api/embeddings")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_DIM", "4")

from src.crawler import url_scorers


class TestUrlScorerFactory:
    def test_keyword_builder_with_keywords(self):
        scorer = url_scorers.build_url_scorer("keyword", keywords=["python"])
        assert scorer is not None

    def test_keyword_builder_without_keywords_returns_none(self):
        scorer = url_scorers.build_url_scorer("keyword", keywords=None)
        assert scorer is None

    def test_none_scorer_returns_none(self):
        assert url_scorers.build_url_scorer("none", keywords=["python"]) is None

    def test_unknown_scorer_falls_back_to_keyword(self):
        scorer = url_scorers.build_url_scorer("future-custom", keywords=["docs"])
        assert scorer is not None

    def test_supported_types_contains_keyword(self):
        assert "keyword" in url_scorers.get_supported_scorer_types()

    def test_register_custom_scorer(self):
        def _builder(**kwargs):
            return {"custom": True, "kwargs": kwargs}

        url_scorers.register_url_scorer("custom_demo", _builder)
        scorer = url_scorers.build_url_scorer("custom_demo", keywords=["x"], extra=1)
        assert scorer["custom"] is True
        assert scorer["kwargs"]["extra"] == 1

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError):
            url_scorers.register_url_scorer("", lambda **kwargs: kwargs)

    def test_factory_without_builders_returns_none(self):
        local_factory = url_scorers.UrlScorerFactory()
        assert local_factory.build("unknown", keywords=["x"]) is None
