"""Tests des embeddings (HashingEmbedder — déterministe, hors-ligne)."""

from __future__ import annotations

import math

import pytest

from governed_omop_rag.retrieval.embeddings import Embedder, HashingEmbedder


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_hashing_embedder_is_protocol() -> None:
    emb = HashingEmbedder(64)
    assert isinstance(emb, Embedder)
    assert emb.dimension == 64
    assert emb.model_name == "hashing-64"


def test_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(0)


def test_embed_shapes_and_batch() -> None:
    emb = HashingEmbedder(128)
    vectors = emb.embed(["diabète de type 2", "asthme"])
    assert len(vectors) == 2
    assert all(len(v) == 128 for v in vectors)


def test_deterministic() -> None:
    emb = HashingEmbedder(128)
    assert emb.embed_one("diabète de type 2") == emb.embed_one("Diabète  de Type 2")


def test_vectors_are_l2_normalized() -> None:
    emb = HashingEmbedder(128)
    v = emb.embed_one("hypertension artérielle essentielle")
    assert _norm(v) == pytest.approx(1.0, abs=1e-9)


def test_empty_text_gives_zero_vector() -> None:
    emb = HashingEmbedder(32)
    v = emb.embed_one("   ")
    assert _norm(v) == 0.0


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = _norm(a), _norm(b)
    return dot / (na * nb) if na and nb else 0.0


def test_similar_texts_are_closer() -> None:
    emb = HashingEmbedder(512)
    base = emb.embed_one("diabète type 2")
    close = emb.embed_one("diabète de type 2")
    far = emb.embed_one("asthme aigu")
    assert _cos(base, close) > _cos(base, far)
