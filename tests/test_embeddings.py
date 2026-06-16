"""Tests for the EmbeddingClient -- all HTTP calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from codelibrarian.embeddings import EmbeddingClient


@pytest.fixture
def client():
    return EmbeddingClient(
        api_url="http://localhost:11434/v1/embeddings",
        model="test-model",
        dimensions=4,
    )


def _resp(data):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"data": data}
    return r


def test_embed_batch_returns_vectors_in_input_order(client):
    resp = _resp([
        {"embedding": [1.0], "index": 1},
        {"embedding": [0.0], "index": 0},
    ])
    with patch.object(client._client, "post", return_value=resp):
        out = client.embed_batch(["a", "b"])
    assert out == [[0.0], [1.0]]


def test_embed_batch_returns_none_on_count_mismatch(client):
    """A short response must fail rather than misalign vectors with inputs (issue #5)."""
    resp = _resp([{"embedding": [0.0], "index": 0}])
    with patch.object(client._client, "post", return_value=resp):
        out = client.embed_batch(["a", "b", "c"])
    assert out is None


def test_embed_texts_pads_none_for_failed_batch(client):
    with patch.object(client, "embed_batch", return_value=None):
        out = client.embed_texts(["a", "b"])
    assert out == [None, None]
