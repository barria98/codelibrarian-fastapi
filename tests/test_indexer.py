"""End-to-end indexer tests using the Python sample fixture."""

from pathlib import Path

import pytest

from codelibrarian.config import Config
from codelibrarian.indexer import Indexer, _is_noise_call
from codelibrarian.storage.store import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures" / "python_sample"


@pytest.fixture
def config_and_store(tmp_path):
    config_dir = tmp_path / ".codelibrarian"
    config_dir.mkdir()

    config = Config(
        data={
            "index": {
                "root": str(FIXTURES),
                "exclude": ["__pycache__/", ".git/"],
                "languages": ["python"],
            },
            "embeddings": {
                "api_url": "http://localhost:11434/v1/embeddings",
                "model": "nomic-embed-text-v2-moe",
                "dimensions": 4,
                "batch_size": 32,
                "max_chars": 1600,
                "enabled": False,
            },
            "database": {"path": str(tmp_path / "test.db")},
        },
        config_dir=config_dir,
    )

    store = SQLiteStore(config.db_path, embedding_dimensions=4)
    store.connect()
    store.init_schema()
    return config, store


class _FailingEmbedder:
    """Embedder stub whose batches always fail (returns None per item)."""

    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [None] * len(texts)


class _OkEmbedder:
    """Embedder stub returning a fixed-dimension vector per non-empty text."""

    def __init__(self, dim=4):
        self.dim = dim
        self.seen_texts: list[str] = []

    def embed_texts(self, texts):
        self.seen_texts.extend(texts)
        return [[0.1] * self.dim for _ in texts]


def test_embed_pending_terminates_on_persistent_failure(config_and_store):
    """A consistently-failing embedder must not loop forever (issue #3)."""
    config, store = config_and_store
    config._data["embeddings"]["enabled"] = True
    embedder = _FailingEmbedder()
    indexer = Indexer(store, config, embedder=embedder)

    stats = indexer.index_root()  # must return rather than hang

    assert stats.embeddings_added == 0
    assert embedder.calls >= 1
    store.close()


def test_embed_pending_skips_empty_text(config_and_store):
    """Symbols with no signature/docstring text are not sent to the embedder (issue #3)."""
    config, store = config_and_store
    config._data["embeddings"]["enabled"] = True
    embedder = _OkEmbedder(dim=4)
    indexer = Indexer(store, config, embedder=embedder)

    indexer.index_root()

    assert all(t.strip() for t in embedder.seen_texts)
    store.close()


def test_indexer_finds_symbols(config_and_store):
    config, store = config_and_store
    indexer = Indexer(store, config)
    stats = indexer.index_root()

    assert stats.files_indexed >= 1
    assert stats.symbols_added > 0

    results = store.lookup_symbol("Animal")
    assert len(results) > 0
    assert results[0].kind == "class"

    store.close()


def test_indexer_incremental_skip(config_and_store):
    config, store = config_and_store
    indexer = Indexer(store, config)

    stats1 = indexer.index_root(full=False)
    store.conn.commit()
    stats2 = indexer.index_root(full=False)

    assert stats2.files_skipped >= stats1.files_indexed
    assert stats2.files_indexed == 0
    store.close()


def test_indexer_full_reindex(config_and_store):
    config, store = config_and_store
    indexer = Indexer(store, config)

    indexer.index_root(full=False)
    store.conn.commit()
    stats2 = indexer.index_root(full=True)

    assert stats2.files_indexed >= 1
    store.close()


def test_indexer_graph_edges(config_and_store):
    config, store = config_and_store
    indexer = Indexer(store, config)
    indexer.index_root()
    store.conn.commit()

    # Dog inherits Animal
    hierarchy = store.get_class_hierarchy("Dog")
    parent_names = [p["name"] for p in hierarchy["parents"]]
    assert "Animal" in parent_names

    store.close()


# --------------------------------------------------------------------------- #
# Noise call filtering
# --------------------------------------------------------------------------- #


def test_noise_call_filter_builtins():
    """Python builtins should be classified as noise."""
    assert _is_noise_call("len")
    assert _is_noise_call("isinstance")
    assert _is_noise_call("ValueError")
    assert _is_noise_call("any")


def test_noise_call_filter_builtin_methods():
    """Built-in type methods via attribute access or bare name should be noise."""
    assert _is_noise_call("self.items.append")
    assert _is_noise_call("results.extend")
    assert _is_noise_call("name.strip")
    assert _is_noise_call("self._animals.pop")
    # Bare method names (from tree-sitter extractors)
    assert _is_noise_call("fetchall")
    assert _is_noise_call("strip")
    assert _is_noise_call("join")


def test_noise_call_filter_external_modules():
    """Calls to known external modules should be noise."""
    assert _is_noise_call("ast.get_docstring")
    assert _is_noise_call("click.echo")
    assert _is_noise_call("json.dumps")
    assert _is_noise_call("re.sub")


def test_noise_call_filter_allows_project_calls():
    """Calls to project-internal symbols should NOT be noise."""
    assert not _is_noise_call("find_oldest")
    assert not _is_noise_call("self.store.upsert_file")
    assert not _is_noise_call("parser.parse")
    assert not _is_noise_call("_make_symbol")


def test_indexed_calls_exclude_noise(config_and_store):
    """After indexing, the calls table should not contain noise entries."""
    config, store = config_and_store
    indexer = Indexer(store, config)
    indexer.index_root()
    store.conn.commit()

    rows = store.conn.execute(
        "SELECT callee_name FROM calls"
    ).fetchall()
    callee_names = [r["callee_name"] for r in rows]

    # These noise calls exist in the fixture source but should be filtered
    for noise in ("len", "isinstance", "max", "enumerate",
                  "NotImplementedError", "self._animals.append",
                  "self._animals.pop"):
        assert noise not in callee_names, f"{noise} should have been filtered"

    # Project-internal call should still be present
    assert "find_oldest" in callee_names

    store.close()
