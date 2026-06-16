"""Tests for Mermaid diagram generation."""

import pytest

from codelibrarian.models import Parameter, Symbol
from codelibrarian.storage.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    db = SQLiteStore(tmp_path / "test.db", embedding_dimensions=4)
    with db:
        db.init_schema()
        yield db


def _sym(name, qname, kind, file_path="/a/b.py", line=1, params=None, parent_qname=None):
    return Symbol(
        name=name,
        qualified_name=qname,
        kind=kind,
        file_path=file_path,
        line_start=line,
        line_end=line + 5,
        signature=f"def {name}()" if kind != "class" else f"class {name}",
        parameters=params or [],
        parent_qualified_name=parent_qname,
    )


def _setup_hierarchy(store):
    """Create Animal -> Dog, Animal -> Cat hierarchy with methods."""
    fid = store.upsert_file("/a/b.py", "b.py", "python", 1.0, "x")

    animal_id = store.insert_symbol(_sym("Animal", "m.Animal", "class"), fid, None)
    store.insert_symbol(
        _sym("speak", "m.Animal.speak", "method",
             params=[Parameter("self", None)],
             parent_qname="m.Animal"),
        fid, animal_id,
    )

    dog_id = store.insert_symbol(_sym("Dog", "m.Dog", "class"), fid, None)
    store.insert_symbol(
        _sym("fetch", "m.Dog.fetch", "method",
             params=[Parameter("self", None), Parameter("item", "str")],
             parent_qname="m.Dog"),
        fid, dog_id,
    )

    cat_id = store.insert_symbol(_sym("Cat", "m.Cat", "class"), fid, None)

    store.insert_inherit(dog_id, "m.Animal")
    store.insert_inherit(cat_id, "m.Animal")
    store.resolve_graph_edges()
    store.conn.commit()
    return fid


def _setup_call_chain(store):
    """Create a -> b -> c call chain."""
    fid = store.upsert_file("/a/b.py", "b.py", "python", 1.0, "x")
    a_id = store.insert_symbol(_sym("a_fn", "m.a_fn", "function"), fid, None)
    b_id = store.insert_symbol(_sym("b_fn", "m.b_fn", "function"), fid, None)
    c_id = store.insert_symbol(_sym("c_fn", "m.c_fn", "function"), fid, None)
    store.conn.commit()

    store.insert_call(a_id, "m.b_fn")
    store.insert_call(b_id, "m.c_fn")
    store.resolve_graph_edges()
    store.conn.commit()
    return fid


# --------------------------------------------------------------------------- #
# Class diagram
# --------------------------------------------------------------------------- #


class TestClassDiagram:
    def test_contains_mermaid_header(self, store):
        _setup_hierarchy(store)
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "Animal")
        assert result.startswith("classDiagram")

    def test_shows_inheritance(self, store):
        _setup_hierarchy(store)
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "Animal")
        # Arrows use sanitized IDs; labels appear in ["..."] syntax
        assert "<|--" in result
        assert '"Animal"' in result
        assert '"Dog"' in result
        assert '"Cat"' in result

    def test_shows_methods(self, store):
        _setup_hierarchy(store)
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "Animal")
        assert "speak()" in result

    def test_shows_child_class_methods(self, store):
        _setup_hierarchy(store)
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "Animal")
        assert "fetch(item: str)" in result

    def test_unknown_class_returns_empty(self, store):
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "NoSuchClass")
        assert result == ""

    def test_class_without_methods_still_appears(self, store):
        _setup_hierarchy(store)
        from codelibrarian.diagrams import mermaid_class_diagram
        result = mermaid_class_diagram(store, "Animal")
        # Cat has no methods but should still appear in the diagram
        assert '"Cat"' in result


# --------------------------------------------------------------------------- #
# Call graph
# --------------------------------------------------------------------------- #


class TestCallGraph:
    def test_contains_mermaid_header(self, store):
        _setup_call_chain(store)
        from codelibrarian.diagrams import mermaid_call_graph
        result = mermaid_call_graph(store, "m.a_fn", depth=2, direction="callees")
        assert result.startswith("flowchart LR")

    def test_shows_edges(self, store):
        _setup_call_chain(store)
        from codelibrarian.diagrams import mermaid_call_graph
        result = mermaid_call_graph(store, "m.a_fn", depth=2, direction="callees")
        assert "a_fn" in result
        assert "b_fn" in result
        assert "c_fn" in result
        assert "-->" in result

    def test_depth_limits_edges(self, store):
        _setup_call_chain(store)
        from codelibrarian.diagrams import mermaid_call_graph
        result = mermaid_call_graph(store, "m.a_fn", depth=1, direction="callees")
        assert "a_fn" in result
        assert "b_fn" in result
        # c_fn should NOT appear at depth=1
        assert "c_fn" not in result

    def test_callers_direction(self, store):
        _setup_call_chain(store)
        from codelibrarian.diagrams import mermaid_call_graph
        result = mermaid_call_graph(store, "m.c_fn", depth=2, direction="callers")
        assert "a_fn" in result
        assert "b_fn" in result
        assert "c_fn" in result

    def test_unknown_symbol_returns_empty(self, store):
        from codelibrarian.diagrams import mermaid_call_graph
        result = mermaid_call_graph(store, "no.such.fn", depth=1, direction="callees")
        assert result == ""

    def test_cyclic_call_graph_does_not_crash(self, store):
        """Mutual recursion (A->B->A) should not cause infinite CTE recursion."""
        fid = store.upsert_file("/a/b.py", "b.py", "python", 1.0, "x")
        a_id = store.insert_symbol(_sym("alpha", "m.alpha", "function"), fid, None)
        b_id = store.insert_symbol(_sym("beta", "m.beta", "function"), fid, None)
        store.conn.commit()

        store.insert_call(a_id, "m.beta")
        store.insert_call(b_id, "m.alpha")
        store.resolve_graph_edges()
        store.conn.commit()

        from codelibrarian.diagrams import mermaid_call_graph
        # Should terminate without error and show both edges
        result = mermaid_call_graph(store, "m.alpha", depth=5, direction="callees")
        assert "alpha" in result
        assert "beta" in result
        assert "-->" in result


# --------------------------------------------------------------------------- #
# Import graph
# --------------------------------------------------------------------------- #


class TestImportGraph:
    def _setup_imports(self, store):
        fid1 = store.upsert_file("/a/mod_a.py", "src/mod_a.py", "python", 1.0, "x")
        fid2 = store.upsert_file("/a/mod_b.py", "src/mod_b.py", "python", 1.0, "y")
        store.insert_import(fid1, "mod_b")
        store.conn.execute(
            "UPDATE imports SET to_file_id = ? WHERE from_file_id = ?", (fid2, fid1)
        )
        store.conn.commit()
        return fid1, fid2

    def test_contains_mermaid_header(self, store):
        self._setup_imports(store)
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store)
        assert result.startswith("flowchart LR")

    def test_shows_file_edges(self, store):
        self._setup_imports(store)
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store)
        assert "mod_a" in result
        assert "mod_b" in result
        assert "-->" in result

    def test_scoped_to_file(self, store):
        self._setup_imports(store)
        fid3 = store.upsert_file("/a/mod_c.py", "lib/mod_c.py", "python", 1.0, "z")
        fid1 = store.conn.execute(
            "SELECT id FROM files WHERE relative_path = 'src/mod_a.py'"
        ).fetchone()["id"]
        store.insert_import(fid3, "mod_a")
        store.conn.execute(
            "UPDATE imports SET to_file_id = ? WHERE from_file_id = ? AND to_module = ?",
            (fid1, fid3, "mod_a"),
        )
        store.conn.commit()

        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store, file_path="src/mod_a.py")
        # Should include both edges involving mod_a
        assert "mod_a" in result
        assert "mod_b" in result
        assert "mod_c" in result

    def test_scoped_by_absolute_path(self, store):
        """Scoping accepts the stored absolute path, not just relative (issue #6)."""
        self._setup_imports(store)
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store, file_path="/a/mod_a.py")
        assert "mod_a" in result
        assert "mod_b" in result

    def test_scoped_by_basename(self, store):
        """Scoping accepts a bare filename (issue #6)."""
        self._setup_imports(store)
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store, file_path="mod_a.py")
        assert "mod_a" in result
        assert "mod_b" in result

    def test_empty_project_returns_empty(self, store):
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store)
        assert result == ""

    def test_subgraph_grouping(self, store):
        self._setup_imports(store)
        from codelibrarian.diagrams import mermaid_import_graph
        result = mermaid_import_graph(store)
        assert "subgraph" in result
        assert '"src"' in result


# --------------------------------------------------------------------------- #
# Sanitize ID collision safety
# --------------------------------------------------------------------------- #


class TestSanitizeId:
    def test_dot_and_underscore_produce_different_ids(self):
        from codelibrarian.diagrams import _sanitize_id
        id1 = _sanitize_id("foo.bar")
        id2 = _sanitize_id("foo_bar")
        assert id1 != id2
