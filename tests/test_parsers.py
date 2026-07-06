"""Tests for Python and tree-sitter parsers."""

from pathlib import Path

import pytest

from codelibrarian.parsers.python_parser import PythonParser
from codelibrarian.parsers.treesitter_parser import TreeSitterParser

FIXTURES = Path(__file__).parent / "fixtures"
PYTHON_SAMPLE = FIXTURES / "python_sample" / "models.py"
TS_SAMPLE = FIXTURES / "typescript_sample" / "utils.ts"


# --------------------------------------------------------------------------- #
# Python parser
# --------------------------------------------------------------------------- #


@pytest.fixture
def py_result():
    parser = PythonParser()
    source = PYTHON_SAMPLE.read_text()
    return parser.parse(PYTHON_SAMPLE, source, "models")


def test_python_finds_classes(py_result):
    class_names = {s.name for s in py_result.symbols if s.kind == "class"}
    assert "Animal" in class_names
    assert "Dog" in class_names
    assert "Cat" in class_names
    assert "PetRecord" in class_names


def test_python_finds_methods(py_result):
    method_names = {s.name for s in py_result.symbols if s.kind == "method"}
    assert "speak" in method_names
    assert "fetch" in method_names
    assert "is_adopted" in method_names


def test_python_finds_functions(py_result):
    func_names = {s.name for s in py_result.symbols if s.kind == "function"}
    assert "find_oldest" in func_names


def test_python_qualified_names(py_result):
    qnames = {s.qualified_name for s in py_result.symbols}
    assert "models.Animal" in qnames
    assert "models.Animal.speak" in qnames
    assert "models.Dog.fetch" in qnames
    assert "models.find_oldest" in qnames


def test_python_signature(py_result):
    fetch = next(s for s in py_result.symbols if s.qualified_name == "models.Dog.fetch")
    assert "fetch" in fetch.signature
    assert "item" in fetch.signature


def test_python_docstring(py_result):
    animal = next(s for s in py_result.symbols if s.name == "Animal")
    assert "Base class" in animal.docstring


def test_python_parameters(py_result):
    fetch = next(s for s in py_result.symbols if s.qualified_name == "models.Dog.fetch")
    param_names = [p.name for p in fetch.parameters]
    assert "item" in param_names


def test_python_param_kinds_order_and_defaults():
    """Positional-only ordering and keyword-only defaults (issue #7)."""
    src = (
        "def f(a, b, /, c, d=1, *args, e, g=2, **kw):\n"
        "    pass\n"
    )
    parser = PythonParser()
    result = parser.parse(Path("mod.py"), src, "mod")
    fn = next(s for s in result.symbols if s.name == "f")

    by_name = {p.name: p for p in fn.parameters}
    names = [p.name for p in fn.parameters]
    # Canonical order: posonly, pos-or-kw, *args, kw-only, **kw
    assert names == ["a", "b", "c", "d", "*args", "e", "g", "**kw"]
    # positional default attaches to the right parameter
    assert by_name["d"].default == "1"
    assert by_name["c"].default is None
    # keyword-only defaults are captured (previously dropped)
    assert by_name["g"].default == "2"
    assert by_name["e"].default is None


def test_python_posonly_default_alignment():
    """A default on a positional-only param must not shift onto another (issue #7)."""
    parser = PythonParser()
    result = parser.parse(Path("mod.py"), "def g(x, y=5, /):\n    pass\n", "mod")
    fn = next(s for s in result.symbols if s.name == "g")
    by_name = {p.name: p for p in fn.parameters}
    assert by_name["x"].default is None
    assert by_name["y"].default == "5"


def test_python_return_type(py_result):
    fetch = next(s for s in py_result.symbols if s.qualified_name == "models.Dog.fetch")
    assert fetch.return_type == "str"


def test_python_inheritance_edges(py_result):
    inherits = py_result.edges.inherits
    child_names = {edge[0] for edge in inherits}
    assert "models.Dog" in child_names
    assert "models.Cat" in child_names


def test_python_import_edges(py_result):
    modules = {edge[1] for edge in py_result.edges.imports}
    assert "dataclasses" in modules
    assert "typing" in modules


def test_python_parent_qualified_name(py_result):
    fetch = next(s for s in py_result.symbols if s.qualified_name == "models.Dog.fetch")
    assert fetch.parent_qualified_name == "models.Dog"


# --------------------------------------------------------------------------- #
# TypeScript parser
# --------------------------------------------------------------------------- #


@pytest.fixture
def ts_result():
    try:
        import tree_sitter_typescript  # noqa: F401
    except ImportError:
        pytest.skip("tree-sitter-typescript not installed")
    parser = TreeSitterParser()
    source = TS_SAMPLE.read_text()
    return parser.parse(TS_SAMPLE, source, "utils")


def test_ts_finds_class(ts_result):
    class_names = {s.name for s in ts_result.symbols if s.kind == "class"}
    assert "UserService" in class_names


def test_ts_finds_methods(ts_result):
    method_names = {s.name for s in ts_result.symbols if s.kind == "method"}
    assert "addUser" in method_names
    assert "findById" in method_names


def test_ts_finds_functions(ts_result):
    func_names = {s.name for s in ts_result.symbols if s.kind == "function"}
    assert "formatDisplayName" in func_names or "fetchUser" in func_names
