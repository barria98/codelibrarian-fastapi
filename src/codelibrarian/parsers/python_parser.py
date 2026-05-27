"""Python AST-based parser.

Extracts functions, methods, classes with signatures, docstrings, parameters,
return types, decorators, plus call graph and import graph edges.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Union

from codelibrarian.models import GraphEdges, Parameter, ParseResult, Symbol
from codelibrarian.parsers.base import BaseParser

FASTAPI_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}

_FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class PythonParser(BaseParser):
    def parse(self, file_path: Path, source: str, module_name: str) -> ParseResult:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return ParseResult(symbols=[], edges=GraphEdges())

        visitor = _Visitor(module_name, source)
        visitor.visit(tree)
        return ParseResult(symbols=visitor.symbols, edges=visitor.edges)


class _Visitor(ast.NodeVisitor):
    def __init__(self, module_name: str, source: str):
        self.module_name = module_name
        self.source = source
        self.symbols: list[Symbol] = []
        self.edges = GraphEdges()
        self._class_stack: list[str] = []  # stack of class qualified names

    # ------------------------------------------------------------------ #
    # Imports
    # ------------------------------------------------------------------ #

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.edges.imports.append((self.module_name, alias.name, None))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.edges.imports.append((self.module_name, module, alias.name))
        self.generic_visit(node)

    # ------------------------------------------------------------------ #
    # Classes
    # ------------------------------------------------------------------ #

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualify(node.name)
        parent_qn = self._class_stack[-1] if self._class_stack else None

        sig = _class_signature(node)
        doc = ast.get_docstring(node) or ""

        sym = Symbol(
            name=node.name,
            qualified_name=qualified,
            kind="class",
            file_path="",  # filled in by indexer
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=sig,
            docstring=doc,
            decorators=[_decorator_name(d) for d in node.decorator_list],
            parent_qualified_name=parent_qn,
        )
        self.symbols.append(sym)

        for base in node.bases:
            base_name = _expr_to_name(base)
            if base_name:
                self.edges.inherits.append((qualified, base_name))

        self._class_stack.append(qualified)
        self.generic_visit(node)
        self._class_stack.pop()

    # ------------------------------------------------------------------ #
    # Functions / Methods
    # ------------------------------------------------------------------ #

    def visit_FunctionDef(self, node: _FuncNode) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: _FuncNode) -> None:
        self._visit_func(node)

    def _visit_func(self, node: _FuncNode) -> None:
        kind = "method" if self._class_stack else "function"
        qualified = self._qualify(node.name)
        parent_qn = self._class_stack[-1] if self._class_stack else None

        params = _extract_params(node)
        return_type = _annotation_to_str(node.returns) if node.returns else None
        sig = _build_signature(node, params, return_type)
        doc = ast.get_docstring(node) or ""
        decs = [_decorator_name(d) for d in node.decorator_list]

        http_method = None
        route = None

        for d in node.decorator_list:
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                attr_name = d.func.attr.lower()
                if attr_name in FASTAPI_HTTP_METHODS:
                    kind = "fastapi_endpoint"
                    http_method = attr_name.upper()
                    if d.args and isinstance(d.args[0], ast.Constant):
                        route = str(d.args[0].value)
                    break

        sym = Symbol(
            name=node.name,
            qualified_name=qualified,
            kind=kind,  # type: ignore[arg-type]
            file_path="",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=sig,
            docstring=doc,
            parameters=params,
            return_type=return_type,
            decorators=decs,
            parent_qualified_name=parent_qn,
            http_method=http_method,
            route=route,
        )
        self.symbols.append(sym)

        # Extract calls within this function body.
        # We visit each body statement individually rather than the FunctionDef
        # node itself, because _CallExtractor.visit_FunctionDef is a no-op
        # (to avoid recursing into nested function definitions).
        call_extractor = _CallExtractor()
        for stmt in node.body:
            call_extractor.visit(stmt)
        for callee in call_extractor.calls:
            self.edges.calls.append((qualified, callee))

        # Synthetic edge: make the FastAPI endpoint a caller of its own function
        if kind == "fastapi_endpoint" and node.name not in call_extractor.calls:
            self.edges.calls.append((qualified, node.name))

        # Visit nested classes/functions without descending into calls again
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _qualify(self, name: str) -> str:
        if self._class_stack:
            return f"{self._class_stack[-1]}.{name}"
        return f"{self.module_name}.{name}"


class _CallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _expr_to_name(node.func)
        if name:
            self.calls.append(name)
        # Don't recurse into nested function defs
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # don't recurse into nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _expr_to_name(node: ast.expr) -> str | None:
    """Convert an AST expression to a dotted name string, or None if not representable."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        val = _expr_to_name(node.value)
        return f"{val}.{node.attr}" if val else node.attr
    return None


def _annotation_to_str(node: ast.expr) -> str:
    """Unparse an annotation AST node to its source representation."""
    return ast.unparse(node)


def _decorator_name(node: ast.expr) -> str:
    """Return a human-readable name for a decorator expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expr_to_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ast.unparse(node)


def _extract_params(node: _FuncNode) -> list[Parameter]:
    """Extract parameter metadata from a function/method AST node."""
    params: list[Parameter] = []
    args = node.args
    defaults = args.defaults
    num_args = len(args.args)
    # defaults list is right-aligned against args.args
    defaults_offset = num_args - len(defaults)

    for i, arg in enumerate(args.args):
        if arg.arg in ("self", "cls"):
            continue
        type_str = _annotation_to_str(arg.annotation) if arg.annotation else None
        default_idx = i - defaults_offset
        default_str = ast.unparse(defaults[default_idx]) if default_idx >= 0 else None
        params.append(Parameter(name=arg.arg, type=type_str, default=default_str))

    for arg in args.posonlyargs:
        type_str = _annotation_to_str(arg.annotation) if arg.annotation else None
        params.append(Parameter(name=arg.arg, type=type_str))

    if args.vararg:
        params.append(Parameter(name=f"*{args.vararg.arg}"))

    for arg in args.kwonlyargs:
        type_str = _annotation_to_str(arg.annotation) if arg.annotation else None
        params.append(Parameter(name=arg.arg, type=type_str))

    if args.kwarg:
        params.append(Parameter(name=f"**{args.kwarg.arg}"))

    return params


def _build_signature(
    node: _FuncNode,
    params: list[Parameter],
    return_type: str | None,
) -> str:
    """Build a human-readable function signature string from its AST node."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    param_parts: list[str] = []
    for p in params:
        part = p.name
        if p.type:
            part += f": {p.type}"
        if p.default is not None:
            part += f" = {p.default}"
        param_parts.append(part)
    sig = f"{prefix} {node.name}({', '.join(param_parts)})"
    if return_type:
        sig += f" -> {return_type}"
    return sig


def _class_signature(node: ast.ClassDef) -> str:
    """Build a human-readable class signature string from its AST node."""
    bases = [ast.unparse(b) for b in node.bases]
    if bases:
        return f"class {node.name}({', '.join(bases)})"
    return f"class {node.name}"
