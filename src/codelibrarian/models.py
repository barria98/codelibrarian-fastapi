"""Shared data models used across the codelibrarian package."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


SymbolKind = Literal["function", "method", "class", "module"]


@dataclass
class Parameter:
    name: str
    type: str | None = None
    default: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "default": self.default}

    @classmethod
    def from_dict(cls, d: dict) -> "Parameter":
        return cls(name=d["name"], type=d.get("type"), default=d.get("default"))


@dataclass
class Symbol:
    """A parsed code symbol (function, method, class)."""

    name: str
    qualified_name: str
    kind: SymbolKind
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = field(default_factory=list)
    parent_qualified_name: str | None = None  # qualified_name of containing class

    def parameters_json(self) -> str:
        return json.dumps([p.to_dict() for p in self.parameters])

    def decorators_json(self) -> str:
        return json.dumps(self.decorators)


@dataclass
class GraphEdges:
    """Graph relationships extracted from a single file."""

    # (from_qualified_name, to_module, import_name_or_None)
    imports: list[tuple[str, str, str | None]] = field(default_factory=list)
    # (caller_qualified_name, callee_name)
    calls: list[tuple[str, str]] = field(default_factory=list)
    # (child_qualified_name, parent_name)
    inherits: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ParseResult:
    """Output from a parser for a single file."""

    symbols: list[Symbol]
    edges: GraphEdges


@dataclass
class RewrittenQuery:
    """Result of LLM-based query rewriting."""

    terms: list[str]
    focus: str = "all"  # "implementation", "tests", "all"


@dataclass
class FileRecord:
    id: int
    path: str
    relative_path: str
    language: str | None
    last_modified: float | None
    content_hash: str | None


@dataclass
class SymbolRecord:
    """A symbol as stored in and retrieved from the database."""

    id: int
    file_id: int
    name: str
    qualified_name: str
    kind: SymbolKind
    file_path: str
    relative_path: str
    line_start: int | None
    line_end: int | None
    signature: str | None
    docstring: str | None
    parameters: list[Parameter]
    return_type: str | None
    decorators: list[str]
    parent_id: int | None

    @classmethod
    def from_row(cls, row: dict) -> "SymbolRecord":
        params_raw = row.get("parameters") or "[]"
        decs_raw = row.get("decorators") or "[]"
        return cls(
            id=row["id"],
            file_id=row["file_id"],
            name=row["name"],
            qualified_name=row["qualified_name"],
            kind=row["kind"],
            file_path=row.get("path", ""),
            relative_path=row.get("relative_path", ""),
            line_start=row.get("line_start"),
            line_end=row.get("line_end"),
            signature=row.get("signature"),
            docstring=row.get("docstring"),
            parameters=[Parameter.from_dict(p) for p in json.loads(params_raw)],
            return_type=row.get("return_type"),
            decorators=json.loads(decs_raw),
            parent_id=row.get("parent_id"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "docstring": self.docstring,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "decorators": self.decorators,
        }


@dataclass
class SearchResult:
    symbol: SymbolRecord
    score: float
    match_type: Literal["semantic", "fulltext", "hybrid", "graph"]

    def to_dict(self) -> dict:
        d = self.symbol.to_dict()
        d["score"] = self.score
        d["match_type"] = self.match_type
        return d
