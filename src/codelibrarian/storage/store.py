"""SQLite store: symbols, FTS5, sqlite-vec embeddings, and graph edges."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterator

import sqlite_vec

# --------------------------------------------------------------------------- #
# Query limits — use these constants everywhere instead of inline literals
# --------------------------------------------------------------------------- #

#: Maximum rows returned by exact/prefix symbol look-ups.
_LOOKUP_LIMIT: int = 20
#: Maximum rows returned by :meth:`SQLiteStore.list_symbols`.
_LIST_LIMIT: int = 200
#: Maximum symbols fetched per embedding batch cycle.
_EMBED_BATCH_CEILING: int = 1000
#: Maximum recursion depth for ancestor/descendant class-hierarchy CTEs.
_HIERARCHY_DEPTH: int = 5
#: Maximum symbol names injected into the LLM query-rewrite vocabulary prompt.
_VOCABULARY_LIMIT: int = 500

from codelibrarian.models import (
    GraphEdges,
    Parameter,
    ParseResult,
    Symbol,
    SymbolRecord,
)


def _top_qualifier(qualified_name: str) -> str:
    """Return the top-level module/namespace of a qualified name.

    Handles both dot (Python/TS/Swift) and ``::`` (Rust) separators, e.g.
    ``models.Dog.fetch`` -> ``models`` and ``crate::mod::f`` -> ``crate``.
    """
    return re.split(r"\.|::", qualified_name, maxsplit=1)[0]


# --------------------------------------------------------------------------- #
# Schema DDL
# --------------------------------------------------------------------------- #

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    relative_path TEXT NOT NULL,
    language      TEXT,
    last_modified REAL,
    content_hash  TEXT
);

CREATE TABLE IF NOT EXISTS symbols (
    id             INTEGER PRIMARY KEY,
    file_id        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind           TEXT NOT NULL,
    line_start     INTEGER,
    line_end       INTEGER,
    signature      TEXT,
    docstring      TEXT,
    parameters     TEXT DEFAULT '[]',
    return_type    TEXT,
    decorators     TEXT DEFAULT '[]',
    parent_id      INTEGER REFERENCES symbols(id),
    http_method    TEXT,
    route          TEXT
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    qualified_name,
    signature,
    docstring,
    content=symbols,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (new.id, new.name, new.qualified_name,
            COALESCE(new.signature, ''), COALESCE(new.docstring, ''));
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualified_name,
            COALESCE(old.signature, ''), COALESCE(old.docstring, ''));
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualified_name,
            COALESCE(old.signature, ''), COALESCE(old.docstring, ''));
    INSERT INTO symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (new.id, new.name, new.qualified_name,
            COALESCE(new.signature, ''), COALESCE(new.docstring, ''));
END;

CREATE TABLE IF NOT EXISTS imports (
    from_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_module     TEXT NOT NULL,
    to_file_id    INTEGER REFERENCES files(id),
    import_name   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (from_file_id, to_module, import_name)
);

CREATE TABLE IF NOT EXISTS calls (
    caller_id   INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_name TEXT NOT NULL,
    callee_id   INTEGER REFERENCES symbols(id),
    PRIMARY KEY (caller_id, callee_name)
);

CREATE TABLE IF NOT EXISTS inherits (
    child_id    INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    parent_name TEXT NOT NULL,
    parent_id   INTEGER REFERENCES symbols(id),
    PRIMARY KEY (child_id, parent_name)
);
"""

_VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_embeddings USING vec0(
    symbol_id INTEGER PRIMARY KEY,
    embedding float[{dimensions}]
);
"""


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class SQLiteStore:
    def __init__(self, db_path: Path, embedding_dimensions: int = 768):
        self.db_path = db_path
        self.embedding_dimensions = embedding_dimensions
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        try:
            conn.enable_load_extension(True)
        except AttributeError:
            raise RuntimeError(
                "Python's sqlite3 module was compiled without extension loading support. "
                "This is common with pyenv or macOS system Python.\n"
                "Fix: install with 'uv tool install /path/to/codelibrarian' which uses "
                "a compatible Python build, or rebuild Python with:\n"
                "  PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' "
                "pyenv install <version>"
            ) from None
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        self._conn = conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteStore":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store not connected. Use as context manager or call connect().")
        return self._conn

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.execute(
            _VEC_TABLE_SQL.format(dimensions=self.embedding_dimensions)
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_version VALUES (1)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #

    def get_file_hash(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row["content_hash"] if row else None

    def upsert_file(
        self,
        path: str,
        relative_path: str,
        language: str | None,
        last_modified: float,
        content_hash: str,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO files (path, relative_path, language, last_modified, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                relative_path  = excluded.relative_path,
                language       = excluded.language,
                last_modified  = excluded.last_modified,
                content_hash   = excluded.content_hash
            RETURNING id
            """,
            (path, relative_path, language, last_modified, content_hash),
        )
        row = cur.fetchone()
        return row[0]

    def delete_file_symbols(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM imports WHERE from_file_id = ?", (file_id,))
        # Delete embeddings for this file's symbols. The symbol_embeddings vec0
        # virtual table does not honour ON DELETE CASCADE, so without this the
        # vectors are orphaned. Worse, SQLite recycles symbol rowids, so a stale
        # embedding row could later be reused for an unrelated new symbol.
        self.conn.execute(
            "DELETE FROM symbol_embeddings WHERE symbol_id IN "
            "(SELECT id FROM symbols WHERE file_id = ?)",
            (file_id,),
        )
        # Clear resolved FK references from other tables pointing to symbols
        # in this file, so that deleting symbols doesn't violate FKs.
        self.conn.execute(
            "UPDATE calls SET callee_id = NULL WHERE callee_id IN "
            "(SELECT id FROM symbols WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute(
            "UPDATE inherits SET parent_id = NULL WHERE parent_id IN "
            "(SELECT id FROM symbols WHERE file_id = ?)",
            (file_id,),
        )
        # Delete child symbols first (those with parent_id set) to avoid
        # FK violations from the self-referencing parent_id column.
        self.conn.execute(
            "DELETE FROM symbols WHERE file_id = ? AND parent_id IS NOT NULL",
            (file_id,),
        )
        self.conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))

    def get_file_id(self, path: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row["id"] if row else None

    def list_files(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, path, relative_path, language, content_hash FROM files"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Symbols
    # ------------------------------------------------------------------ #

    def insert_symbol(self, sym: Symbol, file_id: int, parent_id: int | None) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO symbols
                (file_id, name, qualified_name, kind,
                 line_start, line_end, signature, docstring,
                 parameters, return_type, decorators, parent_id,
                 http_method, route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                sym.name,
                sym.qualified_name,
                sym.kind,
                sym.line_start,
                sym.line_end,
                sym.signature,
                sym.docstring,
                sym.parameters_json(),
                sym.return_type,
                sym.decorators_json(),
                parent_id,
                sym.http_method,
                sym.route,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def get_symbol_by_qualified_name(self, qualified_name: str) -> SymbolRecord | None:
        row = self.conn.execute(
            """
            SELECT s.*, f.path, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.qualified_name = ?
            """,
            (qualified_name,),
        ).fetchone()
        return SymbolRecord.from_row(dict(row)) if row else None

    def get_symbol_by_id(self, symbol_id: int) -> SymbolRecord | None:
        row = self.conn.execute(
            """
            SELECT s.*, f.path, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.id = ?
            """,
            (symbol_id,),
        ).fetchone()
        return SymbolRecord.from_row(dict(row)) if row else None

    def lookup_symbol(self, name: str) -> list[SymbolRecord]:
        rows = self.conn.execute(
            """
            SELECT s.*, f.path, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.name = ? OR s.qualified_name = ?
            ORDER BY length(s.qualified_name)
            LIMIT ?
            """,
            (name, name, _LOOKUP_LIMIT),
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    def lookup_symbol_prefix(self, name: str) -> list[SymbolRecord]:
        rows = self.conn.execute(
            """
            SELECT s.*, f.path, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE s.name LIKE ? OR s.qualified_name LIKE ?
            ORDER BY length(s.qualified_name)
            LIMIT ?
            """,
            (f"{name}%", f"%{name}%", _LOOKUP_LIMIT),
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    def list_symbols(
        self,
        kind: str | None = None,
        pattern: str | None = None,
        file_path: str | None = None,
    ) -> list[SymbolRecord]:
        conditions: list[str] = []
        params: list = []

        if kind:
            conditions.append("s.kind = ?")
            params.append(kind)
        if pattern:
            conditions.append("s.name LIKE ?")
            params.append(pattern)
        if file_path:
            conditions.append("f.path = ?")
            params.append(file_path)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(_LIST_LIMIT)
        rows = self.conn.execute(
            f"""
            SELECT s.*, f.path, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            {where}
            ORDER BY s.qualified_name
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    def get_methods_for_class(self, class_qualified_name: str) -> list[SymbolRecord]:
        """Return all methods belonging to a class, by parent qualified name."""
        rows = self.conn.execute(
            """
            SELECT s.*, f.path, f.relative_path
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            JOIN symbols parent ON s.parent_id = parent.id
            WHERE parent.qualified_name = ? AND s.kind = 'method'
            ORDER BY s.name
            """,
            (class_qualified_name,),
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    # ------------------------------------------------------------------ #
    # Full-text search
    # ------------------------------------------------------------------ #

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """Returns list of (symbol_id, bm25_score) sorted by relevance."""
        rows = self.conn.execute(
            """
            SELECT rowid, bm25(symbols_fts) AS score
            FROM symbols_fts
            WHERE symbols_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        # bm25 returns negative values; more negative = better match
        return [(r["rowid"], -r["score"]) for r in rows]

    # ------------------------------------------------------------------ #
    # Vector embeddings
    # ------------------------------------------------------------------ #

    def upsert_embedding(self, symbol_id: int, embedding: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO symbol_embeddings(symbol_id, embedding) VALUES (?, ?)",
            (symbol_id, sqlite_vec.serialize_float32(embedding)),
        )

    def vector_search(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[tuple[int, float]]:
        """Returns list of (symbol_id, distance) sorted by distance ascending."""
        rows = self.conn.execute(
            """
            SELECT symbol_id, distance
            FROM symbol_embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query_embedding), limit),
        ).fetchall()
        return [(r["symbol_id"], r["distance"]) for r in rows]

    def symbols_with_embeddings(self) -> set[int]:
        rows = self.conn.execute("SELECT symbol_id FROM symbol_embeddings").fetchall()
        return {r["symbol_id"] for r in rows}

    def symbols_without_embeddings(
        self, limit: int | None = _EMBED_BATCH_CEILING
    ) -> list[tuple[int, str, str]]:
        """Returns (id, signature, docstring) for symbols lacking embeddings.

        Pass ``limit=None`` to fetch every pending symbol (used by the indexer,
        which streams the work list in chunks itself).
        """
        sql = """
            SELECT s.id, COALESCE(s.signature, '') as signature,
                   COALESCE(s.docstring, '') as docstring
            FROM symbols s
            LEFT JOIN symbol_embeddings e ON s.id = e.symbol_id
            WHERE e.symbol_id IS NULL
        """
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(sql, params).fetchall()
        return [(r["id"], r["signature"], r["docstring"]) for r in rows]

    # ------------------------------------------------------------------ #
    # Graph edges
    # ------------------------------------------------------------------ #

    def insert_import(
        self,
        from_file_id: int,
        to_module: str,
        import_name: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO imports (from_file_id, to_module, import_name)
            VALUES (?, ?, ?)
            """,
            (from_file_id, to_module, import_name or ""),
        )

    def insert_call(self, caller_id: int, callee_name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO calls (caller_id, callee_name) VALUES (?, ?)",
            (caller_id, callee_name),
        )

    def insert_inherit(self, child_id: int, parent_name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO inherits (child_id, parent_name) VALUES (?, ?)",
            (child_id, parent_name),
        )

    def resolve_graph_edges(self) -> None:
        """Attempt to resolve callee/parent names to known symbol IDs."""
        self._resolve_calls()
        self.conn.execute(
            """
            UPDATE inherits SET parent_id = (
                SELECT id FROM symbols
                WHERE (qualified_name = inherits.parent_name
                    OR name = inherits.parent_name)
                  AND kind = 'class'
                LIMIT 1
            )
            WHERE parent_id IS NULL
            """
        )
        self.conn.execute(
            """
            UPDATE imports SET to_file_id = (
                SELECT id FROM files
                WHERE relative_path LIKE '%' || replace(imports.to_module, '.', '/') || '%'
                LIMIT 1
            )
            WHERE to_file_id IS NULL
            """
        )

    def _resolve_calls(self) -> None:
        """Resolve unresolved call edges to symbol IDs, disambiguating by locality.

        A bare callee name (or the last component of a dotted name like
        ``store.upsert_file``) can match many symbols. Rather than picking an
        arbitrary row (the old ``LIMIT 1`` behaviour), score candidates by:

        1. exact ``qualified_name`` match with the callee,
        2. residing in the same file as the caller,
        3. sharing the caller's top-level module/qualifier prefix.

        Ties break deterministically (shortest qualified_name, then lowest id)
        so resolution is reproducible. Calls with no candidate stay unresolved.
        """
        rows = self.conn.execute(
            """
            SELECT c.caller_id, c.callee_name,
                   s.file_id AS caller_file_id, s.qualified_name AS caller_qn
            FROM calls c
            JOIN symbols s ON c.caller_id = s.id
            WHERE c.callee_id IS NULL
            """
        ).fetchall()

        # Cache candidate lists keyed by (callee_name, lookup_key) to avoid
        # re-querying for repeated callee names.
        candidate_cache: dict[str, list[sqlite3.Row]] = {}

        for row in rows:
            callee_name = row["callee_name"]
            key = callee_name.rsplit(".", 1)[-1] if "." in callee_name else callee_name

            cache_key = f"{callee_name}\x00{key}"
            candidates = candidate_cache.get(cache_key)
            if candidates is None:
                candidates = self.conn.execute(
                    """
                    SELECT id, file_id, qualified_name
                    FROM symbols
                    WHERE qualified_name = ? OR name = ?
                    """,
                    (callee_name, key),
                ).fetchall()
                candidate_cache[cache_key] = candidates

            best = self._best_call_candidate(
                candidates, callee_name, row["caller_file_id"], row["caller_qn"]
            )
            if best is not None:
                self.conn.execute(
                    "UPDATE calls SET callee_id = ? "
                    "WHERE caller_id = ? AND callee_name = ?",
                    (best, row["caller_id"], callee_name),
                )

    @staticmethod
    def _best_call_candidate(
        candidates: list[sqlite3.Row],
        callee_name: str,
        caller_file_id: int,
        caller_qn: str | None,
    ) -> int | None:
        """Pick the best-scoring candidate symbol id, or None if there are none."""
        if not candidates:
            return None

        caller_prefix = _top_qualifier(caller_qn) if caller_qn else None

        def score(cand: sqlite3.Row) -> tuple:
            s = 0
            if cand["qualified_name"] == callee_name:
                s += 100
            if cand["file_id"] == caller_file_id:
                s += 10
            if caller_prefix and _top_qualifier(cand["qualified_name"]) == caller_prefix:
                s += 5
            # Sort key: highest score first, then deterministic tie-breaks.
            return (-s, len(cand["qualified_name"]), cand["id"])

        return min(candidates, key=score)["id"]

    def get_callers(self, qualified_name: str, depth: int = 1) -> list[SymbolRecord]:
        """Recursive CTE: find all symbols that (transitively) call this symbol."""
        rows = self.conn.execute(
            """
            WITH RECURSIVE caller_tree(id, depth) AS (
                SELECT c.caller_id, 1
                FROM calls c
                JOIN symbols s ON c.callee_id = s.id
                WHERE s.qualified_name = ? OR s.name = ?
                UNION
                SELECT c2.caller_id, ct.depth + 1
                FROM calls c2
                JOIN caller_tree ct ON c2.callee_id = ct.id
                WHERE ct.depth < ?
            )
            SELECT DISTINCT s.*, f.path, f.relative_path
            FROM caller_tree ct
            JOIN symbols s ON ct.id = s.id
            JOIN files f ON s.file_id = f.id
            """,
            (qualified_name, qualified_name, depth),
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    def get_callees(self, qualified_name: str, depth: int = 1) -> list[SymbolRecord]:
        """Recursive CTE: find all symbols (transitively) called by this symbol."""
        rows = self.conn.execute(
            """
            WITH RECURSIVE callee_tree(id, depth) AS (
                SELECT c.callee_id, 1
                FROM calls c
                JOIN symbols s ON c.caller_id = s.id
                WHERE (s.qualified_name = ? OR s.name = ?)
                  AND c.callee_id IS NOT NULL
                UNION
                SELECT c2.callee_id, ct.depth + 1
                FROM calls c2
                JOIN callee_tree ct ON c2.caller_id = ct.id
                WHERE ct.depth < ? AND c2.callee_id IS NOT NULL
            )
            SELECT DISTINCT s.*, f.path, f.relative_path
            FROM callee_tree ct
            JOIN symbols s ON ct.id = s.id
            JOIN files f ON s.file_id = f.id
            WHERE s.id IS NOT NULL
            """,
            (qualified_name, qualified_name, depth),
        ).fetchall()
        return [SymbolRecord.from_row(dict(r)) for r in rows]

    def get_call_edges(
        self,
        qualified_name: str,
        depth: int = 1,
        direction: str = "callees",
    ) -> list[tuple[str, str]]:
        """Return directed (caller_qname, callee_qname) edge pairs.

        *direction* is ``"callees"`` (forward from the root) or ``"callers"``
        (backward to the root).

        The depth limit is enforced by collecting the set of reachable
        *node* IDs via a depth-bounded CTE (which naturally deduplicates
        and terminates on cycles because ``UNION`` drops duplicate rows),
        then selecting all call edges that fall entirely within that set.
        """
        if direction == "callees":
            # First: collect all node IDs reachable within `depth` hops
            rows = self.conn.execute(
                """
                WITH RECURSIVE reachable(id, d) AS (
                    SELECT s.id, 0
                    FROM symbols s
                    WHERE s.qualified_name = ? OR s.name = ?
                    UNION
                    SELECT c.callee_id, r.d + 1
                    FROM calls c
                    JOIN reachable r ON c.caller_id = r.id
                    WHERE r.d < ? AND c.callee_id IS NOT NULL
                )
                SELECT DISTINCT
                    s1.qualified_name AS caller_qname,
                    s2.qualified_name AS callee_qname
                FROM calls c
                JOIN reachable r1 ON c.caller_id = r1.id
                JOIN reachable r2 ON c.callee_id = r2.id
                JOIN symbols s1 ON c.caller_id = s1.id
                JOIN symbols s2 ON c.callee_id = s2.id
                WHERE c.callee_id IS NOT NULL
                """,
                (qualified_name, qualified_name, depth),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                WITH RECURSIVE reachable(id, d) AS (
                    SELECT s.id, 0
                    FROM symbols s
                    WHERE s.qualified_name = ? OR s.name = ?
                    UNION
                    SELECT c.caller_id, r.d + 1
                    FROM calls c
                    JOIN reachable r ON c.callee_id = r.id
                    WHERE r.d < ? AND c.caller_id IS NOT NULL
                )
                SELECT DISTINCT
                    s1.qualified_name AS caller_qname,
                    s2.qualified_name AS callee_qname
                FROM calls c
                JOIN reachable r1 ON c.caller_id = r1.id
                JOIN reachable r2 ON c.callee_id = r2.id
                JOIN symbols s1 ON c.caller_id = s1.id
                JOIN symbols s2 ON c.callee_id = s2.id
                WHERE c.caller_id IS NOT NULL
                """,
                (qualified_name, qualified_name, depth),
            ).fetchall()
        return [(r["caller_qname"], r["callee_qname"]) for r in rows]

    def resolve_relative_path(self, path: str) -> str | None:
        """Map an arbitrary file path to a stored ``relative_path``.

        Diagram scoping stores edges keyed by ``relative_path``, but callers
        may pass an absolute path, a path relative to CWD, or just a basename.
        Tries, in order: exact match on ``relative_path`` or absolute ``path``;
        the resolved absolute path; then a unique suffix match.
        """
        row = self.conn.execute(
            "SELECT relative_path FROM files WHERE relative_path = ? OR path = ? LIMIT 1",
            (path, path),
        ).fetchone()
        if row:
            return row["relative_path"]

        try:
            resolved = str(Path(path).resolve())
        except OSError:
            resolved = path
        row = self.conn.execute(
            "SELECT relative_path FROM files WHERE path = ? LIMIT 1", (resolved,)
        ).fetchone()
        if row:
            return row["relative_path"]

        # Suffix match (e.g. basename or a trailing path fragment). Prefer the
        # shortest relative_path so an exact filename wins over a longer path.
        like = f"%{path}"
        rows = self.conn.execute(
            "SELECT relative_path FROM files "
            "WHERE relative_path LIKE ? OR path LIKE ? "
            "ORDER BY length(relative_path) LIMIT 1",
            (like, like),
        ).fetchone()
        return rows["relative_path"] if rows else None

    def get_all_import_edges(self) -> list[tuple[str, str]]:
        """Return all resolved file-to-file import edges as (from_path, to_path)."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT f1.relative_path AS from_path, f2.relative_path AS to_path
            FROM imports i
            JOIN files f1 ON i.from_file_id = f1.id
            JOIN files f2 ON i.to_file_id = f2.id
            WHERE i.to_file_id IS NOT NULL
            ORDER BY from_path, to_path
            """
        ).fetchall()
        return [(r["from_path"], r["to_path"]) for r in rows]

    def get_file_imports(self, file_path: str) -> dict:
        file_id = self.get_file_id(file_path)
        if not file_id:
            return {"imports": [], "imported_by": []}

        imports = self.conn.execute(
            """
            SELECT to_module, import_name, f.relative_path as resolved_path
            FROM imports i
            LEFT JOIN files f ON i.to_file_id = f.id
            WHERE i.from_file_id = ?
            ORDER BY to_module
            """,
            (file_id,),
        ).fetchall()

        imported_by = self.conn.execute(
            """
            SELECT f.path, f.relative_path
            FROM imports i
            JOIN files f ON i.from_file_id = f.id
            WHERE i.to_file_id = ?
            """,
            (file_id,),
        ).fetchall()

        return {
            "imports": [dict(r) for r in imports],
            "imported_by": [dict(r) for r in imported_by],
        }

    def get_class_hierarchy(self, class_name: str) -> dict:
        # Find the class
        cls_rows = self.conn.execute(
            """
            SELECT s.id, s.qualified_name, f.relative_path
            FROM symbols s JOIN files f ON s.file_id = f.id
            WHERE (s.name = ? OR s.qualified_name = ?) AND s.kind = 'class'
            """,
            (class_name, class_name),
        ).fetchall()

        if not cls_rows:
            return {"class": None, "parents": [], "children": []}

        cls = dict(cls_rows[0])

        parents = self.conn.execute(
            f"""
            WITH RECURSIVE ancestor(id, depth) AS (
                SELECT i.parent_id, 1
                FROM inherits i
                WHERE i.child_id = ?
                UNION
                SELECT i2.parent_id, a.depth + 1
                FROM inherits i2
                JOIN ancestor a ON i2.child_id = a.id
                WHERE a.depth < {_HIERARCHY_DEPTH}
            )
            SELECT DISTINCT s.name, s.qualified_name, f.relative_path
            FROM ancestor a
            JOIN symbols s ON a.id = s.id
            JOIN files f ON s.file_id = f.id
            """,
            (cls["id"],),
        ).fetchall()

        children = self.conn.execute(
            f"""
            WITH RECURSIVE descendant(id, depth) AS (
                SELECT i.child_id, 1
                FROM inherits i
                WHERE i.parent_id = ?
                UNION
                SELECT i2.child_id, d.depth + 1
                FROM inherits i2
                JOIN descendant d ON i2.parent_id = d.id
                WHERE d.depth < {_HIERARCHY_DEPTH}
            )
            SELECT DISTINCT s.name, s.qualified_name, f.relative_path
            FROM descendant d
            JOIN symbols s ON d.id = s.id
            JOIN files f ON s.file_id = f.id
            """,
            (cls["id"],),
        ).fetchall()

        return {
            "class": cls,
            "parents": [dict(r) for r in parents],
            "children": [dict(r) for r in children],
        }

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def get_symbol_vocabulary(self, limit: int = _VOCABULARY_LIMIT) -> list[str]:
        """Return the most common symbol names, excluding test and dunder symbols.

        The result feeds the LLM query-rewrite prompt, so it must stay bounded:
        injecting every name in a large codebase can overflow the model's
        context window and blow past the rewrite timeout. Names are ranked by
        frequency (how many symbols share the name) so the most representative
        identifiers are kept when truncating to *limit*.
        """
        rows = self.conn.execute(
            "SELECT name, COUNT(*) AS freq FROM symbols "
            "WHERE name NOT LIKE 'test_%' "
            "GROUP BY name "
            "ORDER BY freq DESC, name"
        ).fetchall()
        vocab = [r["name"] for r in rows if not r["name"].startswith("__")]
        return vocab[:limit]

    def stats(self) -> dict:
        counts = {}
        for kind in ("function", "method", "class", "module"):
            row = self.conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE kind = ?", (kind,)
            ).fetchone()
            counts[kind] = row[0]

        file_count = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        embed_count = self.conn.execute(
            "SELECT COUNT(*) FROM symbol_embeddings"
        ).fetchone()[0]

        return {
            "files": file_count,
            "symbols": counts,
            "embeddings": embed_count,
        }
