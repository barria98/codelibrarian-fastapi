"""Indexer: orchestrates file discovery, parsing, storage, and embedding."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

from codelibrarian.config import Config
from codelibrarian.embeddings import EmbeddingClient
from codelibrarian.models import GraphEdges, Symbol
from codelibrarian.parsers import get_parser
from codelibrarian.parsers.base import BaseParser
from codelibrarian.storage.store import SQLiteStore


class IndexStats:
    """Counters and error log collected during one indexing run."""

    def __init__(self) -> None:
        self.files_scanned: int = 0
        self.files_indexed: int = 0
        self.files_skipped: int = 0
        self.symbols_added: int = 0
        self.embeddings_added: int = 0
        self.errors: list[str] = []

    def __str__(self) -> str:
        return (
            f"Scanned: {self.files_scanned}, "
            f"Indexed: {self.files_indexed}, "
            f"Skipped (unchanged): {self.files_skipped}, "
            f"Symbols: {self.symbols_added}, "
            f"Embeddings: {self.embeddings_added}"
        )


class Indexer:
    def __init__(
        self,
        store: SQLiteStore,
        config: Config,
        embedder: EmbeddingClient | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ):
        self.store = store
        self.config = config
        self.embedder = embedder
        self.progress = progress_cb or (lambda _: None)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def index_root(self, full: bool = False, reembed: bool = False) -> IndexStats:
        """Index the entire project root."""
        root = self.config.index_root
        files = list(self._discover_files(root))
        return self._index_files(files, root, full=full, reembed=reembed)

    def index_files(self, file_paths: list[str], full: bool = False) -> IndexStats:
        """Index a specific list of files (e.g. from git hooks)."""
        root = self.config.index_root
        paths = [Path(p).resolve() for p in file_paths if Path(p).exists()]
        return self._index_files(paths, root, full=full)

    # ------------------------------------------------------------------ #
    # File discovery
    # ------------------------------------------------------------------ #

    def _discover_files(self, root: Path) -> list[Path]:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath_obj = Path(dirpath)

            # Filter excluded directories in-place to prevent os.walk descending
            dirnames[:] = [
                d for d in dirnames
                if not self.config.is_excluded(dirpath_obj / d)
            ]

            for fname in filenames:
                fpath = dirpath_obj / fname
                if self.config.is_excluded(fpath):
                    continue
                lang = self.config.language_for_file(fpath)
                if lang:
                    files.append(fpath)
        return files

    # ------------------------------------------------------------------ #
    # Core indexing loop
    # ------------------------------------------------------------------ #

    def _index_files(
        self,
        files: list[Path],
        root: Path,
        full: bool = False,
        reembed: bool = False,
    ) -> IndexStats:
        stats = IndexStats()
        stats.files_scanned = len(files)

        # Maps qualified_name -> symbol_id, built during this run for graph resolution
        qualified_to_id: dict[str, int] = {}

        for fpath in files:
            try:
                result = self._index_single_file(fpath, root, full, qualified_to_id)
                if result is None:
                    stats.files_skipped += 1
                else:
                    stats.files_indexed += 1
                    stats.symbols_added += result
            except Exception as exc:
                stats.errors.append(f"{fpath}: {exc}")
                self.progress(f"ERROR {fpath}: {exc}")

        self.store.conn.commit()

        # Resolve graph edges after all files are indexed
        self.store.resolve_graph_edges()
        self.store.conn.commit()

        # Embeddings pass
        if self.embedder and self.config.embeddings_enabled:
            stats.embeddings_added = self._embed_pending(reembed)
            self.store.conn.commit()

        return stats

    def _index_single_file(
        self,
        fpath: Path,
        root: Path,
        full: bool,
        qualified_to_id: dict[str, int],
    ) -> int | None:
        """Index one file. Returns number of symbols inserted, or None if skipped."""
        lang = self.config.language_for_file(fpath)
        if not lang:
            return None

        content_hash = _file_hash(fpath)
        path_str = str(fpath)

        if not full:
            existing_hash = self.store.get_file_hash(path_str)
            if existing_hash == content_hash:
                return None  # unchanged

        self.progress(f"Indexing {fpath.name}")

        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        try:
            rel_path = str(fpath.relative_to(root))
        except ValueError:
            rel_path = fpath.name

        module_name = BaseParser.derive_module_name(fpath, root)

        parser = get_parser(lang)
        if not parser:
            return None

        parse_result = parser.parse(fpath, source, module_name)

        with self.store.conn:
            file_id = self.store.upsert_file(
                path=path_str,
                relative_path=rel_path,
                language=lang,
                last_modified=fpath.stat().st_mtime,
                content_hash=content_hash,
            )

            # Remove old symbols for this file
            self.store.delete_file_symbols(file_id)

            # Build a map of qualified_name -> parent_id for this file
            # We need to insert parents before children
            parent_id_map: dict[str, int] = {}
            symbol_count = 0

            for sym in parse_result.symbols:
                sym.file_path = path_str
                parent_id = None
                if sym.parent_qualified_name:
                    parent_id = (
                        parent_id_map.get(sym.parent_qualified_name)
                        or qualified_to_id.get(sym.parent_qualified_name)
                    )
                sym_id = self.store.insert_symbol(sym, file_id, parent_id)
                parent_id_map[sym.qualified_name] = sym_id
                qualified_to_id[sym.qualified_name] = sym_id
                symbol_count += 1

            # Insert graph edges
            for from_qn, to_module, import_name in parse_result.edges.imports:
                self.store.insert_import(file_id, to_module, import_name)

            for caller_qn, callee_name in parse_result.edges.calls:
                if _is_noise_call(callee_name):
                    continue
                caller_id = (
                    parent_id_map.get(caller_qn)
                    or qualified_to_id.get(caller_qn)
                )
                if caller_id:
                    self.store.insert_call(caller_id, callee_name)

            for child_qn, parent_name in parse_result.edges.inherits:
                child_id = (
                    parent_id_map.get(child_qn)
                    or qualified_to_id.get(child_qn)
                )
                if child_id:
                    self.store.insert_inherit(child_id, parent_name)

        return symbol_count

    # ------------------------------------------------------------------ #
    # Embedding pass
    # ------------------------------------------------------------------ #

    def _embed_pending(self, reembed: bool = False) -> int:
        if reembed:
            # Drop and recreate the vec0 table
            self.store.conn.execute("DROP TABLE IF EXISTS symbol_embeddings")
            from codelibrarian.storage.store import _VEC_TABLE_SQL
            self.store.conn.execute(
                _VEC_TABLE_SQL.format(dimensions=self.store.embedding_dimensions)
            )
            self.store.conn.commit()

        # Fetch the full pending set once and build a work list of symbols that
        # actually have embeddable text. We deliberately do NOT re-query inside
        # the loop: the old approach relied on stored vectors dropping rows out
        # of the pending query, so any symbol that failed to embed (API error,
        # empty text) was re-fetched forever — an infinite loop.
        pending = self.store.symbols_without_embeddings(limit=None)
        work: list[tuple[int, str]] = []
        for sym_id, signature, docstring in pending:
            # Skip symbols with no embeddable text: an empty string is wasted
            # work and some embedding servers reject empty input outright.
            text = (f"{signature}\n{docstring}").strip()
            if text:
                work.append((sym_id, text))

        count = 0
        chunk_size = max(self.config.embedding_batch_size * 4, 1)
        for batch_num, start in enumerate(range(0, len(work), chunk_size), start=1):
            chunk = work[start : start + chunk_size]
            self.progress(f"Embedding batch {batch_num} ({len(chunk)} symbols)")

            ids = [sid for sid, _ in chunk]
            texts = [text for _, text in chunk]

            embeddings = self.embedder.embed_texts(texts)  # type: ignore[union-attr]
            for sym_id, embedding in zip(ids, embeddings):
                if embedding is not None:
                    self.store.upsert_embedding(sym_id, embedding)
                    count += 1

        return count


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Call-graph noise filter
# --------------------------------------------------------------------------- #

# Python builtins and common exception constructors
_BUILTIN_NAMES: frozenset[str] = frozenset(
    # functions
    "len any all isinstance issubclass print range int str list dict set tuple "
    "type super max min sorted zip map filter next iter open enumerate hasattr "
    "getattr setattr delattr repr hash id abs round sum bool float bytes chr "
    "ord hex oct bin format reversed input vars dir callable staticmethod "
    "classmethod property frozenset complex memoryview bytearray object slice "
    # stdlib constructors commonly used as calls
    "Path "
    # exception types
    "ValueError TypeError RuntimeError KeyError AttributeError ImportError "
    "FileNotFoundError OSError Exception StopIteration NotImplementedError "
    "IndexError NameError IOError PermissionError TimeoutError "
    "AssertionError ZeroDivisionError OverflowError RecursionError "
    "UnicodeDecodeError UnicodeEncodeError".split()
)

# Methods commonly found on built-in types (list, dict, str, set, Path, etc.)
_BUILTIN_METHODS: frozenset[str] = frozenset(
    # list / deque
    "append extend insert remove pop clear copy sort reverse count index "
    # dict
    "get items keys values update setdefault popitem fromkeys "
    # set
    "add discard union intersection difference symmetric_difference "
    "issubset issuperset "
    # str
    "strip lstrip rstrip split rsplit join replace startswith endswith "
    "lower upper find rfind encode decode format center ljust rjust zfill "
    "title capitalize swapcase isdigit isalpha isalnum isupper islower "
    "removeprefix removesuffix partition rpartition expandtabs "
    # bytes
    "hex "
    # pathlib.Path
    "exists mkdir resolve is_absolute relative_to stat read_text read_bytes "
    "write_text write_bytes is_file is_dir glob rglob suffix stem parent "
    # hashlib
    "hexdigest digest "
    # file / io
    "read write close flush seek tell readline readlines writelines "
    # sqlite3 cursor/connection
    "fetchone fetchall fetchmany execute executemany executescript commit "
    "rollback "
    # httpx / requests response
    "raise_for_status json "
    # generic iteration / context
    "visit generic_visit ".split()
)

# Known stdlib / third-party module prefixes — calls into these are external
_EXTERNAL_PREFIXES: tuple[str, ...] = (
    "ast.", "os.", "sys.", "json.", "re.", "hashlib.", "shutil.", "logging.",
    "importlib.", "asyncio.", "pathlib.", "sqlite_vec.", "click.", "pytest.",
    "httpx.", "math.", "collections.", "functools.", "itertools.", "typing.",
    "subprocess.", "threading.", "multiprocessing.", "socket.", "urllib.",
    "contextlib.", "dataclasses.", "inspect.", "textwrap.", "fnmatch.",
)


def _is_noise_call(callee_name: str) -> bool:
    """Return True if *callee_name* is a call to a builtin, stdlib, or
    external dependency that will never resolve to a project symbol."""
    # Simple name: check builtins and builtin type methods
    if callee_name in _BUILTIN_NAMES or callee_name in _BUILTIN_METHODS:
        return True

    # Dotted name: check module prefix and suffix
    if "." in callee_name:
        # Known external module prefix
        for prefix in _EXTERNAL_PREFIXES:
            if callee_name.startswith(prefix):
                return True
        # Last component is a builtin method (e.g. self.items.append)
        suffix = callee_name.rsplit(".", 1)[-1]
        if suffix in _BUILTIN_METHODS:
            return True

    return False
