"""Command-line interface for codelibrarian."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from codelibrarian.config import Config, DEFAULT_CONFIG_TOML


@click.group()
def main():
    """codelibrarian — self-maintaining code index for LLMs and humans."""


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #

@main.command()
@click.option("--path", default=".", help="Project root directory")
def init(path: str):
    """Initialise .codelibrarian/ in the project root."""
    root = Path(path).resolve()
    config_dir = root / ".codelibrarian"
    config_file = config_dir / "config.toml"

    if config_dir.exists():
        click.echo(f"Already initialised at {config_dir}")
    else:
        config_dir.mkdir(parents=True)
        click.echo(f"Created {config_dir}")

    if not config_file.exists():
        config_file.write_text(DEFAULT_CONFIG_TOML)
        click.echo(f"Created {config_file}")
    else:
        click.echo(f"Config already exists: {config_file}")

    # Create the database with schema
    config = Config.load(root)
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        store.init_schema()
    click.echo(f"Initialised database at {config.db_path}")
    click.echo("Done. Run 'codelibrarian index' to index the codebase.")


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #

@main.command()
@click.option("--full", is_flag=True, help="Reindex all files (ignore hash cache)")
@click.option("--reembed", is_flag=True, help="Regenerate all embeddings")
@click.option(
    "--files",
    multiple=True,
    help="Index specific files only (e.g. from git hooks)",
)
@click.option("--path", default=None, help="Project root (default: auto-detect)")
def index(full: bool, reembed: bool, files: tuple[str, ...], path: str | None):
    """Index the codebase."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    from codelibrarian.embeddings import EmbeddingClient
    from codelibrarian.indexer import Indexer
    from codelibrarian.storage.store import SQLiteStore

    embedder = None
    if config.embeddings_enabled:
        embedder = EmbeddingClient(
            api_url=config.embedding_api_url,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            batch_size=config.embedding_batch_size,
            max_chars=config.embedding_max_chars,
        )
        ok, msg = embedder.check_connection()
        if not ok:
            click.echo(f"Warning: embeddings disabled — {msg}", err=True)
            embedder.close()
            embedder = None

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        store.init_schema()
        indexer = Indexer(
            store=store,
            config=config,
            embedder=embedder,
            progress_cb=lambda msg: click.echo(f"  {msg}"),
        )

        if files:
            stats = indexer.index_files(list(files), full=full)
        else:
            stats = indexer.index_root(full=full, reembed=reembed)

    if embedder:
        embedder.close()

    click.echo(f"\nIndex complete: {stats}")
    if stats.errors:
        click.echo(f"\nErrors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            click.echo(f"  {err}", err=True)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

@main.command()
@click.option("--path", default=None, help="Project root")
def status(path: str | None):
    """Show index statistics."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init' first.")
        sys.exit(1)

    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        stats = store.stats()

    click.echo(f"Database: {config.db_path}")
    click.echo(f"Files indexed:   {stats['files']}")
    click.echo(f"Symbols:")
    for kind, count in stats["symbols"].items():
        click.echo(f"  {kind:<12} {count:>6}")
    click.echo(f"Embeddings:      {stats['embeddings']}")


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--semantic-only", is_flag=True)
@click.option("--text-only", is_flag=True)
@click.option("--rewrite", "-r", is_flag=True, help="Force LLM query rewriting")
@click.option("--path", default=None, help="Project root")
def search(query: str, limit: int, semantic_only: bool, text_only: bool, rewrite: bool, path: str | None):
    """Search the code index with a natural language or keyword query."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.embeddings import EmbeddingClient
    from codelibrarian.searcher import Searcher
    from codelibrarian.storage.store import SQLiteStore

    embedder = None
    if config.embeddings_enabled and not text_only:
        embedder = EmbeddingClient(
            api_url=config.embedding_api_url,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            max_chars=config.embedding_max_chars,
        )

    rewriter = None
    if config.query_rewrite_enabled:
        from codelibrarian.query_rewriter import QueryRewriter

        rewriter = QueryRewriter(
            api_url=config.query_rewrite_api_url,
            model=config.query_rewrite_model,
            timeout=config.query_rewrite_timeout,
        )

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        searcher = Searcher(store, embedder, rewriter=rewriter)
        results = searcher.search(
            query,
            limit=limit,
            semantic_only=semantic_only,
            text_only=text_only,
            rewrite=rewrite,
        )

    if embedder:
        embedder.close()

    if rewriter:
        rewriter.close()

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"{'Score':>6}  {'Kind':<8}  {'Symbol':<40}  Location")
    click.echo("-" * 80)
    for r in results:
        sym = r.symbol
        location = f"{sym.relative_path}:{sym.line_start}"
        click.echo(
            f"{r.score:6.3f}  {sym.kind:<8}  {sym.qualified_name:<40}  {location}"
        )


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #

@main.command()
@click.argument("name")
@click.option("--path", default=None, help="Project root")
def lookup(name: str, path: str | None):
    """Look up a symbol by name and show its full details."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.searcher import Searcher
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        searcher = Searcher(store)
        results = searcher.lookup_symbol(name)

    if not results:
        click.echo(f"Symbol '{name}' not found.")
        return

    for sym in results:
        click.echo(f"\n{'='*60}")
        click.echo(f"Name:      {sym.name}")
        click.echo(f"Qualified: {sym.qualified_name}")
        click.echo(f"Kind:      {sym.kind}")
        if sym.http_method:
            click.echo(f"Method:    {sym.http_method}")
        if sym.route:
            click.echo(f"Route:     {sym.route}")
        click.echo(f"File:      {sym.relative_path}:{sym.line_start}-{sym.line_end}")
        if sym.signature:
            click.echo(f"Signature: {sym.signature}")
        if sym.return_type:
            click.echo(f"Returns:   {sym.return_type}")
        if sym.parameters:
            click.echo("Parameters:")
            for p in sym.parameters:
                line = f"  {p.name}"
                if p.type:
                    line += f": {p.type}"
                if p.default is not None:
                    line += f" = {p.default}"
                click.echo(line)
        if sym.decorators:
            click.echo(f"Decorators: {', '.join(sym.decorators)}")
        if sym.docstring:
            click.echo(f"\nDocstring:\n  {sym.docstring[:500]}")


# --------------------------------------------------------------------------- #
# callers / callees
# --------------------------------------------------------------------------- #

@main.command()
@click.argument("name")
@click.option("--depth", "-d", default=1, help="Call-graph hops to traverse")
@click.option("--path", default=None, help="Project root")
def callers(name: str, depth: int, path: str | None):
    """Find all functions/methods that call the named symbol."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.searcher import Searcher
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        searcher = Searcher(store)
        results = searcher.get_callers(name, depth=depth)

    if not results:
        click.echo(f"No callers found for '{name}'.")
        return

    click.echo(f"{'Kind':<10}  {'Symbol':<45}  Location")
    click.echo("-" * 80)
    for sym in results:
        location = f"{sym.relative_path}:{sym.line_start}"
        click.echo(f"{sym.kind:<10}  {sym.qualified_name:<45}  {location}")


@main.command()
@click.argument("name")
@click.option("--depth", "-d", default=1, help="Call-graph hops to traverse")
@click.option("--path", default=None, help="Project root")
def callees(name: str, depth: int, path: str | None):
    """Find all functions/methods called by the named symbol."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.searcher import Searcher
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        searcher = Searcher(store)
        results = searcher.get_callees(name, depth=depth)

    if not results:
        click.echo(f"No callees found for '{name}'.")
        return

    click.echo(f"{'Kind':<10}  {'Symbol':<45}  Location")
    click.echo("-" * 80)
    for sym in results:
        location = f"{sym.relative_path}:{sym.line_start}"
        click.echo(f"{sym.kind:<10}  {sym.qualified_name:<45}  {location}")


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #

@main.group()
def hooks():
    """Manage git hooks."""


@hooks.command("install")
@click.option("--path", default=".", help="Project root (must contain .git/)")
def hooks_install(path: str):
    """Install post-commit and post-merge hooks into .git/hooks/."""
    root = Path(path).resolve()
    git_hooks_dir = root / ".git" / "hooks"

    if not git_hooks_dir.exists():
        click.echo(f"No .git/hooks/ found at {root}. Are you in a git repo?")
        sys.exit(1)

    # Find our hook sources
    hooks_src = Path(__file__).parent.parent.parent.parent / "hooks"
    if not hooks_src.exists():
        # Try relative to package install
        import importlib.util
        spec = importlib.util.find_spec("codelibrarian")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent
            hooks_src = pkg_dir.parent.parent.parent / "hooks"

    for hook_name in ("post-commit", "post-merge"):
        src = hooks_src / hook_name
        dst = git_hooks_dir / hook_name

        if not src.exists():
            # Write hook content directly
            _write_hook(dst, hook_name)
        else:
            import shutil
            shutil.copy2(str(src), str(dst))

        dst.chmod(0o755)
        click.echo(f"Installed {dst}")

    click.echo("Done. Hooks will trigger incremental reindexing on commit/merge.")


def _write_hook(path: Path, hook_name: str) -> None:
    content = f"""#!/bin/sh
# codelibrarian git hook: {hook_name}
# Incrementally reindex changed files after each commit/merge

CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD 2>/dev/null)
if [ -n "$CHANGED" ]; then
    codelibrarian index --files $CHANGED 2>/dev/null &
fi
"""
    path.write_text(content)


# --------------------------------------------------------------------------- #
# diagram
# --------------------------------------------------------------------------- #

def _emit_output(text: str, output_file: str | None) -> None:
    """Write text to file or stdout."""
    if output_file:
        Path(output_file).write_text(text, encoding="utf-8")
        click.echo(f"Written to {output_file}")
    else:
        click.echo(text)


@main.group()
def diagram():
    """Generate Mermaid diagrams from the code index."""


@diagram.command("class")
@click.argument("name")
@click.option("--html", "as_html", is_flag=True, help="Output self-contained HTML instead of Mermaid text")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None, help="Write output to file")
@click.option("--path", default=None, help="Project root")
def diagram_class(name: str, as_html: bool, output_file: str | None, path: str | None):
    """Generate a class hierarchy diagram in Mermaid syntax."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.diagrams import mermaid_class_diagram
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        result = mermaid_class_diagram(store, name)

    if not result:
        click.echo(f"Class '{name}' not found in index.")
        sys.exit(1)

    if as_html:
        from codelibrarian.html_renderer import render_html
        result = render_html(result, title=f"Class Diagram: {name}")

    _emit_output(result, output_file)


@diagram.command("calls")
@click.argument("name")
@click.option("--depth", "-d", default=2, help="Call-graph hops to traverse")
@click.option(
    "--direction",
    type=click.Choice(["callees", "callers"]),
    default="callees",
    help="Traverse callees (forward) or callers (backward)",
)
@click.option("--html", "as_html", is_flag=True, help="Output self-contained HTML instead of Mermaid text")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None, help="Write output to file")
@click.option("--path", default=None, help="Project root")
def diagram_calls(name: str, depth: int, direction: str, as_html: bool, output_file: str | None, path: str | None):
    """Generate a call graph diagram in Mermaid syntax."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.diagrams import mermaid_call_graph
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        result = mermaid_call_graph(store, name, depth=depth, direction=direction)

    if not result:
        click.echo(f"Symbol '{name}' not found or has no call edges.")
        sys.exit(1)

    if as_html:
        from codelibrarian.html_renderer import render_html
        result = render_html(result, title=f"Call Graph: {name}")

    _emit_output(result, output_file)


@diagram.command("imports")
@click.option("--file", "file_path", default=None, help="Scope to a single file")
@click.option("--html", "as_html", is_flag=True, help="Output self-contained HTML instead of Mermaid text")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None, help="Write output to file")
@click.option("--path", default=None, help="Project root")
def diagram_imports(file_path: str | None, as_html: bool, output_file: str | None, path: str | None):
    """Generate a module import graph diagram in Mermaid syntax."""
    root = Path(path).resolve() if path else None
    config = Config.load(root) if root else Config.load_from_cwd()

    if not config.db_path.exists():
        click.echo("No index found. Run 'codelibrarian init && codelibrarian index' first.")
        sys.exit(1)

    from codelibrarian.diagrams import mermaid_import_graph
    from codelibrarian.storage.store import SQLiteStore

    with SQLiteStore(config.db_path, config.embedding_dimensions) as store:
        result = mermaid_import_graph(store, file_path=file_path)

    if not result:
        click.echo("No import edges found in index.")
        sys.exit(1)

    if as_html:
        from codelibrarian.html_renderer import render_html
        title = f"Import Graph: {file_path}" if file_path else "Import Graph"
        result = render_html(result, title=title)

    _emit_output(result, output_file)


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #

@main.command()
@click.option("--path", default=None, help="Project root")
def serve(path: str | None):
    """Start the MCP server on stdio."""
    root = Path(path).resolve() if path else None

    from codelibrarian.mcp_server import run_server

    asyncio.run(run_server(root))
