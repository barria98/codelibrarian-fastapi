"""Mermaid diagram generation from the codelibrarian index."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from codelibrarian.storage.store import SQLiteStore


def _sanitize_id(name: str) -> str:
    """Convert a qualified name into a valid Mermaid node ID.

    Uses the full qualified name so that ``foo.bar`` and ``foo_bar``
    produce distinct IDs (``foo_bar`` vs ``foo__bar`` after prefix).
    A short hash suffix is appended to avoid collisions from different
    separator-replacement patterns.

    The suffix uses a stable hash (not Python's salted ``hash()``) so the
    same input always yields the same node ID across runs/processes.
    """
    base = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Append a short stable hash of the original name to disambiguate
    # collisions (e.g. "foo.bar" -> "foo_bar" vs "foo_bar" -> "foo_bar").
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def _short_name(qualified_name: str) -> str:
    """Extract the last component of a qualified name for display."""
    return qualified_name.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------- #
# Class hierarchy diagram
# ---------------------------------------------------------------------- #


def mermaid_class_diagram(store: SQLiteStore, class_name: str) -> str:
    """Generate a Mermaid classDiagram for a class and its hierarchy."""
    hierarchy = store.get_class_hierarchy(class_name)
    if hierarchy["class"] is None:
        return ""

    root_qname = hierarchy["class"]["qualified_name"]
    lines = ["classDiagram"]

    # Collect all class qualified names to render
    all_classes: list[str] = [root_qname]
    for p in hierarchy["parents"]:
        all_classes.append(p["qualified_name"])
    for c in hierarchy["children"]:
        all_classes.append(c["qualified_name"])

    # Build unique Mermaid IDs for each class
    class_ids = {qname: _sanitize_id(qname) for qname in all_classes}

    # Emit class blocks with methods (single targeted query per class)
    for qname in all_classes:
        cid = class_ids[qname]
        short = _short_name(qname)
        methods = store.get_methods_for_class(qname)
        if methods:
            lines.append(f"    class {cid}[\"{short}\"] {{")
            for m in methods:
                params = ", ".join(
                    p.name + (f": {p.type}" if p.type else "")
                    for p in m.parameters
                    if p.name != "self" and p.name != "cls"
                )
                ret = f" {m.return_type}" if m.return_type else ""
                lines.append(f"        +{m.name}({params}){ret}")
            lines.append("    }")
        else:
            lines.append(f"    class {cid}[\"{short}\"]")

    # Inheritance arrows: parent <|-- child
    for p in hierarchy["parents"]:
        lines.append(f"    {class_ids[p['qualified_name']]} <|-- {class_ids[root_qname]}")

    for c in hierarchy["children"]:
        lines.append(f"    {class_ids[root_qname]} <|-- {class_ids[c['qualified_name']]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Call graph diagram
# ---------------------------------------------------------------------- #


def mermaid_call_graph(
    store: SQLiteStore,
    qualified_name: str,
    depth: int = 2,
    direction: str = "callees",
) -> str:
    """Generate a Mermaid flowchart for call relationships."""
    edges = store.get_call_edges(qualified_name, depth=depth, direction=direction)
    if not edges:
        return ""

    lines = ["flowchart LR"]

    # Collect unique nodes
    nodes: set[str] = set()
    for caller, callee in edges:
        nodes.add(caller)
        nodes.add(callee)

    # Define nodes with short labels
    for qname in sorted(nodes):
        nid = _sanitize_id(qname)
        label = _short_name(qname)
        lines.append(f"    {nid}[\"{label}\"]")

    # Add edges
    for caller, callee in edges:
        lines.append(f"    {_sanitize_id(caller)} --> {_sanitize_id(callee)}")

    # Highlight the root node
    root_id = _sanitize_id(qualified_name)
    if root_id in {_sanitize_id(n) for n in nodes}:
        lines.append(f"    style {root_id} fill:#f96,stroke:#333,stroke-width:2px")

    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Module import graph diagram
# ---------------------------------------------------------------------- #


def mermaid_import_graph(
    store: SQLiteStore,
    file_path: str | None = None,
) -> str:
    """Generate a Mermaid flowchart of file-to-file import dependencies."""
    all_edges = store.get_all_import_edges()

    if file_path:
        # Edges are keyed by relative_path; resolve whatever the caller passed
        # (absolute, CWD-relative, or basename) to that stored form first.
        scoped = store.resolve_relative_path(file_path) or file_path
        all_edges = [
            (f, t) for f, t in all_edges
            if f == scoped or t == scoped
        ]

    if not all_edges:
        return ""

    lines = ["flowchart LR"]

    # Group files by top-level directory for subgraphs
    dir_files: dict[str, set[str]] = defaultdict(set)
    all_files: set[str] = set()
    for from_path, to_path in all_edges:
        all_files.add(from_path)
        all_files.add(to_path)

    for fp in all_files:
        parts = fp.split("/")
        group = parts[0] if len(parts) > 1 else "."
        dir_files[group].add(fp)

    # Emit subgraphs
    for group in sorted(dir_files):
        files_in_group = sorted(dir_files[group])
        if group == ".":
            # Top-level files, no subgraph
            for fp in files_in_group:
                nid = _sanitize_id(fp)
                label = _file_label(fp)
                lines.append(f"    {nid}[\"{label}\"]")
        else:
            lines.append(f"    subgraph {_sanitize_id(group)}[\"{group}\"]")
            for fp in files_in_group:
                nid = _sanitize_id(fp)
                label = _file_label(fp)
                lines.append(f"        {nid}[\"{label}\"]")
            lines.append("    end")

    # Add edges
    for from_path, to_path in all_edges:
        lines.append(f"    {_sanitize_id(from_path)} --> {_sanitize_id(to_path)}")

    return "\n".join(lines)


def _file_label(relative_path: str) -> str:
    """Short label for a file node: filename without directory prefix."""
    return relative_path.rsplit("/", 1)[-1]
