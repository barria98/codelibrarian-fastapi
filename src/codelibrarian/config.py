"""Configuration loading and defaults."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG = {
    "index": {
        "root": ".",
        "exclude": [
            "node_modules/",
            ".git/",
            "__pycache__/",
            "dist/",
            "build/",
            ".codelibrarian/",
            "*.min.js",
            "*.min.css",
            "*.lock",
        ],
        "languages": ["python", "typescript", "javascript", "rust", "java", "cpp", "swift", "kotlin"],
    },
    "embeddings": {
        "api_url": "http://localhost:11434/v1/embeddings",
        "model": "nomic-embed-text-v2-moe",
        "dimensions": 768,
        "batch_size": 32,
        "max_chars": 1600,  # ~400 tokens; model window is 512
        "enabled": True,
    },
    "database": {
        "path": ".codelibrarian/index.db",
    },
    "query_rewrite": {
        "enabled": True,
        "api_url": "http://localhost:11434/v1/chat/completions",
        "model": "qwen2.5:3b",
        "timeout": 5.0,
    },
}

# File extensions mapped to language names
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
}


class Config:
    def __init__(self, data: dict, config_dir: Path):
        self._data = data
        self.config_dir = config_dir

    @classmethod
    def load(cls, project_root: Path) -> "Config":
        config_dir = project_root / ".codelibrarian"
        config_file = config_dir / "config.toml"

        data = _deep_merge(DEFAULT_CONFIG, {})

        if config_file.exists():
            with open(config_file, "rb") as f:
                user_data = tomllib.load(f)
            data = _deep_merge(DEFAULT_CONFIG, user_data)

        return cls(data, config_dir)

    @classmethod
    def load_from_cwd(cls) -> "Config":
        root = _find_project_root(Path.cwd())
        return cls.load(root)

    # --- index ---
    @property
    def index_root(self) -> Path:
        return self.config_dir.parent / self._data["index"]["root"]

    @property
    def exclude_patterns(self) -> list[str]:
        return self._data["index"]["exclude"]

    @property
    def languages(self) -> list[str]:
        return self._data["index"]["languages"]

    # --- embeddings ---
    @property
    def embeddings_enabled(self) -> bool:
        return self._data["embeddings"]["enabled"]

    @property
    def embedding_api_url(self) -> str:
        return self._data["embeddings"]["api_url"]

    @property
    def embedding_model(self) -> str:
        return self._data["embeddings"]["model"]

    @property
    def embedding_dimensions(self) -> int:
        return self._data["embeddings"]["dimensions"]

    @property
    def embedding_batch_size(self) -> int:
        return self._data["embeddings"]["batch_size"]

    @property
    def embedding_max_chars(self) -> int:
        return self._data["embeddings"]["max_chars"]

    # --- database ---
    @property
    def db_path(self) -> Path:
        raw = self._data["database"]["path"]
        p = Path(raw)
        if not p.is_absolute():
            p = self.config_dir.parent / p
        return p

    # --- query rewrite ---
    @property
    def query_rewrite_enabled(self) -> bool:
        return self._data.get("query_rewrite", {}).get("enabled", True)

    @property
    def query_rewrite_api_url(self) -> str:
        return self._data.get("query_rewrite", {}).get(
            "api_url", "http://localhost:11434/v1/chat/completions"
        )

    @property
    def query_rewrite_model(self) -> str:
        return self._data.get("query_rewrite", {}).get("model", "qwen2.5:3b")

    @property
    def query_rewrite_timeout(self) -> float:
        return self._data.get("query_rewrite", {}).get("timeout", 5.0)

    def is_excluded(self, path: Path) -> bool:
        parts = path.parts
        name = path.name
        posix = path.as_posix()
        for pattern in self.exclude_patterns:
            if pattern.endswith("/"):
                # Directory pattern: match a whole path component, not a
                # substring (so "build/" excludes a "build" dir but not
                # "rebuild/"). fnmatch on each component allows globs.
                dir_pat = pattern.rstrip("/")
                if any(fnmatch.fnmatch(part, dir_pat) for part in parts):
                    return True
            else:
                # Glob pattern: match the basename (e.g. "*.min.js"). If the
                # pattern itself contains a slash, also match the full path.
                if fnmatch.fnmatch(name, pattern):
                    return True
                if "/" in pattern and fnmatch.fnmatch(posix, f"*{pattern}*"):
                    return True
        return False

    def language_for_file(self, path: Path) -> str | None:
        lang = LANGUAGE_EXTENSIONS.get(path.suffix.lower())
        if lang and lang in self.languages:
            return lang
        return None


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _find_project_root(start: Path) -> Path:
    """Walk up to find the directory containing .codelibrarian/ or .git/."""
    current = start.resolve()
    while True:
        if (current / ".codelibrarian").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return start.resolve()
        current = parent


DEFAULT_CONFIG_TOML = """\
[index]
root = "."
exclude = [
    "node_modules/",
    ".git/",
    "__pycache__/",
    "dist/",
    "build/",
    ".codelibrarian/",
    "*.min.js",
]
languages = ["python", "typescript", "javascript", "rust", "java", "cpp", "swift", "kotlin"]

[embeddings]
api_url    = "http://localhost:11434/v1/embeddings"
model      = "nomic-embed-text-v2-moe"
dimensions = 768
batch_size = 32
max_chars  = 1600
enabled    = true

[database]
path = ".codelibrarian/index.db"

[query_rewrite]
enabled = true
api_url = "http://localhost:11434/v1/chat/completions"
model   = "qwen2.5:3b"
timeout = 5.0
"""
