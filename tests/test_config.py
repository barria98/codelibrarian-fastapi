"""Tests for Config, focused on path exclusion semantics."""

from pathlib import Path

import pytest

from codelibrarian.config import Config


@pytest.fixture
def config(tmp_path):
    config_dir = tmp_path / ".codelibrarian"
    config_dir.mkdir()
    return Config(
        data={
            "index": {
                "root": ".",
                "exclude": [
                    "node_modules/",
                    ".git/",
                    "build/",
                    "*.min.js",
                    "*.lock",
                ],
                "languages": ["python"],
            },
        },
        config_dir=config_dir,
    )


class TestIsExcluded:
    def test_excludes_directory_component(self, config):
        assert config.is_excluded(Path("/proj/node_modules/x/y.js"))
        assert config.is_excluded(Path("/proj/.git/config"))
        assert config.is_excluded(Path("/proj/build/out.py"))

    def test_does_not_over_exclude_similar_dir_names(self, config):
        """'build/' must not exclude 'rebuild/' or 'prebuild/' (issue #9)."""
        assert not config.is_excluded(Path("/proj/rebuild/out.py"))
        assert not config.is_excluded(Path("/proj/prebuild/out.py"))

    def test_dir_pattern_requires_whole_component(self, config):
        """'.git/' must not match a path that merely contains '.github/'."""
        assert not config.is_excluded(Path("/proj/.github/workflows/ci.yml"))

    def test_glob_matches_basename(self, config):
        assert config.is_excluded(Path("/proj/src/app.min.js"))
        assert config.is_excluded(Path("/proj/poetry.lock"))

    def test_glob_does_not_over_match_trailing(self, config):
        """'*.lock' should not match 'foo.lock.bak' (anchored to basename)."""
        assert not config.is_excluded(Path("/proj/foo.lock.bak"))

    def test_normal_file_not_excluded(self, config):
        assert not config.is_excluded(Path("/proj/src/main.py"))
