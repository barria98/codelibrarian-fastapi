"""Tests for QueryRewriter -- all LLM calls are mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from codelibrarian.models import RewrittenQuery
from codelibrarian.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter():
    return QueryRewriter(
        api_url="http://localhost:11434/v1/chat/completions",
        model="qwen2.5:3b",
        timeout=5.0,
    )


class TestRewrite:
    def test_parses_valid_json_response(self, rewriter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "terms": ["insert_call", "INSERT INTO", "store_parse_result"],
                                "focus": "implementation",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response):
            result = rewriter.rewrite("how are edges inserted into the graph?")

        assert result is not None
        assert result.terms == ["insert_call", "INSERT INTO", "store_parse_result"]
        assert result.focus == "implementation"

    def test_returns_none_on_timeout(self, rewriter):
        import httpx

        with patch.object(
            rewriter._client, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            result = rewriter.rewrite("some query")

        assert result is None

    def test_returns_none_on_connection_error(self, rewriter):
        import httpx

        with patch.object(
            rewriter._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            result = rewriter.rewrite("some query")

        assert result is None

    def test_returns_none_on_invalid_json(self, rewriter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response):
            result = rewriter.rewrite("some query")

        assert result is None

    def test_returns_none_on_missing_terms(self, rewriter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"focus": "implementation"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response):
            result = rewriter.rewrite("some query")

        assert result is None

    def test_defaults_focus_to_all(self, rewriter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"terms": ["find_oldest", "animal"]}
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response):
            result = rewriter.rewrite("find oldest animal")

        assert result is not None
        assert result.focus == "all"

    def test_strips_markdown_fences_from_response(self, rewriter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"terms": ["foo", "bar"], "focus": "all"}\n```'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response):
            result = rewriter.rewrite("some query")

        assert result is not None
        assert result.terms == ["foo", "bar"]


    def test_vocabulary_included_in_prompt(self, rewriter):
        """When vocabulary is provided, it should appear in the system prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"terms": ["insert_call", "GraphEdges"], "focus": "implementation"}
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        vocab = ["insert_call", "insert_import", "GraphEdges", "resolve_graph_edges"]
        with patch.object(rewriter._client, "post", return_value=mock_response) as mock_post:
            result = rewriter.rewrite("how are edges inserted?", vocabulary=vocab)

        assert result is not None
        # Verify the system prompt contained the vocabulary
        call_args = mock_post.call_args
        messages = call_args.kwargs.get("json", call_args[1].get("json", {}))["messages"]
        system_msg = messages[0]["content"]
        assert "insert_call" in system_msg
        assert "GraphEdges" in system_msg

    def test_rewrite_without_vocabulary(self, rewriter):
        """When no vocabulary is provided, prompt should still work."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"terms": ["edges", "insert"], "focus": "all"}
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(rewriter._client, "post", return_value=mock_response) as mock_post:
            result = rewriter.rewrite("how are edges inserted?")

        assert result is not None
        call_args = mock_post.call_args
        messages = call_args.kwargs.get("json", call_args[1].get("json", {}))["messages"]
        system_msg = messages[0]["content"]
        assert "Available symbols" not in system_msg


class TestApiUrl:
    def test_appends_chat_completions_to_base_url(self):
        rw = QueryRewriter(api_url="http://localhost:11434/v1", model="m")
        assert rw.api_url == "http://localhost:11434/v1/chat/completions"

    def test_leaves_full_url_unchanged(self):
        rw = QueryRewriter(
            api_url="http://localhost:11434/v1/chat/completions/", model="m"
        )
        assert rw.api_url == "http://localhost:11434/v1/chat/completions"


class TestContextManager:
    def test_enters_and_exits(self):
        rw = QueryRewriter(
            api_url="http://localhost:11434/v1/chat/completions",
            model="qwen2.5:3b",
        )
        with rw as r:
            assert r is rw
