"""LLM-powered query rewriter using an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import logging
import re

import httpx

from codelibrarian.models import RewrittenQuery

logger = logging.getLogger(__name__)

_BASE_SYSTEM_PROMPT = """\
You are a code search assistant. Given a natural language question about a codebase, \
return JSON with search terms a developer would use to find the relevant code.

{vocabulary_section}\
Return ONLY valid JSON:
{{"terms": ["term1", "term2", ...], "focus": "implementation"|"tests"|"all"}}

Rules:
- terms: 3-6 search terms, preferring actual symbol names from the codebase
- focus: "implementation" if asking about how code works, "tests" if asking about testing, "all" if unclear
- No explanations, just JSON"""


def _build_system_prompt(vocabulary: list[str] | None = None) -> str:
    """Build the system prompt, optionally with codebase vocabulary."""
    if vocabulary:
        vocab_text = ", ".join(vocabulary)
        section = f"Available symbols in the codebase:\n{vocab_text}\n\n"
    else:
        section = ""
    return _BASE_SYSTEM_PROMPT.format(vocabulary_section=section)


class QueryRewriter:
    def __init__(
        self,
        api_url: str,
        model: str,
        timeout: float = 5.0,
    ):
        self.api_url = api_url.rstrip("/")
        # Accept either a full chat-completions URL or a base URL, mirroring
        # EmbeddingClient's handling of the /embeddings suffix.
        if not self.api_url.endswith("/chat/completions"):
            self.api_url = f"{self.api_url}/chat/completions"
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def rewrite(
        self, query: str, vocabulary: list[str] | None = None
    ) -> RewrittenQuery | None:
        """Rewrite a natural language query into code search terms.

        If *vocabulary* is provided, the LLM prompt includes the codebase's
        symbol names so it can pick actual identifiers instead of generic words.

        Returns None on any failure (timeout, connection error, bad JSON).
        """
        system_prompt = _build_system_prompt(vocabulary)
        try:
            resp = self._client.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_response(content)
        except Exception as exc:
            logger.debug("Query rewrite failed: %s", exc)
            return None

    def _parse_response(self, content: str) -> RewrittenQuery | None:
        """Parse the LLM response into a RewrittenQuery."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Query rewrite returned invalid JSON: %s", content)
            return None

        terms = parsed.get("terms")
        if not terms or not isinstance(terms, list):
            return None

        focus = parsed.get("focus", "all")
        if focus not in ("implementation", "tests", "all"):
            focus = "all"

        return RewrittenQuery(terms=terms, focus=focus)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QueryRewriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
