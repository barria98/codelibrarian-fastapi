"""OpenAI-compatible embedding client."""

from __future__ import annotations

import logging
import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        api_url: str,
        model: str,
        dimensions: int,
        batch_size: int = 32,
        max_chars: int = 1600,
        timeout: float = 30.0,
    ):
        self.api_url = api_url.rstrip("/")
        # Ensure we hit the /embeddings endpoint
        if not self.api_url.endswith("/embeddings"):
            self.api_url = f"{self.api_url}/embeddings"
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._client = httpx.Client(timeout=timeout)

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None on failure."""
        truncated = [t[: self.max_chars] for t in texts]
        try:
            resp = self._client.post(
                self.api_url,
                json={"model": self.model, "input": truncated},
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI format: {"data": [{"embedding": [...], "index": N}]}
            items = sorted(data["data"], key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in items]
            # Guard against a server returning a different number of vectors
            # than inputs (or non-contiguous indices). Treat any mismatch as a
            # failure rather than silently misaligning vectors with symbols.
            if len(embeddings) != len(truncated):
                logger.debug(
                    "Embedding count mismatch: sent %d, got %d",
                    len(truncated),
                    len(embeddings),
                )
                return None
            return embeddings
        except Exception as exc:
            logger.debug("Embedding request failed: %s", exc)
            return None

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed all texts in batches. Returns None for each failed item."""
        results: list[list[float] | None] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self.embed_batch(batch)
            if batch_results is None:
                results.extend([None] * len(batch))
            else:
                results.extend(batch_results)
        return results

    def embed_one(self, text: str) -> list[float] | None:
        results = self.embed_batch([text])
        if results and len(results) > 0:
            return results[0]
        return None

    def check_connection(self) -> tuple[bool, str]:
        """Verify the embedding API is reachable and returns expected dimensions."""
        result = self.embed_one("test")
        if result is None:
            return False, f"Could not reach embedding API at {self.api_url}"
        if len(result) != self.dimensions:
            return False, (
                f"Dimension mismatch: got {len(result)}, "
                f"expected {self.dimensions}. "
                f"Update config.embeddings.dimensions or use --reembed."
            )
        return True, f"OK (model={self.model}, dimensions={len(result)})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
