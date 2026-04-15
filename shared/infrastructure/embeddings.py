"""Shared Google GenAI embedding client."""

from __future__ import annotations

import structlog
from google import genai
from google.genai import types as genai_types

logger = structlog.get_logger(__name__)


class GoogleGenAIEmbeddingClient:
    """Thin async wrapper around the Google GenAI embedding API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        output_dimensions: int,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty Google API key is required.")
        self._api_key = api_key
        self._model = model
        self._output_dimensions = output_dimensions
        self._client = client

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options={"api_version": "v1beta"},
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in a single batch request."""
        if not texts:
            return []

        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed blank text in batch.")

        client = self._get_client()

        try:
            result = await client.aio.models.embed_content(
                model=self._model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._output_dimensions,
                ),
            )

            embeddings = getattr(result, "embeddings", None)
            if not embeddings:
                raise ValueError("Google GenAI returned no embeddings.")

            if len(embeddings) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, but got {len(embeddings)}")

            return [[float(value) for value in emb.values] for emb in embeddings]
        except Exception as e:
            logger.error(
                "batch_embedding_failed", error=str(e), model=self._model, count=len(texts)
            )
            raise
