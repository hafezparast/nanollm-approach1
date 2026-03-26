"""Abstract base adapter — interface that every provider adapter must implement.

Adapters are responsible for translating between NanoLLM's unified API and
the provider-specific HTTP request/response formats.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._config import ProviderConfig
    from .._types import EmbeddingResponse, ModelResponse


class BaseAdapter(ABC):
    """Abstract base class for all provider adapters.

    Subclasses must implement the chat completion methods. Embedding methods
    have default implementations that raise NotImplementedError, since not
    every provider supports embeddings.
    """

    # ── Chat completions ─────────────────────────────────────────────

    @abstractmethod
    def build_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str,
        base_url: str,
        stream: bool,
        provider_config: "ProviderConfig",
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build an HTTP request for a chat completion.

        Args:
            model: The model id (without provider prefix).
            messages: Chat messages in OpenAI format.
            api_key: Resolved API key.
            base_url: Resolved base URL (no trailing slash).
            stream: Whether to request streaming.
            provider_config: Full provider configuration.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Tuple of (url, headers, body).
        """

    @abstractmethod
    def parse_response(
        self,
        data: dict[str, Any],
        model: str,
    ) -> "ModelResponse":
        """Parse a non-streaming JSON response into a ModelResponse.

        Args:
            data: Raw JSON response body.
            model: The model id (for tagging the response).

        Returns:
            A populated ModelResponse.
        """

    @abstractmethod
    def parse_stream_chunk(
        self,
        line: str,
        model: str,
    ) -> dict[str, Any] | None:
        """Parse a single SSE line from a streaming response.

        Args:
            line: A single line from the SSE stream (after "data: " prefix removal).
            model: The model id.

        Returns:
            A streaming chunk dict, or None if the line should be skipped
            (e.g. empty keep-alive, "[DONE]" sentinel).
        """

    # ── Embeddings ───────────────────────────────────────────────────

    def build_embedding_request(
        self,
        model: str,
        input: str | list[str],
        api_key: str,
        base_url: str,
        provider_config: "ProviderConfig",
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build an HTTP request for an embedding call.

        Default implementation raises NotImplementedError. Override in
        adapters that support embeddings.

        Args:
            model: The model id.
            input: Text or list of texts to embed.
            api_key: Resolved API key.
            base_url: Resolved base URL.
            provider_config: Full provider configuration.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (url, headers, body).
        """
        raise NotImplementedError(
            f"Embeddings not supported by {type(self).__name__}"
        )

    def parse_embedding_response(
        self,
        data: dict[str, Any],
        model: str,
    ) -> "EmbeddingResponse":
        """Parse a JSON response into an EmbeddingResponse.

        Default implementation raises NotImplementedError.

        Args:
            data: Raw JSON response body.
            model: The model id.

        Returns:
            A populated EmbeddingResponse.
        """
        raise NotImplementedError(
            f"Embeddings not supported by {type(self).__name__}"
        )

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def filter_params(
        kwargs: dict[str, Any],
        supported: frozenset[str],
    ) -> dict[str, Any]:
        """Filter kwargs to only include parameters the provider supports.

        Unsupported parameters are silently dropped with a debug warning,
        preventing 400 errors from providers that reject unknown fields.

        Args:
            kwargs: All user-provided parameters.
            supported: Set of parameter names the provider accepts.

        Returns:
            New dict containing only supported parameters.
        """
        filtered = {}
        for key, value in kwargs.items():
            if key in supported:
                filtered[key] = value
            else:
                warnings.warn(
                    f"Dropping unsupported parameter {key!r}",
                    stacklevel=3,
                )
        return filtered
