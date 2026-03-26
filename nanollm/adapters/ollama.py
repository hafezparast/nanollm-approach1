"""Ollama adapter for NanoLLM.

Thin wrapper around the OpenAI-compatible adapter. Ollama exposes an
OpenAI-compatible API at /v1, so we reuse all logic and just adjust
the default base URL and remove authentication.
"""

from __future__ import annotations

from typing import Any

from . import openai_compat

# Re-export parse functions directly — no changes needed
parse_response = openai_compat.parse_response
parse_stream_chunk = openai_compat.parse_stream_chunk
parse_embedding_response = openai_compat.parse_embedding_response

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def build_request(
    *,
    base_url: str = "",
    api_key: str = "",
    model: str,
    messages: list[dict],
    stream: bool = False,
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an HTTP request for Ollama's OpenAI-compatible endpoint.

    Uses the default localhost URL if none provided. Removes the
    Authorization header since Ollama doesn't require authentication.
    """
    result = openai_compat.build_request(
        base_url=base_url or _DEFAULT_BASE_URL,
        api_key=api_key,
        model=model,
        messages=messages,
        stream=stream,
        extra_headers=extra_headers,
        **kwargs,
    )

    # Remove auth header — Ollama doesn't use it
    result["headers"].pop("Authorization", None)

    return result


def build_embedding_request(
    *,
    base_url: str = "",
    api_key: str = "",
    model: str,
    input: str | list[str],
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an embedding request for Ollama."""
    result = openai_compat.build_embedding_request(
        base_url=base_url or _DEFAULT_BASE_URL,
        api_key=api_key,
        model=model,
        input=input,
        extra_headers=extra_headers,
        **kwargs,
    )

    result["headers"].pop("Authorization", None)

    return result
