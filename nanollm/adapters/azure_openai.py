"""Azure OpenAI adapter for NanoLLM.

Extends the OpenAI-compatible adapter with Azure-specific URL patterns
and authentication (api-key header instead of Bearer token).

Azure URLs include the deployment in the base_url:
    https://{resource}.openai.azure.com/openai/deployments/{deployment}

The adapter appends /chat/completions?api-version=... to that.
"""

from __future__ import annotations

from typing import Any

from . import openai_compat

# Re-export parse functions directly — identical to OpenAI
parse_response = openai_compat.parse_response
parse_stream_chunk = openai_compat.parse_stream_chunk
parse_embedding_response = openai_compat.parse_embedding_response

_DEFAULT_API_VERSION = "2025-02-01-preview"


def build_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    extra_headers: dict[str, str] | None = None,
    api_version: str = "",
    **kwargs: Any,
) -> dict:
    """Build an HTTP request for Azure OpenAI chat completions.

    URL: {base_url}/chat/completions?api-version={api_version}
    Auth: api-key header (not Authorization/Bearer).
    """
    version = api_version or _DEFAULT_API_VERSION
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions?api-version={version}"

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    if stream:
        body["stream_options"] = {"include_usage": True}

    for key in openai_compat._CHAT_PARAMS:
        if key in kwargs:
            body[key] = kwargs[key]

    return {"url": url, "headers": headers, "body": body}


def build_embedding_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    input: str | list[str],
    extra_headers: dict[str, str] | None = None,
    api_version: str = "",
    **kwargs: Any,
) -> dict:
    """Build an embedding request for Azure OpenAI.

    URL: {base_url}/embeddings?api-version={api_version}
    """
    version = api_version or _DEFAULT_API_VERSION
    base = base_url.rstrip("/")
    url = f"{base}/embeddings?api-version={version}"

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": model,
        "input": input,
    }
    for key in ("encoding_format", "dimensions", "user"):
        if key in kwargs:
            body[key] = kwargs[key]

    return {"url": url, "headers": headers, "body": body}
