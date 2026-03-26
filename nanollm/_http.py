"""HTTP transport layer for NanoLLM.

Thin wrapper around httpx for sync/async POST and SSE streaming.
Maps HTTP error status codes to NanoLLM exceptions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .exceptions import APIConnectionError, Timeout, raise_for_status

_DEFAULT_TIMEOUT = 600.0  # 10 minutes


def _parse_error_body(response: httpx.Response) -> dict | str:
    """Try to parse error response as JSON, fall back to text."""
    try:
        return response.json()
    except Exception:
        return response.text


def _wrap_connection_error(exc: Exception, provider: str | None, model: str | None):
    """Wrap httpx connection/timeout errors into NanoLLM exceptions."""
    if isinstance(exc, httpx.TimeoutException):
        raise Timeout(
            message=f"Request timed out: {exc}",
            llm_provider=provider,
            model=model,
        ) from exc
    if isinstance(exc, httpx.ConnectError):
        raise APIConnectionError(
            message=f"Connection error: {exc}",
            llm_provider=provider,
            model=model,
        ) from exc
    raise APIConnectionError(
        message=f"HTTP error: {exc}",
        llm_provider=provider,
        model=model,
    ) from exc


def sync_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Synchronous JSON POST request."""
    try:
        with httpx.Client(timeout=timeout, http2=True) as client:
            response = client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise_for_status(
                    response.status_code,
                    _parse_error_body(response),
                    provider=provider,
                    model=model,
                )
            return response.json()
    except httpx.HTTPError as exc:
        _wrap_connection_error(exc, provider, model)


async def async_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Asynchronous JSON POST request."""
    try:
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise_for_status(
                    response.status_code,
                    _parse_error_body(response),
                    provider=provider,
                    model=model,
                )
            return response.json()
    except httpx.HTTPError as exc:
        _wrap_connection_error(exc, provider, model)


def sync_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Synchronous SSE streaming POST. Yields data lines."""
    try:
        with httpx.Client(timeout=timeout, http2=True) as client:
            with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    raise_for_status(
                        response.status_code,
                        _parse_error_body(response),
                        provider=provider,
                        model=model,
                    )
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        yield data
    except httpx.HTTPError as exc:
        _wrap_connection_error(exc, provider, model)


async def async_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Asynchronous SSE streaming POST. Yields data lines."""
    try:
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(
                        response.status_code,
                        _parse_error_body(response),
                        provider=provider,
                        model=model,
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        yield data
    except httpx.HTTPError as exc:
        _wrap_connection_error(exc, provider, model)
