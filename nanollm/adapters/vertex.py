"""Google Vertex AI adapter for NanoLLM.

Extends the GeminiAdapter base class with Vertex AI URL construction
and authentication (google-auth library or VERTEX_API_KEY env var).
Supports multimodal/vision messages via to_gemini_image().
"""

from __future__ import annotations

import json
import os
from typing import Any

from .._image import extract_image_url, to_gemini_image


# ── Gemini message conversion (shared with Vertex) ──────────────────


def _convert_content_block(block: dict) -> dict:
    """Convert a single OpenAI content block to Gemini format."""
    if block.get("type") == "text":
        return {"text": block["text"]}

    url = extract_image_url(block)
    if url:
        return to_gemini_image(url)

    # Fallback: treat as text
    return {"text": str(block.get("text", block.get("content", "")))}


def _convert_content(content: str | list) -> list[dict]:
    """Convert OpenAI message content to Gemini parts list."""
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        return [_convert_content_block(b) for b in content]
    return [{"text": str(content)}]


# ── GeminiAdapter base class ─────────────────────────────────────────


_ROLE_MAP = {"assistant": "model", "user": "user"}

_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}


class GeminiAdapter:
    """Adapter for the Google Gemini (generativelanguage) API.

    Converts OpenAI messages to Gemini format and builds requests
    for the Gemini REST API. Subclassed by VertexAdapter for
    Vertex AI-specific URL and auth.
    """

    def build_request(
        self,
        *,
        base_url: str | None = None,
        api_key: str = "",
        model: str,
        messages: list[dict],
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Build an HTTP request dict for Gemini generateContent."""
        base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{base}/models/{model}:{action}"

        if api_key:
            url += f"?key={api_key}"

        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        system_instruction, contents = self._convert_messages(messages)

        body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction

        # Map generation parameters
        generation_config: dict[str, Any] = {}
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs:
            generation_config["stopSequences"] = kwargs["stop"]

        if generation_config:
            body["generationConfig"] = generation_config

        return {"url": url, "headers": headers, "body": body}

    def _convert_messages(
        self, messages: list[dict]
    ) -> tuple[dict | None, list[dict]]:
        """Convert OpenAI messages to Gemini format.

        Returns:
            (system_instruction, contents)
        """
        system_parts: list[dict] = []
        contents: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append({"text": content})
                elif isinstance(content, list):
                    system_parts.extend(_convert_content(content))
                continue

            gemini_role = _ROLE_MAP.get(role, "user")
            parts = _convert_content(content)
            contents.append({"role": gemini_role, "parts": parts})

        system_instruction = {"parts": system_parts} if system_parts else None
        return system_instruction, contents

    @staticmethod
    def parse_response(raw: dict) -> dict:
        """Parse a Gemini generateContent response into normalized format."""
        candidates = raw.get("candidates", [])
        if not candidates:
            return {
                "content": None,
                "finish_reason": None,
                "tool_calls": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "",
            }

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_parts = [p["text"] for p in parts if "text" in p]
        text = "".join(text_parts) if text_parts else None

        finish_reason_raw = candidate.get("finishReason", "")
        finish_reason = _FINISH_REASON_MAP.get(finish_reason_raw, finish_reason_raw or None)

        usage_raw = raw.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_raw.get("promptTokenCount", 0),
            "completion_tokens": usage_raw.get("candidatesTokenCount", 0),
            "total_tokens": usage_raw.get("totalTokenCount", 0),
        }

        return {
            "content": text,
            "finish_reason": finish_reason,
            "tool_calls": None,
            "usage": usage,
            "model": raw.get("modelVersion", ""),
        }

    @staticmethod
    def parse_stream_chunk(line: str) -> dict | None:
        """Parse a Gemini streaming chunk.

        Gemini streams JSON array elements. Each chunk has candidates
        with content parts.
        """
        line = line.strip()
        if not line:
            return None

        # Gemini streams JSON lines, sometimes prefixed with array comma
        line = line.lstrip("[,").rstrip("]")
        if not line:
            return None

        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None

        candidates = event.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]
            text = "".join(text_parts) if text_parts else None

            finish_reason_raw = candidate.get("finishReason", "")
            finish_reason = _FINISH_REASON_MAP.get(finish_reason_raw)

            chunk: dict[str, Any] = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text} if text else {},
                        "finish_reason": finish_reason,
                    }
                ],
            }

            # Include usage if present
            usage_raw = event.get("usageMetadata", {})
            if usage_raw:
                chunk["usage"] = {
                    "prompt_tokens": usage_raw.get("promptTokenCount", 0),
                    "completion_tokens": usage_raw.get("candidatesTokenCount", 0),
                    "total_tokens": usage_raw.get("totalTokenCount", 0),
                }

            return chunk

        return None


# ── VertexAdapter ────────────────────────────────────────────────────


def _get_vertex_token() -> str | None:
    """Get a Vertex AI access token via google-auth library or env var."""
    # Try google-auth library first
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except Exception:
        pass

    # Fall back to env var
    return os.environ.get("VERTEX_API_KEY")


class VertexAdapter(GeminiAdapter):
    """Adapter for Google Vertex AI (extends GeminiAdapter).

    Overrides URL construction and authentication. Uses the same
    Gemini message format (inherited).
    """

    def build_request(
        self,
        *,
        base_url: str | None = None,
        api_key: str = "",
        model: str,
        messages: list[dict],
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Build an HTTP request dict for Vertex AI generateContent."""
        location = os.environ.get("VERTEX_LOCATION", "us-central1")
        project = os.environ.get(
            "VERTEX_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        )
        if not project:
            raise ValueError(
                "Vertex AI project not found. Set VERTEX_PROJECT or "
                "GOOGLE_CLOUD_PROJECT environment variable."
            )

        action = "streamGenerateContent" if stream else "generateContent"
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1"
            f"/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:{action}"
        )

        # Auth: Bearer token
        token = api_key or _get_vertex_token()
        if not token:
            raise ValueError(
                "Vertex AI credentials not found. Install google-auth "
                "library, set VERTEX_API_KEY, or provide api_key."
            )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if extra_headers:
            headers.update(extra_headers)

        # Use inherited message conversion
        system_instruction, contents = self._convert_messages(messages)

        body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction

        # Map generation parameters
        generation_config: dict[str, Any] = {}
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs:
            generation_config["stopSequences"] = kwargs["stop"]

        if generation_config:
            body["generationConfig"] = generation_config

        return {"url": url, "headers": headers, "body": body}


# ── Module-level convenience functions (using VertexAdapter) ─────────

_adapter = VertexAdapter()


def build_request(
    *,
    model: str,
    messages: list[dict],
    stream: bool = False,
    api_key: str = "",
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an HTTP request for Vertex AI (module-level convenience)."""
    return _adapter.build_request(
        model=model,
        messages=messages,
        stream=stream,
        api_key=api_key,
        extra_headers=extra_headers,
        **kwargs,
    )


def parse_response(raw: dict) -> dict:
    """Parse a Vertex AI response (same format as Gemini)."""
    return _adapter.parse_response(raw)


def parse_stream_chunk(line: str) -> dict | None:
    """Parse a Vertex AI streaming chunk (same format as Gemini)."""
    return _adapter.parse_stream_chunk(line)
