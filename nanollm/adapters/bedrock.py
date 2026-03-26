"""AWS Bedrock Converse API adapter for NanoLLM.

Implements AWS SigV4 request signing inline (no boto3 required),
with optional boto3/botocore credential chain support when available.
Supports multimodal/vision messages via to_bedrock_image().
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from typing import Any
from urllib.parse import quote, urlparse

from .._image import extract_image_url, to_bedrock_image


# ── AWS SigV4 signing ────────────────────────────────────────────────


def _get_credentials() -> tuple[str, str, str | None]:
    """Get AWS credentials from boto3 (if available) or environment variables.

    Returns:
        (access_key, secret_key, session_token)
    """
    # Try boto3 credential chain first (handles IAM roles, profiles, etc.)
    try:
        import botocore.session

        session = botocore.session.get_session()
        creds = session.get_credentials()
        if creds:
            resolved = creds.get_frozen_credentials()
            return resolved.access_key, resolved.secret_key, resolved.token
    except Exception:
        pass

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN")
    return access_key, secret_key, session_token


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    secret: str, date_stamp: str, region: str, service: str
) -> bytes:
    k_date = _sign(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _sigv4_headers(
    method: str,
    url: str,
    headers: dict[str, str],
    body: str,
    region: str,
    service: str = "bedrock",
) -> dict[str, str]:
    """Compute AWS SigV4 Authorization header and return updated headers."""
    access_key, secret_key, session_token = _get_credentials()
    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or configure boto3 credentials."
        )

    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = quote(parsed.path or "/", safe="/")

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    # Build headers to sign
    signing_headers = {
        "host": host,
        "x-amz-date": amz_date,
        "content-type": headers.get("Content-Type", "application/json"),
    }
    if session_token:
        signing_headers["x-amz-security-token"] = session_token

    # Canonical headers and signed headers (must be sorted)
    sorted_keys = sorted(signing_headers.keys())
    canonical_headers = "".join(
        f"{k}:{signing_headers[k]}\n" for k in sorted_keys
    )
    signed_headers = ";".join(sorted_keys)

    # Payload hash
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    # Canonical request
    canonical_request = "\n".join([
        method,
        path,
        "",  # no query string
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    # String to sign
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    # Signature
    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Authorization header
    authorization = (
        f"AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    result = {
        **headers,
        "X-Amz-Date": amz_date,
        "Authorization": authorization,
    }
    if session_token:
        result["X-Amz-Security-Token"] = session_token

    return result


# ── Message conversion ───────────────────────────────────────────────


def _convert_content_block(block: dict) -> dict:
    """Convert a single OpenAI content block to Bedrock format."""
    if block.get("type") == "text":
        return {"text": block["text"]}

    url = extract_image_url(block)
    if url:
        return to_bedrock_image(url)

    # Fallback: treat as text
    return {"text": str(block.get("text", block.get("content", "")))}


def _convert_messages(
    messages: list[dict],
) -> tuple[list[dict] | None, list[dict]]:
    """Convert OpenAI messages to Bedrock Converse format.

    Returns:
        (system_blocks, conversation_messages)
        system_blocks: list of {"text": ...} for system messages, or None
        conversation_messages: list of {"role": ..., "content": [...]} dicts
    """
    system_blocks: list[dict] = []
    conversation: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            text = msg.get("content", "")
            if isinstance(text, str):
                system_blocks.append({"text": text})
            elif isinstance(text, list):
                for block in text:
                    if isinstance(block, str):
                        system_blocks.append({"text": block})
                    elif isinstance(block, dict):
                        system_blocks.append({"text": block.get("text", "")})
            continue

        # Map role
        bedrock_role = "assistant" if role == "assistant" else "user"

        # Convert content
        content = msg.get("content", "")
        if isinstance(content, str):
            bedrock_content = [{"text": content}]
        elif isinstance(content, list):
            bedrock_content = [_convert_content_block(b) for b in content]
        else:
            bedrock_content = [{"text": str(content)}]

        conversation.append({"role": bedrock_role, "content": bedrock_content})

    return system_blocks or None, conversation


# ── Public adapter functions ─────────────────────────────────────────


def build_request(
    *,
    model: str,
    messages: list[dict],
    stream: bool = False,
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an HTTP request dict for Bedrock Converse API.

    No api_key/base_url needed -- uses AWS SigV4 signing with
    credentials from environment or boto3 credential chain.
    """
    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    action = "converse-stream" if stream else "converse"
    url = (
        f"https://bedrock-runtime.{region}.amazonaws.com"
        f"/model/{model}/{action}"
    )

    system_blocks, conversation = _convert_messages(messages)

    body: dict[str, Any] = {
        "messages": conversation,
    }
    if system_blocks:
        body["system"] = system_blocks

    # Map inference parameters
    inference_config: dict[str, Any] = {}
    if "max_tokens" in kwargs:
        inference_config["maxTokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        inference_config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        inference_config["topP"] = kwargs["top_p"]
    if "stop" in kwargs:
        inference_config["stopSequences"] = kwargs["stop"]

    if inference_config:
        body["inferenceConfig"] = inference_config

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    body_str = json.dumps(body)
    signed_headers = _sigv4_headers("POST", url, headers, body_str, region)

    return {"url": url, "headers": signed_headers, "body": body}


# ── Response parsing ─────────────────────────────────────────────────


_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "content_filtered": "content_filter",
}


def parse_response(raw: dict) -> dict:
    """Parse a Bedrock Converse response into normalized format.

    Returns:
        {
            "content": str | None,
            "finish_reason": str | None,
            "tool_calls": list[dict] | None,
            "usage": dict,
            "model": str,
        }
    """
    output = raw.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    # Extract text from content blocks
    text_parts = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])

    content = "".join(text_parts) if text_parts else None

    # Map stop reason
    stop_reason = raw.get("stopReason", "")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason or None)

    # Extract usage
    usage_raw = raw.get("usage", {})
    usage = {
        "prompt_tokens": usage_raw.get("inputTokens", 0),
        "completion_tokens": usage_raw.get("outputTokens", 0),
        "total_tokens": usage_raw.get("totalTokens", 0),
    }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "tool_calls": None,
        "usage": usage,
        "model": raw.get("model", ""),
    }


def parse_stream_chunk(line: str) -> dict | None:
    """Parse a Bedrock Converse streaming event.

    Bedrock streams JSON events with types like:
    - contentBlockDelta: contains delta.text
    - messageStop: contains stopReason
    - metadata: contains usage

    Args:
        line: A JSON line from the Bedrock event stream.

    Returns:
        OpenAI-format chunk dict, or None for irrelevant events.
    """
    line = line.strip()
    if not line:
        return None

    try:
        event = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None

    # contentBlockDelta -> text content
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"].get("delta", {})
        text = delta.get("text")
        if text is not None:
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }

    # messageStop -> finish reason
    if "messageStop" in event:
        stop_reason = event["messageStop"].get("stopReason", "")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason or "stop")
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    # metadata -> usage
    if "metadata" in event:
        usage_raw = event["metadata"].get("usage", {})
        if usage_raw:
            return {
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": usage_raw.get("inputTokens", 0),
                    "completion_tokens": usage_raw.get("outputTokens", 0),
                    "total_tokens": usage_raw.get("totalTokens", 0),
                },
            }

    return None
