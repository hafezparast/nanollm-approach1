"""Exhaustive tests for nanollm/_image.py.

Covers: parse_data_uri, guess_mime_from_url, is_multimodal_message,
has_multimodal_messages, extract_image_url, extract_image_detail,
to_anthropic_image, to_gemini_image, to_bedrock_image,
download_image_as_base64, async_download_image_as_base64.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from nanollm._image import (
    async_download_image_as_base64,
    download_image_as_base64,
    extract_image_detail,
    extract_image_url,
    guess_mime_from_url,
    has_multimodal_messages,
    is_multimodal_message,
    parse_data_uri,
    to_anthropic_image,
    to_bedrock_image,
    to_gemini_image,
)


# ── parse_data_uri ────────────────────────────────────────────────────


class TestParseDataUri:
    def test_valid_jpeg(self):
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        result = parse_data_uri(uri)
        assert result == ("image/jpeg", "/9j/4AAQ")

    def test_valid_png(self):
        uri = "data:image/png;base64,iVBORw0KGgo"
        result = parse_data_uri(uri)
        assert result == ("image/png", "iVBORw0KGgo")

    def test_valid_gif(self):
        uri = "data:image/gif;base64,R0lGODlh"
        result = parse_data_uri(uri)
        assert result == ("image/gif", "R0lGODlh")

    def test_valid_webp(self):
        uri = "data:image/webp;base64,UklGRl"
        result = parse_data_uri(uri)
        assert result == ("image/webp", "UklGRl")

    def test_returns_none_for_plain_url(self):
        assert parse_data_uri("https://example.com/image.jpg") is None

    def test_returns_none_for_empty_string(self):
        assert parse_data_uri("") is None

    def test_returns_none_for_non_image_data_uri(self):
        # _DATA_URI_RE only matches image/* MIME types
        assert parse_data_uri("data:text/plain;base64,SGVsbG8=") is None

    def test_returns_none_for_missing_base64_prefix(self):
        assert parse_data_uri("data:image/jpeg;SGVsbG8=") is None

    def test_long_base64_data(self):
        long_data = "A" * 10000
        uri = f"data:image/png;base64,{long_data}"
        result = parse_data_uri(uri)
        assert result == ("image/png", long_data)

    def test_data_with_special_chars(self):
        # base64 can contain +, /, and =
        data = "abc+def/ghi=="
        uri = f"data:image/jpeg;base64,{data}"
        result = parse_data_uri(uri)
        assert result == ("image/jpeg", data)


# ── guess_mime_from_url ───────────────────────────────────────────────


class TestGuessMimeFromUrl:
    def test_jpg_extension(self):
        assert guess_mime_from_url("https://example.com/photo.jpg") == "image/jpeg"

    def test_jpeg_extension(self):
        assert guess_mime_from_url("https://example.com/photo.jpeg") == "image/jpeg"

    def test_png_extension(self):
        assert guess_mime_from_url("https://example.com/photo.png") == "image/png"

    def test_gif_extension(self):
        assert guess_mime_from_url("https://example.com/photo.gif") == "image/gif"

    def test_webp_extension(self):
        assert guess_mime_from_url("https://example.com/photo.webp") == "image/webp"

    def test_unknown_extension_defaults_to_jpeg(self):
        assert guess_mime_from_url("https://example.com/photo.bmp") == "image/jpeg"

    def test_no_extension_defaults_to_jpeg(self):
        assert guess_mime_from_url("https://example.com/photo") == "image/jpeg"

    def test_url_with_query_params(self):
        url = "https://example.com/photo.png?width=100&height=100"
        assert guess_mime_from_url(url) == "image/png"

    def test_uppercase_extension(self):
        assert guess_mime_from_url("https://example.com/PHOTO.PNG") == "image/png"

    def test_mixed_case_extension(self):
        assert guess_mime_from_url("https://example.com/photo.JpG") == "image/jpeg"


# ── is_multimodal_message ─────────────────────────────────────────────


class TestIsMultimodalMessage:
    def test_plain_text_message(self):
        msg = {"role": "user", "content": "Hello"}
        assert is_multimodal_message(msg) is False

    def test_list_content_with_image_url(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ],
        }
        assert is_multimodal_message(msg) is True

    def test_list_content_without_image_url(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        assert is_multimodal_message(msg) is False

    def test_empty_content_list(self):
        msg = {"role": "user", "content": []}
        assert is_multimodal_message(msg) is False

    def test_none_content(self):
        msg = {"role": "assistant", "content": None}
        assert is_multimodal_message(msg) is False

    def test_missing_content_key(self):
        msg = {"role": "user"}
        assert is_multimodal_message(msg) is False


# ── has_multimodal_messages ───────────────────────────────────────────


class TestHasMultimodalMessages:
    def test_mixed_list(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://img.jpg"}},
                ],
            },
        ]
        assert has_multimodal_messages(msgs) is True

    def test_all_text(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert has_multimodal_messages(msgs) is False

    def test_all_image(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://a.jpg"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://b.jpg"}},
                ],
            },
        ]
        assert has_multimodal_messages(msgs) is True

    def test_empty_list(self):
        assert has_multimodal_messages([]) is False


# ── extract_image_url ─────────────────────────────────────────────────


class TestExtractImageUrl:
    def test_string_url(self):
        block = {"type": "image_url", "image_url": "https://example.com/img.jpg"}
        assert extract_image_url(block) == "https://example.com/img.jpg"

    def test_dict_url(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg"},
        }
        assert extract_image_url(block) == "https://example.com/img.jpg"

    def test_non_image_block(self):
        block = {"type": "text", "text": "Hello"}
        assert extract_image_url(block) is None

    def test_missing_url_key(self):
        block = {"type": "image_url", "image_url": {}}
        assert extract_image_url(block) is None

    def test_missing_image_url_key(self):
        block = {"type": "image_url"}
        assert extract_image_url(block) is None


# ── extract_image_detail ──────────────────────────────────────────────


class TestExtractImageDetail:
    def test_with_detail(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://x.com/img.jpg", "detail": "high"},
        }
        assert extract_image_detail(block) == "high"

    def test_without_detail(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://x.com/img.jpg"},
        }
        assert extract_image_detail(block) is None

    def test_string_image_url(self):
        block = {"type": "image_url", "image_url": "https://x.com/img.jpg"}
        assert extract_image_detail(block) is None

    def test_detail_auto(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://x.com/img.jpg", "detail": "auto"},
        }
        assert extract_image_detail(block) == "auto"

    def test_detail_low(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://x.com/img.jpg", "detail": "low"},
        }
        assert extract_image_detail(block) == "low"


# ── to_anthropic_image ────────────────────────────────────────────────


class TestToAnthropicImage:
    def test_data_uri_to_base64(self):
        uri = "data:image/png;base64,iVBORw0KGgo"
        result = to_anthropic_image(uri)
        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo",
            },
        }

    def test_https_url_to_url_type(self):
        url = "https://example.com/image.jpg"
        result = to_anthropic_image(url)
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": url},
        }

    @patch("nanollm._image.download_image_as_base64")
    def test_http_url_downloads_and_converts(self, mock_download):
        mock_download.return_value = ("image/jpeg", "abc123base64data")
        url = "http://example.com/image.jpg"
        result = to_anthropic_image(url)
        mock_download.assert_called_once_with(url)
        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "abc123base64data",
            },
        }

    def test_data_uri_jpeg(self):
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        result = to_anthropic_image(uri)
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"


# ── to_gemini_image ───────────────────────────────────────────────────


class TestToGeminiImage:
    def test_data_uri_to_inline_data(self):
        uri = "data:image/png;base64,iVBORw0KGgo"
        result = to_gemini_image(uri)
        assert result == {
            "inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo"},
        }

    def test_https_url_to_file_data(self):
        url = "https://example.com/image.png"
        result = to_gemini_image(url)
        assert result == {
            "file_data": {"mime_type": "image/png", "file_uri": url},
        }

    def test_gcs_uri_to_file_data(self):
        url = "gs://my-bucket/image.jpg"
        result = to_gemini_image(url)
        assert result == {
            "file_data": {"mime_type": "image/jpeg", "file_uri": url},
        }

    @patch("nanollm._image.download_image_as_base64")
    def test_http_url_downloads_and_inlines(self, mock_download):
        mock_download.return_value = ("image/jpeg", "downloaded_data")
        url = "http://example.com/image.jpg"
        result = to_gemini_image(url)
        mock_download.assert_called_once_with(url)
        assert result == {
            "inline_data": {"mime_type": "image/jpeg", "data": "downloaded_data"},
        }

    def test_https_url_with_query_params(self):
        url = "https://example.com/image.webp?token=abc"
        result = to_gemini_image(url)
        assert result["file_data"]["file_uri"] == url
        assert result["file_data"]["mime_type"] == "image/webp"


# ── to_bedrock_image ──────────────────────────────────────────────────


class TestToBedrockImage:
    def test_data_uri_to_bytes_and_format(self):
        uri = "data:image/png;base64,iVBORw0KGgo"
        result = to_bedrock_image(uri)
        assert result == {
            "image": {"source": {"bytes": "iVBORw0KGgo"}, "format": "png"},
        }

    def test_data_uri_jpeg_format(self):
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        result = to_bedrock_image(uri)
        assert result["image"]["format"] == "jpeg"

    @patch("nanollm._image.download_image_as_base64")
    def test_url_downloads_and_converts(self, mock_download):
        mock_download.return_value = ("image/png", "downloaded_png")
        url = "https://example.com/image.png"
        result = to_bedrock_image(url)
        mock_download.assert_called_once_with(url)
        assert result == {
            "image": {"source": {"bytes": "downloaded_png"}, "format": "png"},
        }

    @patch("nanollm._image.download_image_as_base64")
    def test_http_url_downloads(self, mock_download):
        mock_download.return_value = ("image/gif", "gif_data")
        result = to_bedrock_image("http://example.com/anim.gif")
        assert result["image"]["format"] == "gif"

    def test_data_uri_webp_format(self):
        uri = "data:image/webp;base64,UklGRl"
        result = to_bedrock_image(uri)
        assert result["image"]["format"] == "webp"


# ── download_image_as_base64 ─────────────────────────────────────────


class TestDownloadImageAsBase64:
    @patch("nanollm._image.httpx.Client")
    def test_download_with_content_type(self, mock_client_cls):
        raw_bytes = b"fake image bytes"
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/png; charset=utf-8"}
        mock_response.content = raw_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        mime, b64 = download_image_as_base64("https://example.com/img.png")
        assert mime == "image/png"
        assert b64 == base64.b64encode(raw_bytes).decode("ascii")

    @patch("nanollm._image.httpx.Client")
    def test_download_without_image_content_type(self, mock_client_cls):
        raw_bytes = b"bytes"
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = raw_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        mime, b64 = download_image_as_base64("https://example.com/img.webp")
        assert mime == "image/webp"  # falls back to URL extension


class TestAsyncDownloadImageAsBase64:
    @patch("nanollm._image.httpx.AsyncClient")
    async def test_async_download(self, mock_client_cls):
        from unittest.mock import AsyncMock

        raw_bytes = b"async image bytes"
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = raw_bytes
        mock_response.raise_for_status = MagicMock()

        # Build an async context manager that returns a client with async get
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_client_cls.return_value = mock_client

        mime, b64 = await async_download_image_as_base64("https://example.com/img.jpg")
        assert mime == "image/jpeg"
        assert b64 == base64.b64encode(raw_bytes).decode("ascii")
