"""Exhaustive tests for the public API functions in nanollm/__init__.py.

Uses unittest.mock to mock _http calls and adapter resolution.
Tests: completion, acompletion, batch_completion, embedding, aembedding,
text_completion, atext_completion, drop_params filtering, error propagation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nanollm
from nanollm import (
    acompletion,
    aembedding,
    atext_completion,
    batch_completion,
    completion,
    embedding,
    text_completion,
)
from nanollm._types import (
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
    make_model_response,
)
from nanollm.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _mock_adapter():
    """Create a mock adapter with build_request, parse_response, etc."""
    adapter = MagicMock()
    adapter.build_request.return_value = (
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": "Bearer test"},
        {"model": "gpt-4o", "messages": [], "stream": False},
    )
    adapter.parse_response.return_value = make_model_response(
        content="Hello!", model="gpt-4o", finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    adapter.filter_params.side_effect = lambda kwargs, supported: kwargs
    adapter.build_embedding_request.return_value = (
        "https://api.openai.com/v1/embeddings",
        {"Authorization": "Bearer test"},
        {"model": "text-embedding-ada-002", "input": ["hello"]},
    )
    from nanollm._types import make_embedding_response
    adapter.parse_embedding_response.return_value = make_embedding_response(
        embeddings=[[0.1, 0.2, 0.3]], model="text-embedding-ada-002",
    )
    return adapter


def _mock_config():
    """Create a mock ProviderConfig."""
    from nanollm._config import ProviderConfig
    return ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        adapter="openai_compat",
    )


def _mock_resolve(adapter=None, config=None):
    """Return a mock for nanollm._router.resolve."""
    a = adapter or _mock_adapter()
    c = config or _mock_config()
    return MagicMock(return_value=("openai", "gpt-4o", a, c))


# ═══════════════════════════════════════════════════════════════════════
# completion
# ═══════════════════════════════════════════════════════════════════════


class TestCompletion:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_non_streaming(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {"choices": [{"message": {"content": "Hi"}}]}

        result = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, ModelResponse)
        adapter.build_request.assert_called_once()
        adapter.parse_response.assert_called_once()

    @patch("nanollm.sync_stream")
    @patch("nanollm.resolve")
    def test_streaming(self, mock_resolve, mock_stream):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_stream.return_value = iter(["chunk1", "chunk2"])

        result = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        # Should return a stream iterator, not a ModelResponse
        assert hasattr(result, "__iter__")

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_with_api_key(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(
            model="openai/gpt-4o",
            messages=[],
            api_key="sk-custom-key",
        )
        # build_request should receive the api_key
        call_kwargs = adapter.build_request.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-custom-key" or \
               call_kwargs[1].get("api_key") == "sk-custom-key"

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_with_kwargs(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(
            model="openai/gpt-4o",
            messages=[],
            temperature=0.5,
            max_tokens=100,
        )
        # kwargs should be passed through (after filter_params)
        adapter.filter_params.assert_called_once()

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_with_base_url_override(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(
            model="openai/gpt-4o",
            messages=[],
            base_url="https://custom.api.com/v1",
        )
        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args
        # base_url should be passed to resolve
        assert "custom.api.com" in str(call_kwargs)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_api_base_alias(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(
            model="openai/gpt-4o",
            messages=[],
            api_base="https://custom.api.com/v1",
        )
        # api_base should work as alias for base_url

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_default_timeout(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(model="openai/gpt-4o", messages=[])
        # sync_post should be called with timeout=600.0
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("timeout") == 600.0 or \
               (len(call_kwargs.args) > 3 and call_kwargs.args[3] == 600.0)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_custom_timeout(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(model="openai/gpt-4o", messages=[], timeout=30.0)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_none_messages_defaults_to_empty_list(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        completion(model="openai/gpt-4o")
        call_kwargs = adapter.build_request.call_args
        assert call_kwargs.kwargs.get("messages") == [] or \
               call_kwargs[1].get("messages") == []


# ═══════════════════════════════════════════════════════════════════════
# acompletion
# ═══════════════════════════════════════════════════════════════════════


class TestAcompletion:
    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_non_streaming(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {"choices": [{"message": {"content": "Hi"}}]}

        result = await acompletion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, ModelResponse)

    @patch("nanollm.async_stream")
    @patch("nanollm.resolve")
    async def test_streaming(self, mock_resolve, mock_stream):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())

        async def mock_gen():
            yield "chunk1"
            yield "chunk2"

        mock_stream.return_value = mock_gen()

        result = await acompletion(
            model="openai/gpt-4o",
            messages=[],
            stream=True,
        )
        assert hasattr(result, "__aiter__")

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_with_api_key(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        await acompletion(
            model="openai/gpt-4o",
            messages=[],
            api_key="sk-async-key",
        )
        call_kwargs = adapter.build_request.call_args
        assert "sk-async-key" in str(call_kwargs)

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_with_kwargs(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        await acompletion(
            model="openai/gpt-4o",
            messages=[],
            temperature=0.7,
        )
        adapter.filter_params.assert_called_once()

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_none_messages_defaults_to_empty(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        await acompletion(model="openai/gpt-4o")


# ═══════════════════════════════════════════════════════════════════════
# batch_completion
# ═══════════════════════════════════════════════════════════════════════


class TestBatchCompletion:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_multiple_messages(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        msgs_list = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}],
        ]
        results = batch_completion(
            model="openai/gpt-4o",
            messages=msgs_list,
        )
        assert len(results) == 2
        assert all(isinstance(r, ModelResponse) for r in results)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_single_message(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        results = batch_completion(
            model="openai/gpt-4o",
            messages=[[{"role": "user", "content": "Hi"}]],
        )
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════
# embedding
# ═══════════════════════════════════════════════════════════════════════


class TestEmbedding:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_string_input(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        result = embedding(
            model="openai/text-embedding-ada-002",
            input="Hello world",
        )
        assert isinstance(result, EmbeddingResponse)
        # String input should be wrapped in list
        call_kwargs = adapter.build_embedding_request.call_args
        input_val = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
        assert isinstance(input_val, list)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_list_input(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        result = embedding(
            model="openai/text-embedding-ada-002",
            input=["Hello", "World"],
        )
        assert isinstance(result, EmbeddingResponse)

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_with_api_key(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        embedding(
            model="openai/text-embedding-ada-002",
            input=["test"],
            api_key="sk-embed-key",
        )

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_api_base_alias(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        embedding(
            model="openai/text-embedding-ada-002",
            input=["test"],
            api_base="https://custom.com/v1",
        )


# ═══════════════════════════════════════════════════════════════════════
# aembedding
# ═══════════════════════════════════════════════════════════════════════


class TestAembedding:
    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_string_input(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        result = await aembedding(
            model="openai/text-embedding-ada-002",
            input="Hello",
        )
        assert isinstance(result, EmbeddingResponse)

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_list_input(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.return_value = {}

        result = await aembedding(
            model="openai/text-embedding-ada-002",
            input=["Hello", "World"],
        )
        assert isinstance(result, EmbeddingResponse)


# ═══════════════════════════════════════════════════════════════════════
# text_completion
# ═══════════════════════════════════════════════════════════════════════


class TestTextCompletion:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_non_streaming_returns_text_completion_response(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        result = text_completion(
            model="openai/gpt-4o",
            prompt="Once upon a time",
        )
        assert isinstance(result, TextCompletionResponse)
        assert result.object == "text_completion"

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_text_content_extracted(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        adapter.parse_response.return_value = make_model_response(
            content="there was a dragon", model="gpt-4o",
        )
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        result = text_completion(
            model="openai/gpt-4o",
            prompt="Once upon a time",
        )
        assert result.choices[0].text == "there was a dragon"

    @patch("nanollm.sync_stream")
    @patch("nanollm.resolve")
    def test_streaming_passthrough(self, mock_resolve, mock_stream):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_stream.return_value = iter(["chunk"])

        result = text_completion(
            model="openai/gpt-4o",
            prompt="Hello",
            stream=True,
        )
        # Streaming should return a stream iterator
        assert hasattr(result, "__iter__")

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_prompt_converted_to_messages(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        text_completion(model="openai/gpt-4o", prompt="Tell me a story")
        call_kwargs = adapter.build_request.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages == [{"role": "user", "content": "Tell me a story"}]

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_finish_reason_preserved(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        adapter.parse_response.return_value = make_model_response(
            content="text", model="gpt-4o", finish_reason="length",
        )
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        result = text_completion(model="openai/gpt-4o", prompt="Hi")
        assert result.choices[0].finish_reason == "length"


# ═══════════════════════════════════════════════════════════════════════
# atext_completion
# ═══════════════════════════════════════════════════════════════════════


class TestAtextCompletion:
    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_non_streaming(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        result = await atext_completion(
            model="openai/gpt-4o",
            prompt="Hello",
        )
        assert isinstance(result, TextCompletionResponse)

    @patch("nanollm.async_stream")
    @patch("nanollm.resolve")
    async def test_streaming(self, mock_resolve, mock_stream):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())

        async def mock_gen():
            yield "chunk"

        mock_stream.return_value = mock_gen()

        result = await atext_completion(
            model="openai/gpt-4o",
            prompt="Hello",
            stream=True,
        )
        assert hasattr(result, "__aiter__")

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_text_content_extracted(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        adapter.parse_response.return_value = make_model_response(
            content="async result", model="gpt-4o",
        )
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        result = await atext_completion(model="openai/gpt-4o", prompt="Hi")
        assert result.choices[0].text == "async result"


# ═══════════════════════════════════════════════════════════════════════
# drop_params filtering
# ═══════════════════════════════════════════════════════════════════════


class TestDropParams:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_filter_params_called_when_drop_params_true(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        original = nanollm.drop_params
        try:
            nanollm.drop_params = True
            completion(model="openai/gpt-4o", messages=[], unsupported_param="value")
            adapter.filter_params.assert_called_once()
        finally:
            nanollm.drop_params = original

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_filter_params_not_called_when_drop_params_false(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.return_value = {}

        original = nanollm.drop_params
        try:
            nanollm.drop_params = False
            completion(model="openai/gpt-4o", messages=[])
            adapter.filter_params.assert_not_called()
        finally:
            nanollm.drop_params = original

    def test_filter_params_drops_unsupported(self):
        from nanollm._router import _ModuleAdapter
        result = _ModuleAdapter.filter_params(
            {"temperature": 0.5, "unsupported": True, "top_p": 0.9},
            frozenset({"temperature", "top_p"}),
        )
        assert "temperature" in result
        assert "top_p" in result
        assert "unsupported" not in result

    def test_filter_params_empty_supported(self):
        from nanollm._router import _ModuleAdapter
        result = _ModuleAdapter.filter_params(
            {"temperature": 0.5},
            frozenset(),
        )
        assert result == {}

    def test_filter_params_empty_kwargs(self):
        from nanollm._router import _ModuleAdapter
        result = _ModuleAdapter.filter_params({}, frozenset({"temperature"}))
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# Error propagation
# ═══════════════════════════════════════════════════════════════════════


class TestErrorPropagation:
    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_auth_error_propagates(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.side_effect = AuthenticationError(
            message="Invalid API key",
            status_code=401,
            llm_provider="openai",
            model="gpt-4o",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            completion(model="openai/gpt-4o", messages=[])
        assert exc_info.value.status_code == 401

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_rate_limit_error_propagates(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.side_effect = RateLimitError(
            message="Rate limit exceeded",
            status_code=429,
        )

        with pytest.raises(RateLimitError):
            completion(model="openai/gpt-4o", messages=[])

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_bad_request_error_propagates(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.side_effect = BadRequestError(
            message="Invalid request",
            status_code=400,
        )

        with pytest.raises(BadRequestError):
            completion(model="openai/gpt-4o", messages=[])

    @patch("nanollm.async_post", new_callable=AsyncMock)
    @patch("nanollm.resolve")
    async def test_async_auth_error_propagates(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.side_effect = AuthenticationError(
            message="Invalid API key",
            status_code=401,
        )

        with pytest.raises(AuthenticationError):
            await acompletion(model="openai/gpt-4o", messages=[])

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_embedding_error_propagates(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "text-embedding-ada-002", adapter, _mock_config())
        mock_post.side_effect = BadRequestError(
            message="Bad embedding request",
            status_code=400,
        )

        with pytest.raises(BadRequestError):
            embedding(model="openai/text-embedding-ada-002", input=["test"])

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            completion(model="nonexistent_provider/model", messages=[])

    @patch("nanollm.sync_post")
    @patch("nanollm.resolve")
    def test_exception_attributes(self, mock_resolve, mock_post):
        adapter = _mock_adapter()
        mock_resolve.return_value = ("openai", "gpt-4o", adapter, _mock_config())
        mock_post.side_effect = RateLimitError(
            message="Rate limit hit",
            status_code=429,
            llm_provider="openai",
            model="gpt-4o",
        )

        with pytest.raises(RateLimitError) as exc_info:
            completion(model="openai/gpt-4o", messages=[])
        assert exc_info.value.message == "Rate limit hit"
        assert exc_info.value.llm_provider == "openai"
        assert exc_info.value.model == "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════
# _resolve_api_key
# ═══════════════════════════════════════════════════════════════════════


class TestResolveApiKey:
    def test_explicit_key_takes_precedence(self):
        from nanollm import _resolve_api_key
        assert _resolve_api_key("explicit-key", "SOME_ENV_VAR") == "explicit-key"

    def test_env_var_used_when_no_explicit(self):
        import os
        from nanollm import _resolve_api_key
        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}):
            assert _resolve_api_key(None, "TEST_API_KEY") == "env-key"

    def test_none_when_both_missing(self):
        from nanollm import _resolve_api_key
        assert _resolve_api_key(None, None) is None

    def test_none_when_env_not_set(self):
        from nanollm import _resolve_api_key
        assert _resolve_api_key(None, "NONEXISTENT_VAR_12345") is None

    def test_empty_string_key_treated_as_falsy(self):
        from nanollm import _resolve_api_key
        # Empty string is falsy, so falls through to env var
        import os
        with patch.dict(os.environ, {"TEST_KEY": "env-val"}):
            result = _resolve_api_key("", "TEST_KEY")
            assert result == "env-val"


# ═══════════════════════════════════════════════════════════════════════
# raise_for_status
# ═══════════════════════════════════════════════════════════════════════


class TestRaiseForStatus:
    def test_200_no_raise(self):
        from nanollm.exceptions import raise_for_status
        raise_for_status(200, {})  # Should not raise

    def test_400_raises_bad_request(self):
        from nanollm.exceptions import raise_for_status
        with pytest.raises(BadRequestError):
            raise_for_status(400, {"error": {"message": "bad"}})

    def test_401_raises_auth_error(self):
        from nanollm.exceptions import raise_for_status
        with pytest.raises(AuthenticationError):
            raise_for_status(401, "Unauthorized")

    def test_429_raises_rate_limit(self):
        from nanollm.exceptions import raise_for_status
        with pytest.raises(RateLimitError):
            raise_for_status(429, {"error": {"message": "rate limited"}})

    def test_500_raises_internal_server_error(self):
        from nanollm.exceptions import raise_for_status, InternalServerError
        with pytest.raises(InternalServerError):
            raise_for_status(500, "Internal error")

    def test_unknown_status_raises_api_error(self):
        from nanollm.exceptions import raise_for_status, APIError
        with pytest.raises(APIError):
            raise_for_status(418, "I'm a teapot")

    def test_error_message_from_dict(self):
        from nanollm.exceptions import raise_for_status
        with pytest.raises(BadRequestError) as exc_info:
            raise_for_status(400, {"error": {"message": "specific error"}})
        assert "specific error" in str(exc_info.value)

    def test_error_message_from_string(self):
        from nanollm.exceptions import raise_for_status
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(401, "Invalid key")
        assert "Invalid key" in str(exc_info.value)
