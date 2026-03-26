"""Exhaustive tests for nanollm._router and nanollm._config."""

from __future__ import annotations

import pytest

from nanollm._config import (
    ANTHROPIC_PARAMS,
    GEMINI_PARAMS,
    OPENAI_PARAMS,
    ProviderConfig,
    get_provider_config,
    list_providers,
)
from nanollm._router import (
    _SLASH_IN_MODEL,
    _adapter_cache,
    get_adapter,
    parse_model_string,
    resolve,
)


# ════════════════════════════════════════════════════════════════════════
# parse_model_string
# ════════════════════════════════════════════════════════════════════════


class TestParseModelString:
    # ── Standard provider/model format ──

    def test_openai_gpt4o(self):
        assert parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_anthropic_claude3(self):
        assert parse_model_string("anthropic/claude-3-opus") == ("anthropic", "claude-3-opus")

    def test_groq_llama3(self):
        assert parse_model_string("groq/llama3-70b-8192") == ("groq", "llama3-70b-8192")

    def test_gemini(self):
        assert parse_model_string("gemini/gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_deepseek(self):
        assert parse_model_string("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_ollama(self):
        assert parse_model_string("ollama/llama3") == ("ollama", "llama3")

    def test_mistral(self):
        assert parse_model_string("mistral/mistral-large-latest") == ("mistral", "mistral-large-latest")

    # ── Providers with slashes in model names ──

    def test_together_ai_with_slash(self):
        result = parse_model_string("together_ai/meta-llama/Llama-3.3-70B")
        assert result == ("together_ai", "meta-llama/Llama-3.3-70B")

    def test_openrouter_with_slash(self):
        result = parse_model_string("openrouter/anthropic/claude-3.5-sonnet")
        assert result == ("openrouter", "anthropic/claude-3.5-sonnet")

    def test_bedrock_dotted_model(self):
        result = parse_model_string("bedrock/anthropic.claude-3-sonnet")
        assert result == ("bedrock", "anthropic.claude-3-sonnet")

    def test_vertex_ai(self):
        result = parse_model_string("vertex_ai/gemini-pro")
        assert result == ("vertex_ai", "gemini-pro")

    def test_deepinfra_with_slash(self):
        result = parse_model_string("deepinfra/meta-llama/Llama-3-70b")
        assert result == ("deepinfra", "meta-llama/Llama-3-70b")

    def test_fireworks_with_slash(self):
        result = parse_model_string("fireworks/accounts/fireworks/models/llama-v3")
        assert result == ("fireworks", "accounts/fireworks/models/llama-v3")

    # ── Edge cases ──

    def test_bare_model_name_defaults_to_openai(self):
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_bare_model_complex_name(self):
        assert parse_model_string("gpt-3.5-turbo-0125") == ("openai", "gpt-3.5-turbo-0125")

    def test_empty_string(self):
        assert parse_model_string("") == ("openai", "")

    def test_trailing_slash(self):
        provider, model_id = parse_model_string("openai/")
        assert provider == "openai"
        assert model_id == ""

    def test_multiple_slashes_standard_provider(self):
        """Non-slash providers still only split on first slash."""
        result = parse_model_string("openai/ft:gpt-4o:my-org:custom")
        assert result == ("openai", "ft:gpt-4o:my-org:custom")

    def test_provider_lowercased(self):
        assert parse_model_string("OpenAI/gpt-4o") == ("openai", "gpt-4o")

    def test_provider_stripped(self):
        assert parse_model_string(" openai /gpt-4o") == ("openai", "gpt-4o")

    def test_model_stripped(self):
        assert parse_model_string("openai/ gpt-4o ") == ("openai", "gpt-4o")

    def test_together_alias(self):
        result = parse_model_string("together/meta-llama/Llama-3-70b")
        assert result == ("together", "meta-llama/Llama-3-70b")


# ════════════════════════════════════════════════════════════════════════
# _SLASH_IN_MODEL constant
# ════════════════════════════════════════════════════════════════════════


class TestSlashInModel:
    def test_contains_together(self):
        assert "together" in _SLASH_IN_MODEL
        assert "together_ai" in _SLASH_IN_MODEL

    def test_contains_openrouter(self):
        assert "openrouter" in _SLASH_IN_MODEL

    def test_contains_bedrock(self):
        assert "bedrock" in _SLASH_IN_MODEL

    def test_contains_vertex_ai(self):
        assert "vertex_ai" in _SLASH_IN_MODEL

    def test_contains_deepinfra(self):
        assert "deepinfra" in _SLASH_IN_MODEL

    def test_contains_fireworks(self):
        assert "fireworks" in _SLASH_IN_MODEL
        assert "fireworks_ai" in _SLASH_IN_MODEL

    def test_does_not_contain_openai(self):
        assert "openai" not in _SLASH_IN_MODEL


# ════════════════════════════════════════════════════════════════════════
# get_provider_config
# ════════════════════════════════════════════════════════════════════════


class TestGetProviderConfig:
    def test_openai(self):
        cfg = get_provider_config("openai")
        assert cfg.name == "openai"
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.api_key_env == "OPENAI_API_KEY"

    def test_anthropic(self):
        cfg = get_provider_config("anthropic")
        assert cfg.name == "anthropic"
        assert cfg.adapter == "anthropic"
        assert cfg.auth_header == "x-api-key"
        assert cfg.auth_prefix == ""
        assert cfg.supported_params is ANTHROPIC_PARAMS

    def test_gemini(self):
        cfg = get_provider_config("gemini")
        assert cfg.adapter == "gemini"
        assert cfg.supported_params is GEMINI_PARAMS

    def test_groq(self):
        cfg = get_provider_config("groq")
        assert cfg.name == "groq"
        assert "groq.com" in cfg.base_url

    def test_together_ai_alias(self):
        cfg1 = get_provider_config("together")
        cfg2 = get_provider_config("together_ai")
        assert cfg1 is cfg2

    def test_ollama(self):
        cfg = get_provider_config("ollama")
        assert cfg.adapter == "ollama"
        assert "11434" in cfg.base_url

    def test_ollama_chat_alias(self):
        cfg1 = get_provider_config("ollama")
        cfg2 = get_provider_config("ollama_chat")
        assert cfg1 is cfg2

    def test_bedrock(self):
        cfg = get_provider_config("bedrock")
        assert cfg.adapter == "bedrock"
        assert cfg.base_url == ""

    def test_vertex_ai(self):
        cfg = get_provider_config("vertex_ai")
        assert cfg.adapter == "vertex"

    def test_azure(self):
        cfg = get_provider_config("azure")
        assert cfg.adapter == "azure_openai"
        assert cfg.auth_header == "api-key"

    def test_deepseek(self):
        cfg = get_provider_config("deepseek")
        assert "deepseek.com" in cfg.base_url

    def test_openrouter(self):
        cfg = get_provider_config("openrouter")
        assert "openrouter.ai" in cfg.base_url

    def test_fireworks_alias(self):
        cfg1 = get_provider_config("fireworks")
        cfg2 = get_provider_config("fireworks_ai")
        assert cfg1 is cfg2

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider_config("nonexistent_provider")

    def test_unknown_lists_available(self):
        with pytest.raises(ValueError, match="Available providers"):
            get_provider_config("nope")

    def test_all_registered_providers_exist(self):
        expected = [
            "openai", "groq", "together", "together_ai", "mistral",
            "deepseek", "perplexity", "fireworks", "fireworks_ai",
            "openrouter", "deepinfra", "anyscale", "xai", "cerebras",
            "custom_openai", "lm_studio", "anthropic", "gemini",
            "ollama", "ollama_chat", "azure", "bedrock", "vertex_ai",
        ]
        for name in expected:
            cfg = get_provider_config(name)
            assert isinstance(cfg, ProviderConfig)


# ════════════════════════════════════════════════════════════════════════
# list_providers
# ════════════════════════════════════════════════════════════════════════


class TestListProviders:
    def test_returns_sorted_list(self):
        providers = list_providers()
        assert providers == sorted(providers)

    def test_contains_core_providers(self):
        providers = list_providers()
        for p in ["openai", "anthropic", "gemini", "groq", "ollama"]:
            assert p in providers

    def test_returns_list_of_strings(self):
        providers = list_providers()
        assert all(isinstance(p, str) for p in providers)


# ════════════════════════════════════════════════════════════════════════
# ProviderConfig
# ════════════════════════════════════════════════════════════════════════


class TestProviderConfig:
    def test_frozen(self):
        cfg = ProviderConfig(name="test")
        with pytest.raises(AttributeError):
            cfg.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        cfg = ProviderConfig(name="t")
        assert cfg.base_url == ""
        assert cfg.api_key_env == ""
        assert cfg.adapter == "openai_compat"
        assert cfg.supported_params == OPENAI_PARAMS
        assert cfg.auth_header == "Authorization"
        assert cfg.auth_prefix == "Bearer"
        assert cfg.extra_headers == {}


# ════════════════════════════════════════════════════════════════════════
# get_adapter
# ════════════════════════════════════════════════════════════════════════


class TestGetAdapter:
    def test_openai_loads(self):
        adapter = get_adapter("openai")
        assert adapter is not None
        assert hasattr(adapter, "build_request")
        assert hasattr(adapter, "parse_response")

    def test_anthropic_loads(self):
        adapter = get_adapter("anthropic")
        assert hasattr(adapter, "build_request")

    def test_gemini_loads(self):
        adapter = get_adapter("gemini")
        assert hasattr(adapter, "build_request")

    def test_ollama_loads(self):
        adapter = get_adapter("ollama")
        assert hasattr(adapter, "build_request")

    def test_caching_returns_same_object(self):
        a1 = get_adapter("openai")
        a2 = get_adapter("openai")
        assert a1 is a2

    def test_different_providers_same_adapter_cached(self):
        """groq and openai both use openai_compat adapter."""
        a_openai = get_adapter("openai")
        a_groq = get_adapter("groq")
        # They share the same adapter module name, so same cached instance
        assert a_openai is a_groq

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_adapter("nonexistent_provider_xyz")


# ════════════════════════════════════════════════════════════════════════
# resolve
# ════════════════════════════════════════════════════════════════════════


class TestResolve:
    def test_full_chain(self):
        provider, model_id, adapter, config = resolve("openai/gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"
        assert config.name == "openai"
        assert hasattr(adapter, "build_request")

    def test_bare_model_resolves_to_openai(self):
        provider, model_id, adapter, config = resolve("gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"

    def test_anthropic_resolve(self):
        provider, model_id, adapter, config = resolve("anthropic/claude-3-opus")
        assert provider == "anthropic"
        assert model_id == "claude-3-opus"
        assert config.adapter == "anthropic"

    def test_base_url_override_creates_new_config(self):
        _, _, _, config = resolve("openai/gpt-4o", base_url="https://custom.api.com/v1")
        assert config.base_url == "https://custom.api.com/v1"

    def test_api_base_override(self):
        """api_base is a litellm-compat alias for base_url."""
        _, _, _, config = resolve("openai/gpt-4o", api_base="https://alt.api.com/v1")
        assert config.base_url == "https://alt.api.com/v1"

    def test_base_url_takes_priority_over_api_base(self):
        _, _, _, config = resolve(
            "openai/gpt-4o",
            base_url="https://primary.com/v1",
            api_base="https://secondary.com/v1",
        )
        assert config.base_url == "https://primary.com/v1"

    def test_no_override_keeps_original_config(self):
        _, _, _, config = resolve("openai/gpt-4o")
        assert config.base_url == "https://api.openai.com/v1"

    def test_override_preserves_other_fields(self):
        _, _, _, config = resolve("anthropic/claude-3", base_url="https://custom.com/v1")
        assert config.auth_header == "x-api-key"
        assert config.auth_prefix == ""
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.adapter == "anthropic"

    def test_bedrock_no_base_url_ok(self):
        """bedrock and vertex_ai don't require base_url."""
        provider, model_id, adapter, config = resolve("bedrock/anthropic.claude-3")
        assert provider == "bedrock"

    def test_vertex_ai_no_base_url_ok(self):
        provider, model_id, adapter, config = resolve("vertex_ai/gemini-pro")
        assert provider == "vertex_ai"

    def test_groq_resolve(self):
        provider, model_id, _, config = resolve("groq/llama3-70b-8192")
        assert provider == "groq"
        assert model_id == "llama3-70b-8192"

    def test_gemini_resolve(self):
        provider, model_id, _, config = resolve("gemini/gemini-2.0-flash")
        assert provider == "gemini"
        assert config.adapter == "gemini"
