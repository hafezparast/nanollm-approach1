"""Provider registry — configuration for every supported LLM provider.

Each provider entry defines how to reach the API, which environment variable
holds the key, which adapter handles request/response translation, and which
parameters the provider actually supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Supported parameter sets ─────────────────────────────────────────

OPENAI_PARAMS: frozenset[str] = frozenset({
    "temperature", "top_p", "max_tokens", "n", "stop",
    "presence_penalty", "frequency_penalty", "logit_bias", "seed",
    "tools", "tool_choice", "response_format", "stream",
    "logprobs", "top_logprobs", "user",
    "functions", "function_call",
})

ANTHROPIC_PARAMS: frozenset[str] = frozenset({
    "temperature", "top_p", "top_k", "max_tokens", "stop",
    "tools", "tool_choice", "stream", "system",
})

GEMINI_PARAMS: frozenset[str] = frozenset({
    "temperature", "top_p", "top_k", "max_tokens", "n", "stop",
    "response_format", "safety_settings", "stream",
    "tools", "tool_choice", "system_instruction",
})


# ── Provider config dataclass ────────────────────────────────────────


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for a single LLM provider."""

    name: str
    base_url: str = ""
    api_key_env: str = ""
    adapter: str = "openai_compat"
    supported_params: frozenset[str] = field(default_factory=lambda: OPENAI_PARAMS)
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    extra_headers: dict[str, str] = field(default_factory=dict)


# ── Provider registry ────────────────────────────────────────────────

_PROVIDERS: dict[str, ProviderConfig] = {}


def _register(*names: str, **kwargs: Any) -> None:
    """Register a provider config under one or more names."""
    cfg = ProviderConfig(**kwargs)
    for name in names:
        _PROVIDERS[name] = cfg


# OpenAI-compatible cloud providers
_register(
    "openai",
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
)
_register(
    "groq",
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
)
_register(
    "together", "together_ai",
    name="together",
    base_url="https://api.together.xyz/v1",
    api_key_env="TOGETHER_API_KEY",
)
_register(
    "mistral",
    name="mistral",
    base_url="https://api.mistral.ai/v1",
    api_key_env="MISTRAL_API_KEY",
)
_register(
    "deepseek",
    name="deepseek",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
)
_register(
    "perplexity",
    name="perplexity",
    base_url="https://api.perplexity.ai",
    api_key_env="PERPLEXITYAI_API_KEY",
)
_register(
    "fireworks", "fireworks_ai",
    name="fireworks",
    base_url="https://api.fireworks.ai/inference/v1",
    api_key_env="FIREWORKS_API_KEY",
)
_register(
    "openrouter",
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
)
_register(
    "deepinfra",
    name="deepinfra",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key_env="DEEPINFRA_API_KEY",
)
_register(
    "anyscale",
    name="anyscale",
    base_url="https://api.anyscale.com/v1",
    api_key_env="ANYSCALE_API_KEY",
)
_register(
    "xai",
    name="xai",
    base_url="https://api.x.ai/v1",
    api_key_env="XAI_API_KEY",
)
_register(
    "cerebras",
    name="cerebras",
    base_url="https://api.cerebras.ai/v1",
    api_key_env="CEREBRAS_API_KEY",
)
_register(
    "custom_openai",
    name="custom_openai",
    base_url="",
    api_key_env="",
)
_register(
    "lm_studio",
    name="lm_studio",
    base_url="http://localhost:1234/v1",
    api_key_env="",
)

# Non-OpenAI-compatible providers
_register(
    "anthropic",
    name="anthropic",
    base_url="https://api.anthropic.com/v1",
    api_key_env="ANTHROPIC_API_KEY",
    adapter="anthropic",
    supported_params=ANTHROPIC_PARAMS,
    auth_header="x-api-key",
    auth_prefix="",
)
_register(
    "gemini",
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GEMINI_API_KEY",
    adapter="gemini",
    supported_params=GEMINI_PARAMS,
)

# Local inference
_register(
    "ollama", "ollama_chat",
    name="ollama",
    base_url="http://localhost:11434/v1",
    api_key_env="",
    adapter="ollama",
)

# Cloud-specific (require custom auth or endpoints)
_register(
    "azure",
    name="azure",
    base_url="",
    api_key_env="AZURE_API_KEY",
    adapter="azure_openai",
    auth_header="api-key",
    auth_prefix="",
)
_register(
    "bedrock",
    name="bedrock",
    base_url="",
    api_key_env="",
    adapter="bedrock",
)
_register(
    "vertex_ai",
    name="vertex_ai",
    base_url="",
    api_key_env="",
    adapter="vertex",
)


# ── Public API ───────────────────────────────────────────────────────


def get_provider_config(provider: str) -> ProviderConfig:
    """Look up a provider config by name.

    Args:
        provider: Provider name (e.g. "openai", "anthropic", "together_ai").

    Returns:
        The corresponding ProviderConfig.

    Raises:
        ValueError: If the provider is not registered.
    """
    try:
        return _PROVIDERS[provider]
    except KeyError:
        available = sorted(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available providers: {', '.join(available)}"
        ) from None


def list_providers() -> list[str]:
    """Return sorted list of all registered provider names."""
    return sorted(_PROVIDERS.keys())
