"""Factory for constructing provider-backed API managers."""

from typing import Any

from llm.manager import API_MANAGER
from llm.providers.gemini import GeminiProvider
from llm.providers.ollama import OllamaProvider
from llm.providers.openai_provider import OpenAIProvider


def create_api_manager(n_threads: int, config: Any) -> API_MANAGER:
    """Build an API manager for the LLM provider named in config.

    Args:
        n_threads: Worker thread count for queued requests.
        config: Loaded swarm configuration namespace.

    Returns:
        Configured ``API_MANAGER`` instance.

    Raises:
        ValueError: If ``llm.provider`` names an unknown backend.
    """
    provider_name = getattr(config.llm, "provider", "gemini")
    llm_config = config.llm

    if provider_name == "gemini":
        provider = GeminiProvider(llm_config)
    elif provider_name == "openai":
        provider = OpenAIProvider(llm_config)
    elif provider_name == "ollama":
        provider = OllamaProvider(llm_config)
    else:
        raise ValueError(
            f"Unknown llm.provider: {provider_name!r}. "
            "Supported values: gemini, openai, ollama."
        )

    return API_MANAGER(n_threads, config, provider)
