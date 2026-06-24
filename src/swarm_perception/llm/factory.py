"""Factory for constructing provider-backed API managers."""

from typing import Any

from swarm_perception.llm.async_manager import AsyncAPI_MANAGER
from swarm_perception.llm.manager import API_MANAGER
from swarm_perception.llm.providers.gemini import GeminiProvider
from swarm_perception.llm.providers.ollama import OllamaProvider
from swarm_perception.llm.providers.openai_provider import OpenAIProvider
from swarm_perception.llm.providers.vllm import VllmProvider


def create_api_manager(n_threads: int, config: Any) -> API_MANAGER | AsyncAPI_MANAGER:
    """Build an API manager for the LLM provider named in config.

    vLLM uses ``AsyncAPI_MANAGER`` so up to ``thread_workers`` HTTP requests
    run in parallel without blocking each other.

    Args:
        n_threads: Worker / concurrency limit for queued requests.
        config: Loaded swarm configuration namespace.

    Returns:
        Configured API manager instance.

    Raises:
        ValueError: If ``llm.provider`` names an unknown backend.
    """
    provider_name = getattr(config.llm, "provider", "gemini")
    llm_config = config.llm

    if provider_name == "gemini":
        provider = GeminiProvider(llm_config)
        return API_MANAGER(n_threads, config, provider)
    if provider_name == "openai":
        provider = OpenAIProvider(llm_config)
        return API_MANAGER(n_threads, config, provider)
    if provider_name == "ollama":
        provider = OllamaProvider(llm_config)
        return API_MANAGER(n_threads, config, provider)
    if provider_name == "vllm":
        provider = VllmProvider(llm_config)
        return AsyncAPI_MANAGER(n_threads, config, provider)

    raise ValueError(
        f"Unknown llm.provider: {provider_name!r}. "
        "Supported values: gemini, openai, ollama, vllm."
    )
