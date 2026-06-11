"""Backward-compatible re-exports for the LLM manager.

Prefer ``llm.factory.create_api_manager`` for new code.
"""

from llm.factory import create_api_manager
from llm.manager import API_MANAGER

__all__ = ["API_MANAGER", "create_api_manager"]
