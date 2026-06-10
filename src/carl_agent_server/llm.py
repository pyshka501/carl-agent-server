"""LLM client construction for deployed agents (spec fields → env fallback)."""

from __future__ import annotations

import os
from typing import Any

from .models import DeploymentSpec

ENV_API_KEY = "AGENT_LLM_API_KEY"
ENV_BASE_URL = "AGENT_LLM_BASE_URL"
ENV_MODEL = "AGENT_LLM_MODEL"


class LLMNotConfiguredError(RuntimeError):
    """The deployment has no usable LLM configuration."""


def build_llm_client(spec: DeploymentSpec) -> Any:
    """Build an OpenAI-compatible CARL LLM client from the spec, falling back to env vars."""
    api_key = spec.llm_api_key or os.environ.get(ENV_API_KEY)
    model = spec.llm_model or os.environ.get(ENV_MODEL)
    base_url = spec.llm_base_url or os.environ.get(ENV_BASE_URL)
    if not api_key or not model:
        raise LLMNotConfiguredError(
            "LLM is not configured: set llm_api_key/llm_model on the deployment "
            f"or the {ENV_API_KEY}/{ENV_MODEL} environment variables"
        )
    from mmar_carl import OpenAICompatibleClient
    from mmar_carl.llm import OpenAIClientConfig

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "temperature": spec.llm_temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAICompatibleClient(OpenAIClientConfig(**kwargs))
