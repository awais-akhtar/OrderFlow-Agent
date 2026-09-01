"""Shared runtime configuration for the customer and staff Reflex pages."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .providers import ProviderRegistry, ProviderUnavailable, RuntimeSettings


_LOCK = RLock()
_SETTINGS: RuntimeSettings | None = None
_PROVIDER: object | None = None
_PROVIDER_KEY: tuple[str, ...] | None = None


def current_settings() -> RuntimeSettings:
    global _SETTINGS
    with _LOCK:
        if _SETTINGS is None:
            _SETTINGS = RuntimeSettings.from_env()
        return _SETTINGS


def configure_runtime(
    *,
    provider_id: str,
    api_key: str = "",
    response_model: str = "",
    local_model_path: str = "",
    device: str = "CPU",
    base_url: str = "",
) -> RuntimeSettings:
    global _PROVIDER, _PROVIDER_KEY, _SETTINGS
    selected = provider_id.strip().casefold()
    if selected not in {"kernelloom", "openai", "huggingface", "openagent", "disabled"}:
        raise ValueError("Unknown model provider.")
    with _LOCK:
        previous = current_settings()
        _SETTINGS = replace(
            previous,
            provider_id=selected,
            api_key=api_key.strip(),
            response_model=response_model.strip() or previous.response_model,
            base_url=base_url.strip() or previous.base_url,
            kernelloom_transport="python" if selected == "kernelloom" and local_model_path else previous.kernelloom_transport,
            kernelloom_chat_model_path=local_model_path.strip() or previous.kernelloom_chat_model_path,
            kernelloom_backend="openvino" if selected == "kernelloom" and local_model_path else previous.kernelloom_backend,
            kernelloom_device=device.strip().upper() or "CPU",
        )
        stale = _PROVIDER
        _PROVIDER = None
        _PROVIDER_KEY = None
        if stale is not None:
            close = getattr(stale, "close", None)
            if callable(close):
                close()
        return _SETTINGS


def streaming_provider() -> object:
    global _PROVIDER, _PROVIDER_KEY
    settings = current_settings()
    key = (
        settings.provider_id,
        settings.base_url,
        settings.response_model,
        settings.api_key,
        settings.kernelloom_transport,
        settings.kernelloom_chat_model_path,
        settings.kernelloom_backend,
        settings.kernelloom_device,
    )
    with _LOCK:
        if _PROVIDER is not None and _PROVIDER_KEY == key:
            return _PROVIDER
        provider = ProviderRegistry.build(settings)
        if provider is None:
            raise ProviderUnavailable("No streaming language model is configured.")
        _PROVIDER = provider
        _PROVIDER_KEY = key
        return provider


def check_runtime() -> tuple[bool, str]:
    try:
        provider = streaming_provider()
        status = provider.check()
        return bool(status.ok), str(status.detail)
    except Exception as exc:
        return False, str(exc)


__all__ = ["check_runtime", "configure_runtime", "current_settings", "streaming_provider"]
