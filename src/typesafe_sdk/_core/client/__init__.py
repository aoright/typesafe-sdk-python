"""Synchronous and asynchronous clients and their resource accessors."""

from collections.abc import Mapping

import httpx2

from typesafe_sdk._core.config import Config


def resolve_client_config(
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    timeout: float | httpx2.Timeout | None,
    headers: Mapping[str, str] | None,
    transport: httpx2.BaseTransport | httpx2.AsyncBaseTransport | None,
    http_client: httpx2.Client | httpx2.AsyncClient | None,
) -> Config:
    """Validate client arguments and resolve the underlying client configuration."""
    if transport is not None and http_client is not None:
        raise ValueError("transport and http_client are mutually exclusive.")
    if timeout is None and http_client is not None:
        timeout = http_client.timeout
    return Config.resolve(api_key, base_url, model, timeout, headers)
