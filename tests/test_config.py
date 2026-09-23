import traceback

import httpx2
import pytest
from pydantic_core import from_json

from tests.conftest import ClientFactory
from tests.helpers import models, system_one
from tests.test_clients import RESULT
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient, TypeSafeError


async def test_transport_and_http_client_mutually_exclusive(clients: ClientFactory) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("Unexpected request")

    transport = httpx2.MockTransport(handler)
    http_client = httpx2.AsyncClient(transport=transport) if clients.async_mode else httpx2.Client(transport=transport)
    try:
        with pytest.raises(ValueError, match="transport and http_client are mutually exclusive"):
            clients(handler, transport=transport, http_client=http_client)
        assert not http_client.is_closed
    finally:
        if isinstance(http_client, httpx2.AsyncClient):
            await http_client.aclose()
        else:
            http_client.close()


@pytest.mark.parametrize("model", [None, "request-model"])
async def test_model_override(clients: ClientFactory, model: str | None) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert from_json(request.content)["model"] == (model or "client-model")
        return httpx2.Response(200, json=RESULT)

    await system_one(
        clients(handler, model="client-model"),
        state="hello",
        questions={"q": {"type": "noul", "instructions": "?"}},
        model=model,
    )


@pytest.mark.parametrize("source", ["default", "env", "constructor"])
async def test_resolution(clients: ClientFactory, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    if source != "default":
        monkeypatch.setenv("TYPESAFE_API_KEY", "  env-key  ")
        monkeypatch.setenv("TYPESAFE_BASE_URL", "  https://env.test///  ")
        monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "  env-model  ")
    expected_key = "code-key" if source == "constructor" else "env-key" if source == "env" else "test-key"
    expected_url = "https://code.test" if source == "constructor" else "https://env.test" if source == "env" else "https://api.typesafe.ai"
    expected_model = "code-model" if source == "constructor" else "env-model" if source == "env" else "jev-latest"

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        assert str(request.url) == expected_url + "/v1/systemone"
        assert from_json(request.content)["model"] == expected_model
        assert request.extensions["timeout"] == {"connect": 10.0, "read": 10.0, "write": 10.0, "pool": 10.0}
        return httpx2.Response(200, json=RESULT)

    if source == "constructor":
        client = clients(handler, api_key="code-key", base_url="https://code.test///", model="code-model")
    elif source == "env":
        client = clients(handler, api_key=None)
    else:
        client = clients(handler)
    await system_one(client, state="hello", questions={"q": {"type": "noul", "instructions": "?"}})


@pytest.mark.parametrize("client_type", [TypeSafeClient, AsyncTypeSafeClient])
@pytest.mark.parametrize("key", [None, "", " \t\n "])
def test_missing_key(client_type: type[TypeSafeClient] | type[AsyncTypeSafeClient], monkeypatch: pytest.MonkeyPatch, key: str | None) -> None:
    if key is not None:
        monkeypatch.setenv("TYPESAFE_API_KEY", key)
    with pytest.raises(TypeSafeError, match="TYPESAFE_API_KEY"):
        client_type()


@pytest.mark.parametrize("source", ["env", "constructor"])
@pytest.mark.parametrize("padding", ["", "\n", "\r\n", " \t\r\n "])
async def test_api_key_whitespace(clients: ClientFactory, monkeypatch: pytest.MonkeyPatch, source: str, padding: str) -> None:
    key = f"{padding}test-key{padding}"
    monkeypatch.setenv("TYPESAFE_API_KEY", key if source == "env" else "env-key")

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx2.Response(200, json={"models": []})

    assert await models(clients(handler, api_key=None if source == "env" else key)) == ()


@pytest.mark.parametrize("key", ["", " \t\r\n ", "\x00private", "private\x00"])
async def test_invalid_explicit_key_does_not_fall_back_to_env(clients: ClientFactory, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    with pytest.raises(TypeSafeError, match="API key"):
        clients(lambda request: pytest.fail("Unexpected request"), api_key=key)


@pytest.mark.parametrize("source", ["env", "constructor"])
@pytest.mark.parametrize("character", ["\n", "\r", "\t", "\x1f", "\x7f", " ", "\u00e9", "\u200b"])
async def test_invalid_api_key(clients: ClientFactory, monkeypatch: pytest.MonkeyPatch, source: str, character: str) -> None:
    credential = "ts_live_private"
    key = f"{credential}{character}suffix"
    monkeypatch.setenv("TYPESAFE_API_KEY", key if source == "env" else "env-key")
    with pytest.raises(TypeSafeError, match="API key") as caught:
        clients(lambda request: pytest.fail("Unexpected request"), api_key=None if source == "env" else key)
    error = caught.value
    assert credential not in str(error)
    assert credential not in repr(error)
    assert credential not in "".join(traceback.format_exception(type(error), error, error.__traceback__))


async def test_empty_env_unset(clients: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_LOG_LEVEL"):
        monkeypatch.setenv(name, " \t ")

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert from_json(request.content)["model"] == "jev-latest"
        return httpx2.Response(200, json=RESULT)

    await system_one(clients(handler), state="x", questions={"q": {"type": "noul", "instructions": "?"}})


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
async def test_invalid_timeout(clients: ClientFactory, value: float) -> None:
    with pytest.raises(TypeSafeError, match="timeout"):
        clients(lambda request: httpx2.Response(200), timeout=value)
    client = clients(lambda request: pytest.fail("Unexpected request"))
    with pytest.raises(TypeSafeError, match="timeout"):
        await models(client, timeout=value)


async def test_timeout_object(clients: ClientFactory) -> None:
    timeout = httpx2.Timeout(7.0, connect=1.0)

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.extensions["timeout"] == {"connect": 1.0, "read": 7.0, "write": 7.0, "pool": 7.0}
        return httpx2.Response(200, json={"models": []})

    assert await models(clients(handler, timeout=timeout)) == ()


@pytest.mark.parametrize("http_timeout", [httpx2.Timeout(17.0, connect=1.0), httpx2.Timeout(None)])
@pytest.mark.parametrize("sdk_timeout,call_timeout", [(None, None), (3.0, None), (None, 2.0), (3.0, httpx2.Timeout(None))])
async def test_http_client_timeout_precedence(
    clients: ClientFactory,
    http_timeout: httpx2.Timeout,
    sdk_timeout: float | None,
    call_timeout: float | httpx2.Timeout | None,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"models": []})

    http_client = (
        httpx2.AsyncClient(transport=httpx2.MockTransport(handler), timeout=http_timeout)
        if clients.async_mode
        else httpx2.Client(transport=httpx2.MockTransport(handler), timeout=http_timeout)
    )
    client = clients(handler, http_client=http_client, timeout=sdk_timeout)
    assert await models(client, timeout=call_timeout) == ()
    assert await models(client) == ()
    default = http_timeout if sdk_timeout is None else httpx2.Timeout(sdk_timeout)
    expected = default if call_timeout is None else httpx2.Timeout(call_timeout)
    assert requests[0].extensions["timeout"] == expected.as_dict()
    assert requests[1].extensions["timeout"] == default.as_dict()
    assert http_client.timeout == http_timeout


def test_config_resolve_accepts_custom_environ() -> None:
    from typesafe_sdk._core.config import Config

    custom_env = {
        "TYPESAFE_API_KEY": "env-custom-key",
        "TYPESAFE_BASE_URL": "https://custom.api.test",
        "TYPESAFE_DEFAULT_MODEL": "custom-model",
    }
    cfg = Config.resolve(None, None, None, None, None, environ=custom_env)
    assert cfg.api_key == "env-custom-key"
    assert cfg.base_url == "https://custom.api.test"
    assert cfg.default_model == "custom-model"

    # Explicit arguments override custom environ
    cfg_override = Config.resolve("explicit-key", "https://override.test", "override-model", None, None, environ=custom_env)
    assert cfg_override.api_key == "explicit-key"
    assert cfg_override.base_url == "https://override.test"
    assert cfg_override.default_model == "override-model"


def test_resolve_client_config_shared_validation() -> None:
    from typesafe_sdk._core.client import resolve_client_config

    transport = httpx2.MockTransport(lambda request: httpx2.Response(200))
    http_client = httpx2.Client(transport=transport, timeout=7.5)
    try:
        with pytest.raises(ValueError, match="transport and http_client are mutually exclusive"):
            resolve_client_config("key", None, None, None, None, transport=transport, http_client=http_client)

        cfg = resolve_client_config("key", None, None, None, None, transport=None, http_client=http_client)
        assert cfg.timeout == httpx2.Timeout(7.5)
    finally:
        http_client.close()
