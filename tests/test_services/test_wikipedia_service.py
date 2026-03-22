"""Tests for app.services.wikipedia_service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.wikipedia_service import SpeciesSummary, fetch_species_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_PAYLOAD = {
    "title": "Rock ptarmigan",
    "extract": "The rock ptarmigan is a medium-sized gamebird.",
    "thumbnail": {"source": "https://upload.wikimedia.org/ptarmigan.jpg"},
}


def _make_resp(status: int, data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _patch_client(resp: MagicMock):
    """Return (patcher, async_client_mock) for httpx.AsyncClient context manager."""
    client_mock = AsyncMock()
    client_mock.get.return_value = resp

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=None)

    patcher = patch(
        "app.services.wikipedia_service.httpx.AsyncClient",
        return_value=cm,
    )
    return patcher, client_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_title_extract_and_thumbnail() -> None:
    resp = _make_resp(200, _FULL_PAYLOAD)
    patcher, _ = _patch_client(resp)
    with patcher:
        result = asyncio.run(fetch_species_summary("Lagopus muta"))

    assert isinstance(result, SpeciesSummary)
    assert result.title == "Rock ptarmigan"
    assert result.extract == "The rock ptarmigan is a medium-sized gamebird."
    assert result.thumbnail_url == "https://upload.wikimedia.org/ptarmigan.jpg"


def test_no_thumbnail_key_returns_none() -> None:
    payload = {"title": "Rock ptarmigan", "extract": "A bird."}
    resp = _make_resp(200, payload)
    patcher, _ = _patch_client(resp)
    with patcher:
        result = asyncio.run(fetch_species_summary("Lagopus muta"))

    assert result.thumbnail_url is None


def test_thumbnail_key_without_source_returns_none() -> None:
    payload = {"title": "Rock ptarmigan", "extract": "A bird.", "thumbnail": {}}
    resp = _make_resp(200, payload)
    patcher, _ = _patch_client(resp)
    with patcher:
        result = asyncio.run(fetch_species_summary("Lagopus muta"))

    assert result.thumbnail_url is None


def test_404_raises_http_status_error() -> None:
    resp = _make_resp(404, {})
    patcher, _ = _patch_client(resp)
    with patcher, pytest.raises(httpx.HTTPStatusError):
        asyncio.run(fetch_species_summary("Unknown species xyz"))


def test_network_error_raises_request_error() -> None:
    client_mock = AsyncMock()
    client_mock.get.side_effect = httpx.RequestError("timeout")

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("app.services.wikipedia_service.httpx.AsyncClient", return_value=cm),
        pytest.raises(httpx.RequestError),
    ):
        asyncio.run(fetch_species_summary("Lagopus muta"))


def test_spaces_converted_to_underscores() -> None:
    resp = _make_resp(200, _FULL_PAYLOAD)
    patcher, client_mock = _patch_client(resp)
    with patcher:
        asyncio.run(fetch_species_summary("Lagopus muta"))

    called_url: str = client_mock.get.call_args.args[0]
    assert "Lagopus_muta" in called_url
    assert "Lagopus muta" not in called_url


def test_request_includes_user_agent_header() -> None:
    resp = _make_resp(200, _FULL_PAYLOAD)
    patcher, client_mock = _patch_client(resp)
    with patcher:
        asyncio.run(fetch_species_summary("Lagopus muta"))

    kwargs = client_mock.get.call_args.kwargs
    headers = kwargs.get("headers", {})
    assert "User-Agent" in headers
    assert headers["User-Agent"]  # non-empty
