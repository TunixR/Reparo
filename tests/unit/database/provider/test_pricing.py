"""The OpenRouter pricing catalogue: fetched once per TTL, off the event loop."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from database.provider.pricing import (
    PRICING_TTL_SECONDS,
    UNKNOWN_RATES,
    PricingCatalogue,
)

URL = "https://openrouter.test/api/v1/models"

CATALOGUE = {
    "data": [
        {
            "id": "vendor/model",
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        },
        {
            "id": "vendor/other",
            "pricing": {"prompt": "0.000001", "completion": "0.000003"},
        },
    ]
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class Catalogue:
    """A catalogue served by a mocked HTTP layer that counts its requests."""

    def __init__(self) -> None:
        self.requests = 0
        self.gate: asyncio.Event | None = None
        self.fetching = asyncio.Event()
        self.clock = FakeClock()
        self.respond = lambda: httpx.Response(200, json=CATALOGUE)
        self.pricing = PricingCatalogue(
            URL,
            transport=httpx.MockTransport(self._handle),
            clock=self.clock,
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == URL
        self.requests += 1
        self.fetching.set()
        if self.gate is not None:
            _ = await self.gate.wait()
        return self.respond()


@pytest.fixture
def catalogue() -> Catalogue:
    return Catalogue()


@pytest.mark.asyncio
async def test_lookups_within_the_ttl_fetch_the_catalogue_once(catalogue: Catalogue):
    rates = [await catalogue.pricing.rates("vendor/model") for _ in range(5)]
    other = await catalogue.pricing.rates("vendor/other")

    assert catalogue.requests == 1
    assert rates == [(2.0, 10.0)] * 5
    assert other == (1.0, 3.0)
    assert await catalogue.pricing.rates("vendor/missing") == UNKNOWN_RATES


@pytest.mark.asyncio
async def test_a_lookup_after_the_ttl_refetches_the_catalogue(catalogue: Catalogue):
    _ = await catalogue.pricing.rates("vendor/model")
    catalogue.clock.now += PRICING_TTL_SECONDS - 1
    _ = await catalogue.pricing.rates("vendor/model")
    assert catalogue.requests == 1

    catalogue.respond = lambda: httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "vendor/model",
                    "pricing": {"prompt": "0.000004", "completion": "0"},
                }
            ]
        },
    )
    catalogue.clock.now += 1

    assert await catalogue.pricing.rates("vendor/model") == (4.0, 0.0)
    assert catalogue.requests == 2


def _unreachable() -> httpx.Response:
    raise httpx.ConnectTimeout("OpenRouter is down")


FAILURES = {
    "server error": lambda: httpx.Response(503),
    "unreachable": _unreachable,
    "not json": lambda: httpx.Response(200, text="<html>maintenance</html>"),
    "unexpected shape": lambda: httpx.Response(200, json={"models": []}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
async def test_a_failed_fetch_is_unknown_and_retried_after_a_short_backoff(
    catalogue: Catalogue, failure: Callable[[], httpx.Response]
):
    catalogue.respond = failure

    assert await catalogue.pricing.rates("vendor/model") == UNKNOWN_RATES
    assert await catalogue.pricing.rates("vendor/model") == UNKNOWN_RATES
    assert catalogue.requests == 1

    catalogue.respond = lambda: httpx.Response(200, json=CATALOGUE)
    catalogue.clock.now += 59
    assert await catalogue.pricing.rates("vendor/model") == UNKNOWN_RATES
    assert catalogue.requests == 1

    catalogue.clock.now += 1

    assert await catalogue.pricing.rates("vendor/model") == (2.0, 10.0)
    assert catalogue.requests == 2


@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_last_prices(catalogue: Catalogue):
    _ = await catalogue.pricing.rates("vendor/model")
    catalogue.respond = _unreachable
    catalogue.clock.now += PRICING_TTL_SECONDS

    assert await catalogue.pricing.rates("vendor/model") == (2.0, 10.0)
    assert await catalogue.pricing.rates("vendor/model") == (2.0, 10.0)
    assert catalogue.requests == 2


@pytest.mark.asyncio
async def test_a_malformed_entry_is_unknown_and_spares_the_rest(catalogue: Catalogue):
    catalogue.respond = lambda: httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "vendor/bad",
                    "pricing": {"prompt": "varies", "completion": "0"},
                },
                {"id": "vendor/no-pricing", "pricing": None},
                {"id": "vendor/unpriced"},
                {"pricing": {"prompt": "0.1", "completion": "0.1"}},
                *CATALOGUE["data"],
            ]
        },
    )

    assert await catalogue.pricing.rates("vendor/bad") == UNKNOWN_RATES
    assert await catalogue.pricing.rates("vendor/no-pricing") == UNKNOWN_RATES
    assert await catalogue.pricing.rates("vendor/unpriced") == UNKNOWN_RATES
    assert await catalogue.pricing.rates("vendor/model") == (2.0, 10.0)
    assert catalogue.requests == 1


@pytest.mark.asyncio
async def test_concurrent_lookups_share_one_fetch(catalogue: Catalogue):
    catalogue.gate = asyncio.Event()
    lookups = [
        asyncio.create_task(catalogue.pricing.rates(model))
        for model in ["vendor/model", "vendor/other"] * 5
    ]
    _ = await catalogue.fetching.wait()
    for _ in range(10):  # give every lookup the chance to start a fetch of its own
        await asyncio.sleep(0)

    catalogue.gate.set()

    assert await asyncio.gather(*lookups) == [(2.0, 10.0), (1.0, 3.0)] * 5
    assert catalogue.requests == 1


@pytest.mark.asyncio
async def test_a_cancelled_lookup_leaves_the_shared_fetch_running(catalogue: Catalogue):
    catalogue.gate = asyncio.Event()
    cancelled = asyncio.create_task(catalogue.pricing.rates("vendor/model"))
    waiting = asyncio.create_task(catalogue.pricing.rates("vendor/model"))
    _ = await catalogue.fetching.wait()

    _ = cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    catalogue.gate.set()

    assert await waiting == (2.0, 10.0)
    assert catalogue.requests == 1
