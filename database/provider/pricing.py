"""Token prices of the models a provider serves, cached in-process.

OpenRouter publishes its prices only as the whole model catalogue (hundreds of models,
about a megabyte of JSON). Prices change rarely, so the catalogue is fetched at most
once per `PRICING_TTL_SECONDS` and shared by every router and agent run.
"""

import asyncio
import logging
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel, Field, ValidationError

from observability import attributes as attrs
from observability.events import brief, error_type, log_event

logger = logging.getLogger(__name__)

type Rates = tuple[float, float]
"""Prompt and completion prices of a model, in dollars per million tokens."""

UNKNOWN_RATES: Rates = (-1.0, -1.0)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# OpenRouter changes a model's price seldom (days or weeks apart), and a stale price
# only skews the cost written on a trace. An hour keeps the catalogue to one download
# per hour per process and picks up a price change, or a newly listed model, the same day.
PRICING_TTL_SECONDS = 60 * 60
# After a failed fetch, the next one waits this long, so an outage costs one request a
# minute rather than one per agent run, and prices come back a minute after OpenRouter does.
FAILURE_RETRY_SECONDS = 60


class PricingCatalogue:
    """The rates of every model in a provider's catalogue at ``url``."""

    def __init__(
        self,
        url: str,
        *,
        ttl: float = PRICING_TTL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._transport = transport
        self._clock = clock
        self._rates: dict[str, Rates] = {}
        self._expires_at = float("-inf")
        self._refreshing: asyncio.Task[None] | None = None

    async def rates(self, model: str) -> Rates:
        """The rates of ``model``, or `UNKNOWN_RATES` when the catalogue doesn't price it."""
        if self._clock() >= self._expires_at:
            await self._refresh_once()
        return self._rates.get(model, UNKNOWN_RATES)

    async def _refresh_once(self) -> None:
        """Join the refresh in flight, or start one: concurrent lookups fetch once.

        The refresh is a task of its own, shielded so a cancelled lookup doesn't cancel
        it for the others. A task belongs to one event loop, and the catalogue outlives
        loops (``asyncio.run`` in evals and tests), so a new loop starts its own.
        """
        task = self._refreshing
        if (
            task is None
            or task.done()
            or task.get_loop() is not asyncio.get_running_loop()
        ):
            task = self._refreshing = asyncio.create_task(self._refresh())
        await asyncio.shield(task)

    async def _refresh(self) -> None:
        """Replace the rates with a fresh catalogue, or keep the last ones if it fails."""
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                response = (await client.get(self._url, timeout=10)).raise_for_status()
            self._rates = _parse(response.json())
        except (httpx.HTTPError, ValueError) as exc:  # ValidationError, bad JSON
            log_event(
                logger,
                attrs.EVENT_PRICING_FETCH_FAILED,
                {
                    attrs.ERROR_TYPE: error_type(exc),
                    attrs.EXCEPTION_MESSAGE: brief(exc),
                },
                logging.WARNING,
            )
            self._expires_at = self._clock() + FAILURE_RETRY_SECONDS
        else:
            self._expires_at = self._clock() + self._ttl


class _Pricing(BaseModel):
    """A model's price, in dollars per token (OpenRouter sends the numbers as strings)."""

    prompt: float = 0
    completion: float = 0


class _CatalogueModel(BaseModel):
    id: str
    pricing: _Pricing = Field(default_factory=_Pricing)


class _Catalogue(BaseModel):
    data: list[object]


OPENROUTER_PRICING = PricingCatalogue(OPENROUTER_MODELS_URL)
"""The OpenRouter catalogue every OpenRouter router is priced from. The catalogue is
public, so it is fetched without an API key and shared across routers."""


def _parse(body: object) -> dict[str, Rates]:
    """The rates of every well-formed model in ``body``; malformed models are skipped."""
    rates: dict[str, Rates] = {}
    for entry in _Catalogue.model_validate(body).data:
        try:
            model = _CatalogueModel.model_validate(entry)
        except ValidationError:
            continue
        rates[model.id] = (
            model.pricing.prompt * 1_000_000,
            model.pricing.completion * 1_000_000,
        )
    return rates
