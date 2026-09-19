"""The tools the specialist agents call.

Every tool here is a thin wrapper over a BackPAC-BE endpoint. The agent decides
*when* to search; the backend decides *how* (which provider, what pricing). That
split matters: swapping the mock train data for real IRCTC data is a backend
change and the agent never notices.

`@tool` (from LangChain) turns a plain function into something Claude can call.
The docstring is not decoration — Claude reads it to decide when to call the
tool and what to pass, so keep it accurate.

**These are async on purpose.** The agent runs inside a realtime voice pipeline
whose event loop is already busy with 50 audio frames a second plus voice-
activity and end-of-turn inference. A synchronous tool would be handed to
LangGraph's thread pool, where it competes with that work and can sit for tens
of seconds before it runs — long enough that the caller thinks the line went
dead. Async tools do their waiting as ordinary non-blocking I/O instead.
"""

import os

import httpx
from langchain_core.tools import tool

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "15"))


async def _post(path: str, payload: dict) -> list[dict]:
    """One non-blocking call to the backend.

    Errors come back as *data*, not exceptions: the model reads the error and
    tells the traveller "I couldn't reach the train service", instead of the
    whole turn crashing and the call going silent.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(f"{BACKEND_URL}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        return [{"error": f"search failed: {exc}"}]


@tool
async def search_trains(
    origin: str, destination: str, depart_date: str, passengers: int = 1
) -> list[dict]:
    """Find trains between two cities on a date. `depart_date` is YYYY-MM-DD.
    Use when the traveller wants to go by train."""
    return await _post(
        "/api/v1/trips/search/trains",
        {
            "origin": origin,
            "destination": destination,
            "departDate": depart_date,
            "passengers": passengers,
        },
    )


@tool
async def search_flights(
    origin: str, destination: str, depart_date: str, passengers: int = 1
) -> list[dict]:
    """Find flights between two cities on a date. `depart_date` is YYYY-MM-DD.
    Use when the traveller wants to fly or when speed matters."""
    return await _post(
        "/api/v1/trips/search/flights",
        {
            "origin": origin,
            "destination": destination,
            "departDate": depart_date,
            "passengers": passengers,
        },
    )


@tool
async def search_stays(
    destination: str, check_in: str, check_out: str, guests: int = 2
) -> list[dict]:
    """Find places to stay in a city between two dates (YYYY-MM-DD). Use when the
    traveller needs a hotel or somewhere to stay."""
    return await _post(
        "/api/v1/trips/search/stays",
        {
            "destination": destination,
            "checkIn": check_in,
            "checkOut": check_out,
            "guests": guests,
        },
    )
