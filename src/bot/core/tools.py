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

import logging
import os

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "15"))
# The backend only answers search for its own agent (or a signed-in user):
# each search can spend a shared, rate-limited flight API budget.
_HEADERS = {"X-Service-Token": os.getenv("BACKEND_SERVICE_TOKEN", "")}


async def _post(path: str, payload: dict) -> list[dict]:
    """One non-blocking call to the backend.

    Errors come back as *data*, not exceptions: the model reads the error and
    tells the traveller "I couldn't reach the train service", instead of the
    whole turn crashing and the call going silent.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{BACKEND_URL}{path}", json=payload, headers=_HEADERS
            )
    except httpx.HTTPError as exc:
        # Logged in full here; the model — and so the app's card and the
        # stored history — gets a sentence without our internal address in it.
        logger.warning("search %s failed: %s", path, exc)
        return [{"error": "The search service could not be reached."}]

    if resp.status_code == 422:
        # The model sent something malformed — a bad date, too many guests.
        # Say what, so it can fix the call rather than give up.
        try:
            detail = [
                f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}"
                for e in resp.json().get("detail", [])
            ]
        except (ValueError, AttributeError, TypeError):
            detail = []
        return [{"error": "Invalid search: " + ("; ".join(detail) or "check the inputs")}]
    if resp.is_error:
        logger.warning("search %s returned %s: %s", path, resp.status_code, resp.text[:200])
        return [{"error": "The search service is unavailable right now."}]
    return resp.json()


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
