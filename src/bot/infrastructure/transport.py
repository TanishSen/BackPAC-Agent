"""LiveKit transport: the agent's connection into the room.

Builds a `LiveKitTransport` bound to `room_name`, minting the agent's own join
token from the same LiveKit project the backend used. Also sets up an optional
ai-coustics noise filter (only if AIC_SDK_LICENSE is set) and its VAD.

Trimmed from the ajimganj concierge to LiveKit-only (the telephony/websocket
branch is dropped). The one non-obvious bit — the data-packet monkey-patch — is
kept verbatim because it prevents a crash on LiveKit's room-broadcast packets,
which arrive with no participant.
"""

from __future__ import annotations  # lazy annotations (AICFilter may be None)

import os
from typing import Optional, Tuple

from loguru import logger
try:
    from pipecat.audio.filters.aic_filter import AICFilter
except Exception:  # noqa: BLE001 — optional proprietary dep
    AICFilter = None
from pipecat.runner.livekit import generate_token_with_agent
from pipecat.transports.livekit.transport import (
    LiveKitParams,
    LiveKitTransport,
    LiveKitTransportClient,
)

from config.settings import PARTICIPANT_NAME

# --- Patch: tolerate room-broadcast data packets (participant is None) --------
# Without this, a server/room broadcast packet raises inside the client. Kept
# exactly as proven in the ajimganj concierge.
_orig_on_data = LiveKitTransportClient._async_on_data_received


async def _patched_on_data(self, data):
    if not data.participant:
        await self._callbacks.on_data_received(data.data, None)
        return
    await _orig_on_data(self, data)


LiveKitTransportClient._async_on_data_received = _patched_on_data


def create_transport(room_name: str) -> Tuple[LiveKitTransport, Optional[AICFilter]]:
    """Return (transport, aic_filter). `aic_filter` is None unless a license is set."""
    aic_filter = None
    license_key = os.getenv("AIC_SDK_LICENSE")
    if license_key and AICFilter is not None:
        try:
            aic_filter = AICFilter(
                license_key=license_key,
                model_id="quail-vf-2.1-l-16khz",
                enhancement_level=float(os.getenv("AIC_ENHANCEMENT_LEVEL", "1.0")),
            )
            logger.info("AIC noise filter active")
        except Exception as exc:  # noqa: BLE001 — noise filter is optional
            logger.error(f"AIC filter init failed, continuing without it: {exc}")

    transport = LiveKitTransport(
        # LIVEKIT_AGENT_URL lets this process reach LiveKit by a different door
        # than the app does, and it is worth knowing why.
        #
        # LiveKit Cloud's project URL is meant to route you to a nearby edge.
        # From some networks it does not: the connection to the project URL
        # simply hangs, times out after five seconds, falls back to one region,
        # hangs again, and only answers on the second fallback. Measured here,
        # repeatedly: ~12s via the project URL, ~1.1s naming a working region
        # directly. That twelve seconds is the caller sitting in an empty room
        # waiting for the assistant to arrive.
        #
        # The app keeps the project URL, so phones still get routed to whatever
        # edge is nearest them. Only this server-side process is pinned, and
        # unsetting the variable puts it straight back to the default.
        url=os.getenv("LIVEKIT_AGENT_URL") or os.getenv("LIVEKIT_URL"),
        token=generate_token_with_agent(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            participant_name=PARTICIPANT_NAME,
            room_name=room_name,
        ),
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter=aic_filter,
        ),
    )
    return transport, aic_filter
