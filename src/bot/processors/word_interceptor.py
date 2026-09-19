"""Publishes what the agent says to the app, over the LiveKit data channel.

It sits **after** the TTS text filter and before TTS itself, so the app is shown
exactly the words that get spoken — markdown, SSML and ElevenLabs audio tags
already stripped. Publishing earlier would put asterisks and stage directions
like "[warmly]" on screen that nobody ever hears.

Wire format (topic `"transcription"`):

    streaming word   {"role": "agent", "text": "word", "speech_final": false}
    end of the turn  {"role": "agent", "text": "",     "speech_final": true }
    user transcript  {"role": "user",  "text": "...",  "speech_final": true }

The app accumulates agent fragments into one message and commits it when
`speech_final` arrives. Anything the agent says that is *not* part of a
generated turn — the greeting, the idle nudges — arrives as a `TTSSpeakFrame`
rather than a stream of `TextFrame`s, so it is published as that same pair in
one go. Without that, the caller hears "Hi! Where would you like to travel?"
and the chat shows nothing.
"""

import json

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class WordInterceptor(FrameProcessor):
    def __init__(self, transport):
        super().__init__()
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            # TTSSpeakFrame is checked first: it is a DataFrame, not a
            # TextFrame, so the ordering here is about clarity, not dispatch.
            if isinstance(frame, TTSSpeakFrame):
                # A complete line the agent was told to say, not a generated
                # turn. Send it as a fragment plus a close, so the app's
                # accumulate-then-commit logic needs no special case.
                await self._publish_agent(frame.text, speech_final=False)
                await self._publish_agent("", speech_final=True)

            elif isinstance(frame, TextFrame):
                await self._publish_agent(frame.text, speech_final=False)

            elif isinstance(frame, LLMFullResponseEndFrame):
                # Tell the app to commit the text it has accumulated.
                await self._publish_agent("", speech_final=True)

        await self.push_frame(frame, direction)

    async def publish_user_transcript(self, text: str) -> None:
        """Called by the brain when a user turn is recognised — spoken or typed."""
        await self._publish({"role": "user", "text": text, "speech_final": True})

    async def _publish_agent(self, text: str, *, speech_final: bool) -> None:
        await self._publish(
            {"role": "agent", "text": text, "speech_final": speech_final}
        )

    async def _publish(self, data: dict) -> None:
        try:
            # `.room` raises if the transport is not connected — which happens
            # normally at the very end of a call, so this must never propagate.
            room = self._transport._client.room
            await room.local_participant.publish_data(
                json.dumps(data).encode("utf-8"),
                reliable=True,
                topic="transcription",
            )
        except Exception as exc:  # noqa: BLE001 — a dropped caption is not a dropped call
            logger.error(f"WordInterceptor: failed to publish to LiveKit: {exc}")
