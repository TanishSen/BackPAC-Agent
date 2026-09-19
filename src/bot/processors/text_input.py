"""Typed input: start a turn without speaking.

Two things need this:

- **The keyboard.** Someone on a noisy train still wants to use the assistant.
- **STT_MODE=device**, where the phone transcribes locally and posts the words,
  so no server STT is built at all (see pipeline.py).

Both send the same packet over the LiveKit data channel, on topic `user-text`:

    {"type": "user-text", "text": "trains to jaipur on friday"}

and both land where a spoken turn lands, so the brain, the transcript and the
cards behave identically either way.

**Why a pipeline stage rather than a transport event handler.** Pipecat's
LiveKit transport does *not* call an `on_data_received` handler for inbound
packets — it turns each one into a transport-message frame and pushes it down
the pipeline. So a handler registered on the transport is simply never invoked
(it fails silently, which is the worst kind). Reading the frame is the real
contract.

That same frame type also carries Pipecat's own **outbound** RTVI messages on
their way to the transport, so this matches on the payload rather than the
class, and always forwards the frame on. Swallowing it would stop the app ever
receiving a result card.
"""

import json
import logging

from pipecat.frames.frames import Frame, OutputTransportMessageUrgentFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

TOPIC = "user-text"
MESSAGE_TYPE = "user-text"


class TextInputProcessor(FrameProcessor):
    """Turns a `user-text` data packet into a normal turn on the brain."""

    def __init__(self, brain):
        super().__init__()
        self._brain = brain

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Forward first, always: this frame class is shared with outbound
        # traffic, and anything held here is a card the app never sees.
        await self.push_frame(frame, direction)

        if not isinstance(frame, OutputTransportMessageUrgentFrame):
            return
        text = self._typed_text(frame.message)
        if not text:
            return

        logger.info("typed turn: %s", text)
        await self._brain.inject_user_text(text)

    @staticmethod
    def _typed_text(message) -> str:
        """The text of a `user-text` packet, or "" if this isn't one.

        Anything malformed is ignored rather than raised: one bad packet from a
        client must never end a call in progress.
        """
        if isinstance(message, (bytes, bytearray)):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                return ""
        if not isinstance(message, dict) or message.get("type") != MESSAGE_TYPE:
            return ""
        return (message.get("text") or "").strip()
