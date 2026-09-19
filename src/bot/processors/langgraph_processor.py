"""The bridge between Pipecat (voice) and LangGraph (the brain).

Pipecat pushes a `TranscriptionFrame` when the user finishes speaking; this
processor feeds that text into the compiled graph, streams the reply back out as
`TextFrame`s (which the TTS stage downstream speaks), and remembers the
conversation via the graph's checkpointer, keyed by the room.

This is deliberately the *thin* version of ajimganj's UnifiedLanggraphProcessor
— it does the core job (frame in -> graph -> frames out) without the RTVI
data-channel UI signalling ajimganj layers on. Add that when the app needs the
agent to drive on-screen cards; see AGENT_GUIDE.md.
"""

import logging

from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class LangGraphProcessor(FrameProcessor):
    def __init__(self, graph, room_name: str):
        super().__init__()
        self._graph = graph
        self._room_name = room_name
        # The checkpointer keys memory by thread_id — one thread per room means
        # this call's history is isolated from every other call's.
        self._config = {"configurable": {"thread_id": room_name}}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only act on a finished user utterance. Everything else flows through.
        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        user_text = frame.text.strip()
        if not user_text:
            return

        logger.info("[%s] user: %s", self._room_name, user_text)

        # Run one turn through the graph. `astream` lets us speak the reply as it
        # is produced rather than waiting for the whole thing.
        state_in = {
            "messages": [{"role": "user", "content": user_text}],
            "room_name": self._room_name,
        }
        async for chunk in self._graph.astream(
            state_in, self._config, stream_mode="values"
        ):
            messages = chunk.get("messages", [])
            if not messages:
                continue
            last = messages[-1]
            # Only speak assistant text, and only the final assistant message of
            # the turn (tool calls and user echoes are skipped).
            content = getattr(last, "content", "")
            role = getattr(last, "type", "")
            if role == "ai" and isinstance(content, str) and content.strip():
                await self.push_frame(TextFrame(content), FrameDirection.DOWNSTREAM)
