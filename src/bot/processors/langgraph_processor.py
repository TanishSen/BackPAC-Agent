"""The bridge between the voice pipeline (Pipecat) and the brain (LangGraph).

This is the heart of the conversational loop, and it sits where an LLM service
would normally go — the trip graph *is* the LLM as far as the pipeline is
concerned.

**What starts a turn.** Not a raw `TranscriptionFrame`. In Pipecat 1.3 the user
aggregator upstream of us swallows transcriptions, batches them, waits for
voice-activity and turn analysis to agree the person has finished, and only then
emits one `LLMContextFrame` carrying the whole turn. Listening for
`TranscriptionFrame` here would mean never hearing anything at all, and would
also throw away barge-in and end-of-turn detection. So `LLMContextFrame` is the
trigger, and we read the newest user message off it.

**What a turn does.**
  1. broadcast the user's text to the app (so the chat transcript shows it),
  2. stream that text through the trip graph,
  3. push the assistant's reply **token by token** as `TextFrame`s, so TTS starts
     speaking the first words while the model is still writing the rest — that
     streaming is what makes the agent feel responsive rather than laggy,
  4. emit each real tool result (a train/flight/stay list) as a card the app can
     render, ignoring the internal `transfer_to_*` routing calls.

Ported from the ajimganj concierge's UnifiedLanggraphProcessor and adapted to
the BackPAC trip graph.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from pipecat.frames.frames import (
    DataFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class ToolResultFrame(DataFrame):
    """A tool's output travelling downstream — e.g. a list of trains. The card
    dispatcher turns this into a UI card for the app."""

    card_type: str  # the tool name, e.g. "search_trains"
    result: Any
    type: str = "tool_result"


class LangGraphProcessor(FrameProcessor):
    def __init__(self, graph, *, room_name: str, word_interceptor=None):
        super().__init__()
        self._graph = graph
        self._room_name = room_name
        self._word_interceptor = word_interceptor
        # The graph's checkpointer keys memory by thread_id. Using the room name
        # (stable across reconnects) means a dropped-and-rejoined call resumes
        # the same conversation instead of starting over.
        self._thread_id = room_name

    def set_participant_id(self, participant_id: str) -> None:
        """Called from the transport's on_first_participant_joined. We prefer the
        room name (stable), so this only fills in if we somehow have none."""
        self._thread_id = self._thread_id or participant_id

    # --- frame handling ------------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            text = self._latest_user_text(frame)
            if text:
                if self._word_interceptor:
                    await self._word_interceptor.publish_user_transcript(text)
                await self._run_turn(text)
            # Deliberately not forwarded: downstream is TTS, which has no use for
            # a context frame, and we are the thing that would have consumed it.
            return

        await self.push_frame(frame, direction)

    @staticmethod
    def _latest_user_text(frame: LLMContextFrame) -> str:
        """The newest user message on the context, as plain text.

        Content can be a string or a list of parts (Pipecat's universal context
        format), so both shapes are handled.
        """
        messages = frame.context.get_messages() if frame.context else []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                ).strip()
            return ""
        return ""

    @staticmethod
    def _card_payload(output: Any) -> Any:
        """Turn a tool's raw output into the plain JSON the app renders.

        LangGraph hands back a `ToolMessage`, not the list the tool returned:
        the rows are a JSON *string* on its `.content`. Unwrapping here means
        the app receives `[{"provider": "IRCTC", ...}, ...]` and can build a
        card straight from it, instead of every client having to know LangChain's
        message shape and double-decode.
        """
        content = getattr(output, "content", output)
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (ValueError, TypeError):
                return content  # not JSON — hand it over as-is
        return content

    @staticmethod
    def _token_text(chunk: Any) -> str:
        """Pull the text out of a streamed model chunk, whatever its shape."""
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        # Some providers stream a list of content blocks.
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return ""

    async def inject_user_text(self, text: str) -> None:
        """Run a turn from typed text instead of speech.

        Two callers: the keyboard in the app, and STT_MODE=device where the
        phone transcribes and posts the words itself. Both arrive over the
        LiveKit data channel — see infrastructure/text_input.py. From here on
        the turn is identical to a spoken one, which is the point: there is one
        conversation, not a voice one and a typed one.
        """
        text = text.strip()
        if not text:
            return
        if self._word_interceptor:
            await self._word_interceptor.publish_user_transcript(text)
        await self._run_turn(text)

    # --- one user turn -------------------------------------------------------
    async def _run_turn(self, text: str) -> None:
        """One user turn → a streamed spoken reply."""
        logger.info("[%s] user: %s", self._room_name, text)
        await self.push_frame(LLMFullResponseStartFrame())

        input_state = {
            "messages": [HumanMessage(content=text)],
            "active_agent": "orchestrator",
            "room_name": self._room_name,
        }
        config = {"configurable": {"thread_id": self._thread_id}}

        spoke = False
        try:
            # astream_events(v2) gives fine-grained events: per-token model
            # streams, and tool start/end — everything we need to be responsive.
            async for event in self._graph.astream_events(
                input_state, config=config, version="v2"
            ):
                kind = event["event"]
                if kind == "on_chat_model_start" and spoke:
                    # A second model response in the same turn — typically a
                    # short preamble, then the real answer once the tool
                    # returned. Without a separator they run together
                    # ("…searching now!Here are two options").
                    await self.push_frame(TextFrame(" "))
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # Skip tool-call argument chunks — they aren't speech.
                    if getattr(chunk, "tool_call_chunks", None):
                        continue
                    token = self._token_text(chunk)
                    if token:
                        spoke = True
                        await self.push_frame(TextFrame(token))
                elif kind == "on_tool_end":
                    name = event.get("name", "")
                    # Routing handoffs are internal; only real searches are cards.
                    if not name.startswith("transfer_to_"):
                        await self.push_frame(
                            ToolResultFrame(
                                card_type=name,
                                result=self._card_payload(event["data"].get("output")),
                            )
                        )
        except Exception:  # noqa: BLE001 — never let one bad turn kill the call
            logger.exception("[%s] turn failed", self._room_name)

        if not spoke:
            # Silence after someone speaks reads as "it's broken". Say something
            # rather than leave the line dead.
            logger.warning("[%s] turn produced no speech", self._room_name)
            await self.push_frame(
                TextFrame("Sorry, I didn't catch that — could you say it again?")
            )

        # Always close the response, even on error, so the app commits the
        # transcript line and TTS flushes.
        await self.push_frame(LLMFullResponseEndFrame())
