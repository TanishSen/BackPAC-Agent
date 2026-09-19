"""Transport event handlers: what happens when someone joins, and idle nudges.

Three things live here, and two of them exist to stop a call outliving the
person on it — an abandoned session holds an Azure STT connection, a LiveKit
room and an agent slot open indefinitely:

- `setup_event_handlers` — on join, tell the brain the stable conversation id
  (the room name, so a reconnect resumes the same chat) and speak the greeting.
  On the last participant leaving, shut the pipeline down.
- `setup_user_aggregator_handlers` — if the caller goes quiet but stays
  connected, nudge them a couple of times, then politely hang up.

Trimmed from the ajimganj concierge: the multi-language translation path is
dropped (BackPAC is English-first; add it back the same way if you need it).
"""

import logging

from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

from config.settings import GREETING_MESSAGE

logger = logging.getLogger(__name__)


def setup_user_aggregator_handlers(user_aggregator) -> None:
    """Escalating nudges when the caller stops talking, then hang up.

    `idle_count` resets on every turn the user actually takes, so the nudges
    only escalate within one continuous silence.
    """
    state = {"idle_count": 0}

    @user_aggregator.event_handler("on_user_turn_started")
    async def _on_turn_started(*_args, **_kwargs):
        state["idle_count"] = 0

    @user_aggregator.event_handler("on_user_turn_idle")
    async def _on_turn_idle(*_args, **_kwargs):
        state["idle_count"] += 1
        if state["idle_count"] == 1:
            await user_aggregator.push_frame(TTSSpeakFrame("Are you still there?"))
        elif state["idle_count"] == 2:
            await user_aggregator.push_frame(
                TTSSpeakFrame("Would you like to keep planning?")
            )
        else:
            await user_aggregator.push_frame(
                TTSSpeakFrame("No problem — I'll be here when you're ready. Bye for now!")
            )
            # Upstream, so the worker (not a downstream processor) sees it and
            # shuts the pipeline down cleanly.
            await user_aggregator.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


def setup_event_handlers(
    transport, worker, langgraph_processor=None, room_name: str | None = None
) -> None:
    """Wire the greeting and the stable-thread-id handoff on join.

    The greeting is queued on the **worker**, not pushed on the transport: a
    transport is an endpoint, not a frame processor, and has no `push_frame`.
    `worker.queue_frame` injects at the head of the pipeline, so the frame runs
    the normal path (TTS text filter -> ElevenLabs -> jitter buffer -> the room)
    and the assistant aggregator records that the bot spoke.
    """

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(transport, participant_id):
        # Use the room name as the conversation id: it survives reconnects,
        # whereas participant_id changes on every new connection.
        thread_id = room_name or participant_id
        if langgraph_processor and hasattr(langgraph_processor, "set_participant_id"):
            langgraph_processor.set_participant_id(thread_id)

        logger.info("participant joined room %s — greeting", room_name)
        await worker.queue_frame(TTSSpeakFrame(GREETING_MESSAGE))

    @transport.event_handler("on_participant_disconnected")
    async def _on_leave(transport, participant_id, *_args):
        """End the call once the last human has gone.

        Without this the pipeline keeps running against an empty room until the
        worker's idle timeout (15 minutes), holding an Azure STT connection and
        an agent slot for a conversation nobody is having. A reconnect gets a
        fresh session and, because the thread id is the room name, resumes the
        same conversation anyway.
        """
        remaining = transport.get_participants()
        if remaining:
            return
        logger.info("room %s is empty — ending the session", room_name)
        await worker.cancel()
