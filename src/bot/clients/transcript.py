"""Writes the conversation back to BackPAC-BE as it happens.

**Nothing here is ever awaited by the call.** The agent is holding a live phone
conversation: a database write that takes 300ms to Singapore is 300ms of dead
air, and a backend that is down would otherwise take the call down with it. So
every method here schedules a background task and returns immediately, and a
failure is logged and dropped. A lost line of transcript is a small, quiet loss;
a stalled conversation is not.

The agent authenticates with a shared secret rather than a user token — it is a
server, and the person whose conversation this is went through the app hours of
wall-clock ago as far as this process is concerned. See `require_service` on the
backend.

Conversations are identified by room name, because that is what the agent was
given when it was asked to join. It never learns the backend's primary keys.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

#: How many opening turns to keep for naming the conversation. See
#: src/bot/voice/titler.py, which uses the first few of them.
TURNS_KEPT = 6


class TranscriptClient:
    def __init__(
        self,
        *,
        room_name: str,
        base_url: str | None = None,
        service_token: str | None = None,
        timeout: float = 5.0,
    ):
        self._room = room_name
        self._base = (base_url or os.getenv("BACKEND_URL", "")).rstrip("/")
        self._token = service_token or os.getenv("BACKEND_SERVICE_TOKEN", "")
        self._timeout = timeout
        # Tasks are held so the event loop does not garbage-collect one
        # mid-flight — asyncio only keeps weak references to running tasks, and
        # a dropped reference is a write that silently never happens.
        self._pending: set[asyncio.Task] = set()

        # The opening turns, kept so the conversation can be named when it
        # ends. Capped: a title is written from the start of a conversation,
        # so holding a whole twenty-minute transcript in memory to produce
        # five words would be waste.
        self._opening: list[tuple[str, str]] = []
        self._finished = False

        if not self._enabled:
            logger.info(
                "[%s] transcript logging off (BACKEND_URL or "
                "BACKEND_SERVICE_TOKEN unset) — the call still works, it just "
                "will not appear in history",
                room_name,
            )

    @property
    def _enabled(self) -> bool:
        return bool(self._base and self._token)

    def user_said(self, text: str) -> None:
        self._remember("user", text)
        self._post("messages", {"roomName": self._room, "role": "user", "content": text})

    def agent_said(self, text: str, *, meta: dict | None = None) -> None:
        self._remember("agent", text)
        self._post(
            "messages",
            {
                "roomName": self._room,
                "role": "agent",
                "content": text,
                "meta": meta or {},
            },
        )

    def showed_card(self, *, result_type: str, payload: dict) -> None:
        self._post(
            "trip-results",
            {
                "roomName": self._room,
                "resultType": result_type,
                "payload": payload,
            },
        )

    def _remember(self, role: str, text: str) -> None:
        if len(self._opening) < TURNS_KEPT and text and text.strip():
            self._opening.append((role, text.strip()))

    def _post(self, path: str, body: dict) -> None:
        """Fire and forget. Never raises, never blocks the caller."""
        if not self._enabled:
            return
        if not body.get("content", "x").strip():
            return  # an empty turn is not a turn
        task = asyncio.create_task(self._send(path, body))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, path: str, body: dict) -> None:
        url = f"{self._base}/api/v1/sessions/internal/{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.post(
                    url, json=body, headers={"X-Service-Token": self._token}
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "[%s] transcript write refused (%s): %s",
                        self._room,
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as exc:  # noqa: BLE001 — a lost line must not end a call
            logger.warning("[%s] transcript write failed: %s", self._room, exc)

    async def finish(self, timeout: float = 12.0) -> None:
        """Wrap the conversation up: name it, then flush what is in flight.

        Runs after the call is over, so the time it takes costs nobody
        anything. Naming first, because the title is itself a write that then
        needs draining with the others.

        Idempotent, and it has to be: a cancelled call runs its own cleanup and
        *then* the `finally` block, so this is reached twice on the ordinary
        hang-up path. Without the guard every call would be named twice — two
        model calls and two writes for one conversation.
        """
        if self._finished:
            return
        self._finished = True

        await self._name_conversation()
        await self.drain(timeout=timeout)

    async def _name_conversation(self) -> None:
        """One Haiku call to turn the opening turns into a history label.

        Never raises and never blocks anything a user is waiting on: by the
        time this runs the line is dead. A failure leaves the placeholder
        title — the user's first sentence, truncated — which is worse to read
        but perfectly correct.
        """
        if not self._enabled or not self._opening:
            return
        try:
            from src.bot.voice.titler import name_conversation

            title = await name_conversation(self._opening)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] naming failed: %s", self._room, exc)
            return

        if title:
            logger.info("[%s] named this conversation %r", self._room, title)
            await self._send("title", {"roomName": self._room, "title": title})

    async def drain(self, timeout: float = 3.0) -> None:
        """Let the writes still in flight finish, when a call ends.

        Called on the way out, so the last thing said makes it into history
        rather than being cancelled with the event loop. Bounded, because
        hanging up should not wait on a slow network either.
        """
        if not self._pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "[%s] %d transcript write(s) still unfinished at hang-up",
                self._room,
                len(self._pending),
            )
