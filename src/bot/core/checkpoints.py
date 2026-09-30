"""Where a conversation's memory lives between turns — and between deploys.

LangGraph keeps the state of a conversation in a *checkpointer*, keyed by
thread_id. Which one you use decides whether "resume this chat" is real:

- `MemorySaver` is a dictionary in this process. Restart the agent, deploy a
  change, or run a second container, and the state is gone or on the wrong
  machine. The transcript would still come back from the database, so the
  screen would look right — and the agent would then ask where you wanted to go
  about a trip you had already planned with it. That is worse than no resume at
  all, because it looks like it worked.

- `AsyncPostgresSaver` puts it in Postgres, next to everything else. Survives
  restarts, shared by every container, and gone for good when the conversation
  is deleted.

**It lives in its own schema.** `langgraph` rather than `public`, so its tables
are plainly not ours, Alembic's autogenerate ignores them by the rule in
`migrations/env.py`, and Supabase does not expose them over PostgREST. The
schema is created here and the tables by LangGraph's own `setup()`.

**Connection string.** Postgres, not the `postgresql+asyncpg://` SQLAlchemy
URL the backend uses — this is psycopg, a different driver. Set
`LANGGRAPH_DATABASE_URL`, or let it fall back to `DATABASE_URL` with the
SQLAlchemy driver suffix stripped.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

#: Kept apart from the application's own tables. See the module docstring.
SCHEMA = "langgraph"


def _dsn() -> str | None:
    """The psycopg connection string, or None if there is no Postgres to use."""
    url = os.getenv("LANGGRAPH_DATABASE_URL") or os.getenv("DATABASE_URL", "")
    if not url:
        return None
    # SQLAlchemy writes the driver into the scheme ("postgresql+asyncpg://");
    # psycopg wants it plain. Sharing one env var between the two is worth this
    # one line of translation.
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("postgresql+psycopg://", "postgresql://")
    if not url.startswith("postgres"):
        return None
    return url


#: One pool for the whole process, shared by every call.
#:
#: It used to be one pool per call, opened at psycopg's default of four
#: connections plus a fifth for setup — against the same Supabase pooler the
#: API uses, whose free plan allows fifteen clients in total. Three concurrent
#: calls exhausted it: new calls quietly fell back to memory (so resume broke)
#: and the API's own queries started failing. Now the process holds at most
#: CHECKPOINT_POOL_MAX connections however many calls it is carrying, and the
#: schema/setup round trips happen once instead of before every call.
_POOL_MIN = int(os.getenv("CHECKPOINT_POOL_MIN", "1"))
_POOL_MAX = int(os.getenv("CHECKPOINT_POOL_MAX", "4"))

_shared = None  # AsyncPostgresSaver, once built
_shared_stack: AsyncExitStack | None = None
_shared_lock = asyncio.Lock()


async def _build_shared(dsn: str):
    """Open the pool and run LangGraph's setup. Raises on any failure."""
    global _shared, _shared_stack
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    # The schema has to exist before any pooled connection points at it, so
    # this one connection is opened outside the pool.
    async with await AsyncConnection.connect(
        dsn, autocommit=True, prepare_threshold=0
    ) as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    async def _use_our_schema(conn) -> None:
        """Runs on every connection the pool hands out.

        `SET search_path` is per-connection, so doing it once on a borrowed
        connection would leave every later one pointing at `public` — and
        LangGraph would quietly create its tables next to ours. The pool's
        configure hook covers all of them.
        """
        await conn.execute(f"SET search_path TO {SCHEMA}")

    stack = AsyncExitStack()
    try:
        # autocommit: LangGraph's setup() issues DDL, which must not sit inside
        # a transaction something else may roll back.
        # prepare_threshold=0: Supabase is reached through a connection pooler,
        # which hands the same backend connection to different clients over
        # time — a prepared statement one made is not there for the next. Same
        # reason the API disables asyncpg's cache.
        # check: a connection the pooler dropped overnight is replaced when it
        # is borrowed, instead of failing the first turn of the morning.
        pool = await stack.enter_async_context(
            AsyncConnectionPool(
                conninfo=dsn,
                min_size=_POOL_MIN,
                max_size=_POOL_MAX,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                configure=_use_our_schema,
                check=AsyncConnectionPool.check_connection,
            )
        )
        saver = AsyncPostgresSaver(pool)
        # Creates its tables if missing. Safe to call on every boot.
        await saver.setup()
    except BaseException:
        await stack.aclose()
        raise
    _shared, _shared_stack = saver, stack
    logger.info(
        "LangGraph checkpoints in Postgres (schema %s, pool %d-%d)",
        SCHEMA,
        _POOL_MIN,
        _POOL_MAX,
    )


@asynccontextmanager
async def make_checkpointer():
    """Yield the best checkpointer available, as a context manager.

    The Postgres one is shared by every call and outlives this block — see
    `_shared`; it is closed once, at shutdown, by `close_shared_checkpointer`.

    Falls back to `MemorySaver` — with a loud warning — when there is no
    Postgres configured or it cannot be reached. A fallback rather than a crash
    because a checkpoint store being down should cost you resume, not the
    ability to take a call at all. The warning is there so this never passes
    for normal. A failure is not remembered: the next call tries again, so a
    database blip costs the calls during it and not every call until restart.

    Note the shape: everything that can fail happens *before* the `yield`, and
    the `yield` itself is not inside a `try` that catches `Exception`. It was,
    once — and it swallowed errors raised by the caller's own body, decided the
    checkpointer had failed, and yielded a second time, which asyncontextmanager
    reports as the baffling "generator didn't stop after athrow()". A context
    manager must only handle its own setup failures.
    """
    dsn = _dsn()
    if dsn is None:
        logger.warning(
            "no Postgres for LangGraph checkpoints — using in-memory state. "
            "Conversations will NOT survive a restart and cannot be resumed. "
            "Set LANGGRAPH_DATABASE_URL to fix."
        )
        yield MemorySaver()
        return

    saver = _shared
    if saver is None:
        async with _shared_lock:
            if _shared is None:
                try:
                    await _build_shared(dsn)
                except Exception as exc:  # noqa: BLE001 — see the docstring
                    logger.warning(
                        "Postgres checkpointer unavailable (%s) — falling back "
                        "to memory. Conversations will not survive a restart.",
                        exc,
                    )
            saver = _shared

    yield saver if saver is not None else MemorySaver()


async def close_shared_checkpointer() -> None:
    """Close the shared pool. Called once, when the process shuts down."""
    global _shared, _shared_stack
    stack, _shared, _shared_stack = _shared_stack, None, None
    if stack is not None:
        await stack.aclose()
