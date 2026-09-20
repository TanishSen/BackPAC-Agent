"""BackPAC agent runtime — the process that puts a voice bot into a LiveKit room.

HTTP surface (called by BackPAC-BE, mirrors the ajimganj concierge):
  GET  /             health
  POST /start        join a room and start the voice loop
  POST /stop         end a session
  GET  /sessions     list running sessions

Run:  python main.py         (or  uvicorn main:app --port 8080)

End-to-end flow:
  BackPAC-BE mints a LiveKit room + token, then calls POST /start here with the
  room name. This service joins that room, builds the pipeline
  (STT → trip brain → ElevenLabs TTS), and talks to whoever joined with the
  app's token. Agent replies and transcripts are pushed back to the app over
  the LiveKit data channel (see WordInterceptor / CardDispatcher).

Everything Pipecat/LiveKit is confined to this file; the brain (src/bot/core)
and the voice stages (src/bot/processors) stay importable and testable on their
own.
"""

import asyncio
import logging
import os
import uuid

import aiohttp
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from config.env import load_env, require, voice_requirements

# Before any other import that reads os.getenv. Pinned to the project root, so
# it works however the process was launched. See config/env.py.
load_env()

from models.bot import (  # noqa: E402
    SpokenLine,
    WelcomeLinesResponse,
    StartRequest,
    StartResponse,
    StopRequest,
)
from src.bot.core.coordinator import build_graph
from src.bot.infrastructure.events import (
    setup_event_handlers,
    setup_user_aggregator_handlers,
)
from src.bot.infrastructure.transport import create_transport
from src.bot.processors.langgraph_processor import LangGraphProcessor
from src.bot.processors.pipeline import create_pipeline, create_services
from src.bot.clients.transcript import TranscriptClient
from src.bot.core.checkpoints import make_checkpointer
from src.bot.voice.greeting import FRAME_MS, get_greeting, get_welcome_lines

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backpac-agent")

# Die now, with a readable message, rather than accepting a /start and failing
# three layers deep inside a provider SDK once someone is already on the call.
require(*voice_requirements())

app = FastAPI(title="BackPAC Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> the asyncio Task running that bot.
running: dict[str, asyncio.Task] = {}


async def run_bot(
    room_name: str, session_id: str, *, thread_id: str | None = None
) -> None:
    """Join the room and run the pipeline until the session ends or is cancelled."""

    # Writes each turn to the backend so the conversation shows up in history.
    # Fire-and-forget by construction: see the class docstring. If the backend
    # is unreachable, the call carries on unrecorded rather than stalling.
    transcript = TranscriptClient(room_name=room_name)

    transport, aic_filter = create_transport(room_name)

    # The checkpointer holds the conversation between turns, keyed by
    # thread_id. Postgres-backed when one is configured, so resuming a chat
    # after a restart picks up the actual conversation rather than a blank one.
    async with make_checkpointer() as checkpointer, aiohttp.ClientSession() as http:
        graph = build_graph(checkpointer)
        stt, tts = create_services(
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            aiohttp_session=http,
        )

        brain = LangGraphProcessor(
            graph, room_name=room_name, transcript=transcript
        )
        if thread_id:
            brain.set_thread_id(thread_id)
        pipeline, aggregators, word_interceptor = create_pipeline(
            transport, aic_filter, stt, tts, brain, room_name
        )
        # The brain publishes user transcripts through the same data-channel
        # interceptor the pipeline uses for agent text.
        brain._word_interceptor = word_interceptor

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=900,
        )
        setup_event_handlers(transport, worker, brain, room_name=room_name)
        setup_user_aggregator_handlers(aggregators.user())

        runner = WorkerRunner()
        await runner.add_workers(worker)

        logger.info("[%s] bot running in room %s", session_id, room_name)
        try:
            await runner.run()
        except asyncio.CancelledError:
            logger.info("[%s] cancelling; cleaning up", session_id)
            try:
                await asyncio.wait_for(worker.cancel(), timeout=5.0)
            except (TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.warning("[%s] cleanup issue: %s", session_id, exc)
            # Wrap up before the loop goes away: name the conversation and
            # let the last thing said reach history.
            await transcript.finish()
            raise
        finally:
            await transcript.finish()


async def _session(
    room_name: str, session_id: str, thread_id: str | None = None
) -> None:
    try:
        await run_bot(room_name, session_id, thread_id=thread_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — log, never leak a dead task
        logger.exception("[%s] session crashed", session_id)
    finally:
        running.pop(session_id, None)
        logger.info("[%s] session cleaned up", session_id)


@app.get("/")
async def health() -> dict:
    return {"status": "running", "service": "backpac-agent", "sessions": len(running)}


@app.post("/start", response_model=StartResponse)
async def start(request: StartRequest) -> StartResponse:
    session_id = request.session_id or str(uuid.uuid4())
    if session_id in running and not running[session_id].done():
        raise HTTPException(409, f"session {session_id} already running")
    running[session_id] = asyncio.create_task(
        _session(request.room_name, session_id, request.thread_id)
    )
    logger.info(
        "%s session %s for room %s",
        "resumed" if request.is_resuming else "started",
        session_id,
        request.room_name,
    )
    return StartResponse(session_id=session_id)


@app.post("/stop")
async def stop(request: StopRequest) -> dict:
    task = running.get(request.session_id)
    if task is None:
        raise HTTPException(404, f"session {request.session_id} not found")
    task.cancel()
    return {"status": "stopping", "session_id": request.session_id}


@app.get("/greeting", response_model=SpokenLine)
async def greeting(text: str | None = None) -> SpokenLine:
    """The spoken hello for the welcome screen, with a level track for the orb.

    Not a call: no room, no session, no microphone permission. Synthesising here
    also warms the cache, so the `/voice-line.wav` request that follows is
    served from memory.
    """
    said = await get_greeting(text)
    return SpokenLine(text=said.text, levels=said.levels, frame_ms=FRAME_MS)


@app.get("/voice-line.wav")
async def voice_line(text: str) -> Response:
    """The audio for one line.

    A plain file rather than base64 in a JSON body, so the browser and the OS
    can cache it. The text fully determines the audio, so it is safe to mark
    immutable and never ask for it again.
    """
    said = await get_greeting(text)
    return Response(
        content=said.wav,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/welcome-lines", response_model=WelcomeLinesResponse)
async def welcome_lines() -> WelcomeLinesResponse:
    """Every line the orb can say on the welcome screen, audio included.

    The app fetches this once in the background and keeps it, so a tap on the
    orb gets an answer immediately. Cold it takes a few seconds while the lines
    are synthesised in parallel; warm it is instant.
    """
    groups = await get_welcome_lines()
    as_lines = {
        name: [
            SpokenLine(text=line.text, levels=line.levels, frame_ms=FRAME_MS)
            for line in lines
        ]
        for name, lines in groups.items()
    }
    return WelcomeLinesResponse(**as_lines)


@app.get("/sessions")
async def sessions() -> dict:
    return {"running": [sid for sid, t in running.items() if not t.done()]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
