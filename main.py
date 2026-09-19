"""BackPAC agent runtime — the process that puts a voice bot into a room.

HTTP surface (same shape as the ajimganj concierge):
  GET  /             health
  POST /start        join a LiveKit room and start talking      (called by BE)
  POST /stop         end a session
  GET  /sessions     list running sessions

Run it with:  uvicorn main:app --reload --port 8080
(or `python main.py`)

The flow, end to end:
  BackPAC-BE mints a LiveKit room + token, then calls POST /start here with the
  room name. This service joins that room as a bot, runs the voice pipeline
  (STT -> LangGraph brain -> ElevenLabs TTS), and talks to whoever the app put
  in the room with the token.

This file is intentionally the only place Pipecat/LiveKit transport is set up.
The brain (src/bot/core) and the voice stages (src/bot/processors) stay
testable without it.
"""

import asyncio
import logging
import os

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from config.settings import GREETING_MESSAGE, PARTICIPANT_NAME
from models.bot import StartRequest, StartResponse, StopRequest
from src.bot.core.coordinator import build_graph
from src.bot.processors.langgraph_processor import LangGraphProcessor
from src.bot.processors.pipeline import build_stt, build_tts, create_pipeline

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("backpac-agent")

app = FastAPI(title="BackPAC Agent", version="0.1.0")

# session_id -> asyncio.Task running that bot.
running: dict[str, asyncio.Task] = {}


async def run_bot(room_name: str, session_id: str) -> None:
    """Join the room and run the pipeline until the session ends.

    The LiveKit connection details come from the environment (LIVEKIT_URL /
    _API_KEY / _API_SECRET) — the same project BackPAC-BE minted the room in.
    """
    from pipecat.runner.livekit import configure
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.frames.frames import TTSSpeakFrame

    # `configure` reads LIVEKIT_* from env and returns a transport bound to the
    # room. This is the single spot LiveKit is touched.
    transport = await configure(room_name=room_name, participant_name=PARTICIPANT_NAME)

    async with aiohttp.ClientSession() as http:
        stt = build_stt()
        tts = build_tts(http)
        graph = build_graph()
        brain = LangGraphProcessor(graph, room_name)

        pipeline = create_pipeline(transport, stt, tts, brain)
        task = PipelineTask(pipeline)

        # Greet as soon as the bot is in the room, before the user speaks.
        await task.queue_frame(TTSSpeakFrame(GREETING_MESSAGE))

        logger.info("[%s] bot running in room %s", session_id, room_name)
        await PipelineRunner().run(task)


async def _session(room_name: str, session_id: str) -> None:
    try:
        await run_bot(room_name, session_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — log and clean up, never leak a task
        logger.exception("[%s] session crashed", session_id)
    finally:
        running.pop(session_id, None)
        logger.info("[%s] session cleaned up", session_id)


@app.get("/")
async def health() -> dict:
    return {"status": "running", "service": "backpac-agent", "sessions": len(running)}


@app.post("/start", response_model=StartResponse)
async def start(request: StartRequest) -> StartResponse:
    if request.session_id in running and not running[request.session_id].done():
        raise HTTPException(409, f"session {request.session_id} already running")

    task = asyncio.create_task(_session(request.room_name, request.session_id))
    running[request.session_id] = task
    return StartResponse(session_id=request.session_id)


@app.post("/stop")
async def stop(request: StopRequest) -> dict:
    task = running.get(request.session_id)
    if task is None:
        raise HTTPException(404, f"session {request.session_id} not found")
    task.cancel()
    return {"status": "stopping", "session_id": request.session_id}


@app.get("/sessions")
async def sessions() -> dict:
    return {"running": [sid for sid, t in running.items() if not t.done()]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
