# BackPAC Agent — a guide for the developer picking this up

This is the voice brain. Someone talks to the app; this service listens, thinks
with Claude, and talks back in an ElevenLabs voice. It is modelled directly on
the ajimganj concierge, trimmed to what BackPAC needs.

## The one mental model

There are **three layers**, and they don't leak into each other:

```
  main.py            the voice PLUMBING: joins the LiveKit room, runs the pipeline
     │
  src/bot/processors the VOICE STAGES: speech-to-text, our brain, ElevenLabs TTS
     │
  src/bot/core       the BRAIN: a LangGraph of agents. No voice code at all.
  src/bot/agents     the specialists (trains, flights, stays)
  src/bot/prompts    what each agent is told to do
```

The best thing about this split: **the brain has no Pipecat in it.** You can
build and test the whole conversation from a plain Python script, no microphone,
no LiveKit. That is how you should develop the agent logic.

## How a turn actually flows

```
person speaks
  → LiveKit carries the audio into main.py's pipeline
  → STT turns it into text                         (processors/pipeline.py)
  → LangGraphProcessor feeds text to the graph     (processors/langgraph_processor.py)
      → orchestrator reads it, calls transfer_to_trains/flights/stays
      → that specialist runs, calls a tool (search_trains…) which hits the BACKEND
      → specialist writes a short spoken reply
  → ElevenLabs speaks the reply back into the room (processors/elevenlabs_v3_tts.py)
person hears it
```

Routing is **cheap and deterministic**: the orchestrator doesn't reason in prose
about who should handle the request — it *calls a tool*, and `coordinator.py`
reads which tool to pick the next node. No extra LLM call to route.

## Run it

```bash
cp .env.example .env          # fill in the keys (see the table below)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
python main.py                # starts on :8080
```

It won't do anything until BackPAC-BE calls its `POST /start` (that's what puts
it in a room). To develop the brain alone, skip all that — see "Test the brain
without voice" below.

## The keys in `.env`

| Key | What | Where it's used |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | `src/bot/core/llm.py` |
| `LIVEKIT_URL/_API_KEY/_API_SECRET` | the room — **same project as the backend** | `main.py` |
| `ELEVENLABS_API_KEY` | the voice | `processors/pipeline.py` |
| `ELEVENLABS_VOICE_ID` | *which* voice — the id you picked | same |
| `STT_MODE` | `azure` or `device` (see below) | same |
| `BACKEND_URL` | where the tools send searches | `core/tools.py` |

## The models

Two tiers, in `core/llm.py` (unchanged from ajimganj's approach):
- **Haiku** (`get_llm`) — routing + simple specialists. Fast, cheap. Default.
- **Sonnet** (`get_llm_heavy`) — heavier reasoning (the stays agent uses it).

Change the model id in *one place* (`llm.py`) if a newer one ships.

## STT: the one architecture choice

`processors/pipeline.py` → `build_stt()` reads `STT_MODE`:

- **`azure`** (default, ported from ajimganj) — the server transcribes the audio
  stream. Needs `AZURE_SPEECH_API_KEY` + `AZURE_SPEECH_REGION`. Works out of the
  box with an Azure key.
- **`device`** — the phone transcribes (the Flutter app already has a
  speech-to-text seam) and sends text over the LiveKit data channel. No Azure
  key, no server STT. This is the lighter path. **One thing is left to wire:** a
  processor that reads text frames off the data channel — the `TODO` in
  `create_pipeline`. Until that's done, use `azure`.

TTS is always ElevenLabs `eleven_v3` — voice, model and tuning all from env.

## Test the brain without voice

This is the highest-value habit. The graph is pure Python:

```python
import os; os.environ["ANTHROPIC_API_KEY"] = "..."   # your key
from src.bot.core.coordinator import build_graph

g = build_graph()
cfg = {"configurable": {"thread_id": "test-room"}}
for turn in ["I want to go to Jaipur by train next Friday", "the morning one"]:
    out = g.invoke({"messages": [{"role": "user", "content": turn}],
                    "active_agent": "orchestrator", "room_name": "test-room"}, cfg)
    print(out["messages"][-1].content)
```

If it reads well here, it will sound right in voice. Debug logic here, not
through a microphone.

## How to ADD a specialist (the common task)

Say you want a **cabs** agent. Four small edits, no plumbing:

1. **A tool** — in `core/tools.py`, add `search_cabs(...)` with `@tool` and a
   clear docstring, calling a backend endpoint.
2. **A prompt** — `prompts/cabs_prompt.py`, one paragraph telling it its job.
3. **An agent** — in `agents/specialists.py`, add `create_cabs_agent()`
   (copy `create_trains_agent`).
4. **Register it** — add `"cabs": create_cabs_agent` to `SPECIALISTS` in
   `coordinator.py`. The orchestrator automatically gets a `transfer_to_cabs`
   handoff; nothing else changes.

Then add "cabs" to the orchestrator prompt's list so it knows to route there.

## What's deliberately NOT here yet (and where it'd go)

- **On-screen cards** (agent tells the app to show a train list visually).
  ajimganj does this by emitting RTVI frames over the data channel from the
  langgraph processor. Add it in `langgraph_processor.py` when the app needs it.
- **Memory across restarts.** `coordinator.py` uses an in-memory checkpointer —
  conversations reset if the process restarts. Swap `MemorySaver()` for a Redis
  or Postgres checkpointer for durability.
- **Barge-in / interruptions, VAD tuning, jitter buffering.** ajimganj has all
  of these as extra processors. Port them from there when polish matters; the
  pipeline is structured the same way, so they drop in.

## Two rules

- **Keep spoken replies short.** `prompts/global_constraints.py` enforces this
  and is prepended to every agent. Don't let an agent read out lists or prices
  with symbols — it's going through TTS.
- **Never invent results.** Agents may only say what a tool returned. The
  constraints prompt says so; keep it that way, or the bot will confidently
  book a train that doesn't exist.
