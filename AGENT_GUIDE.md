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
  → Azure STT turns it into text                   (processors/pipeline.py)
  → the user aggregator batches it and waits for the person to FINISH,
    then emits one LLMContextFrame for the whole turn
  → LangGraphProcessor feeds that text to the graph (processors/langgraph_processor.py)
      → orchestrator reads it, calls transfer_to_trains/flights/stays
      → that specialist runs, calls a tool (search_trains…) which hits the BACKEND
      → specialist writes a short spoken reply, streamed token by token
  → the text filter strips markdown, the transcript goes to the app,
    ElevenLabs speaks it back into the room
person hears it
```

Routing is **cheap and deterministic**: the orchestrator doesn't reason in prose
about who should handle the request — it *calls a tool*, and `coordinator.py`
reads which tool to pick the next node. No extra LLM call to route.

## Four rules that are not obvious, and each cost a bug

**1. A turn starts on `LLMContextFrame`, not `TranscriptionFrame`.**
The user aggregator upstream *consumes* transcription frames — it batches them
and waits for voice activity and end-of-turn analysis to agree the person has
stopped. Listening for `TranscriptionFrame` in the brain means hearing nothing
at all, ever, and silently: audio flows, STT works, the logs look healthy and
the agent simply never answers.

**2. Every LangGraph node must be `async`.**
LangGraph runs a *synchronous* node on a thread from the default executor. In
this process that pool is already contended — 50 audio frames a second, plus
Silero VAD and end-of-turn ONNX inference. A sync node was measured taking **76
seconds** to reach its first API call, while the caller listened to silence. Use
`ainvoke`, and keep the tools in `core/tools.py` async too.

**3. Cards reach the app inside an RTVI envelope.**
`card_dispatcher.py` emits an `RTVIServerMessageFrame`; Pipecat wraps it as
`{"label":"rtvi-ai","type":"server-message","data":{…}}`. A client checking
`type == "agent-card"` on the outer object drops every card.

**4. Inbound data packets are frames, not events.**
Pipecat's LiveKit transport never calls an `on_data_received` event handler —
it turns each packet into a transport-message frame and pushes it downstream.
A handler registered on the transport is never invoked. That is why typed input
is a pipeline stage (`processors/text_input.py`), not a callback.

## Run it

```bash
cp .env.example .env          # fill in the keys (see the table below)
make install                  # uv venv + deps
make run                      # starts on :8080
```

It refuses to start, with a clear message naming the variable, if a required
key is missing — better than accepting a call and failing mid-conversation.

It won't do anything until BackPAC-BE calls its `POST /start` (that's what puts
it in a room). To develop the brain alone, skip all that — see below.

## The keys in `.env`

| Key | What | Where it's used |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | `src/bot/core/llm.py` |
| `LIVEKIT_URL/_API_KEY/_API_SECRET` | the room — **same project as the backend** | `infrastructure/transport.py` |
| `ELEVENLABS_API_KEY` | the voice | `processors/pipeline.py` |
| `ELEVENLABS_VOICE_ID` | *which* voice — the id you picked | same |
| `STT_MODE` | `azure` or `device` (see below) | same |
| `AZURE_SPEECH_API_KEY/_REGION` | server STT (only when `STT_MODE=azure`) | same |
| `BACKEND_URL` | where the tools send searches | `core/tools.py` |
| `INTERRUPT_ON_TRANSCRIPTION_ONLY` | see barge-in below | `processors/pipeline.py` |
| `AIC_SDK_LICENSE` | optional noise filter; ignored if the SDK isn't installed | `infrastructure/transport.py` |

`config/env.py` loads this file from the project root — pinned by path, not by
searching upward from whichever file happened to call it, which is why a script
in `scripts/` picks up the same keys the server does.

## The models

Two tiers, in `core/llm.py`:
- **Haiku** (`get_llm`) — routing + the trains/flights specialists. Fast, cheap.
- **Sonnet** (`get_llm_heavy`) — heavier reasoning (the stays agent uses it).

Change the model id in *one place* (`llm.py`) if a newer one ships.

## STT: the one architecture choice

`processors/pipeline.py` → `create_services()` reads `STT_MODE`:

- **`azure`** (default) — the server transcribes the audio stream, in the
  language named by `STT_LANGUAGES` (default `en-IN`). Recovers automatically
  from Azure's buffer-overflow disconnects.

  **Do not turn auto-detect back on without a reason.** Azure decides the
  language from the *first* utterance, and the first thing a laptop's
  microphone hears is usually the agent's own greeting off the speaker. Guess
  "hi-IN" from that and the rest of the call arrives transliterated — real
  example, from this app: "I just want to plan my trip" came back as
  "आई एम जस्ट वांट टू प्लान माय ट्रिप". Claude still understands it, which is
  exactly why it is easy to miss, but the transcript on screen looks broken.
  Listing several codes in `STT_LANGUAGES` opts back in.
- **`device`** — the phone transcribes and posts the words itself. No server STT
  is built. The words arrive through `processors/text_input.py`, the same stage
  the app's keyboard uses, so the brain cannot tell the two apart.

TTS is always ElevenLabs `eleven_v3` — voice, model and tuning all from env.

## English only

The assistant answers in English whatever it is asked in. Two things enforce it,
and both are needed:

- `STT_LANGUAGES=en-IN` tells Azure the language rather than asking it to guess
  (see above).
- `prompts/global_constraints.py` and the orchestrator prompt both open with the
  rule, because the orchestrator never sees the global constraints.

Verified by typing Hindi at it: it understood, and replied in English.

## What the orb says on the welcome screen

Three routes serve it, and none of them is a call — greeting someone should not
cost a LiveKit room, an agent session and a microphone permission prompt before
they have tapped anything:

- `GET /greeting` — one opening line.
- `GET /welcome-lines` — the whole set: openers, what it says when poked, what
  it says when left alone. The app fetches this once in the background and
  keeps it, so a tap on the orb answers with no round trip in the way.
- `GET /voice-line.wav?text=…` — the audio for one line, cacheable and marked
  immutable.

One ElevenLabs request gives both the audio and the level track: ask for PCM,
measure it for a loudness value every 50ms, then wrap the same bytes in a WAV
header. No second encode, no audio library. The level track is what makes the
orb's mouth follow the actual words rather than wobble near them.

Every line is cached, and the cache is keyed per line with its own in-flight
task — **not** one global lock, which would serialise a batch and turn a
five-second fetch of nineteen lines into half a minute.

The lines themselves live at the top of `src/bot/voice/greeting.py`. Edit them
there; they are deliberately short, because each one also has to fit in the
speech bubble above the orb's head.

## Barge-in

`INTERRUPT_ON_TRANSCRIPTION_ONLY=true` (the default in `.env.example`) means
only a real transcription stops the agent talking. Leave it on: raw voice
activity also fires on background noise and on the agent's own voice echoing
back off a phone speaker, and being cut off mid-sentence by a passing bus is
worse than a half-second slower interrupt. `DISABLE_INTERRUPTIONS=true` turns
barge-in off entirely, which is occasionally useful for a demo.

## Develop against the brain, not through a microphone

The highest-value habit. Same graph the voice agent runs, no mic, no LiveKit:

```bash
make brain
```

It reads the same `.env`, so there is nothing to export. Tool calls print as
they happen, so you can see whether it actually searched and what came back.
Start BackPAC-BE first or every search returns an error (which is itself worth
seeing — that is what the traveller would hear).

## Prove the whole thing works, with no phone

```bash
make call          # needs BackPAC-BE and this agent running
```

`scripts/call_agent.py` joins the room as an ordinary participant and pushes
**real synthesised speech** in as its microphone, so Azure STT does real work.
It records the transcript, the cards and the agent's audio, and exits non-zero
unless the agent heard, answered and spoke. `--type` drives the same loop
through the keyboard path instead.

## How to ADD a specialist (the common task)

Say you want a **cabs** agent. Four small edits, no plumbing:

1. **A tool** — in `core/tools.py`, add `async def search_cabs(...)` with
   `@tool` and a clear docstring, calling a backend endpoint. Async, per rule 2.
2. **A prompt** — `prompts/cabs_prompt.py`, one paragraph telling it its job.
3. **An agent** — in `agents/specialists.py`, add `create_cabs_agent()`
   (copy `create_trains_agent`).
4. **Register it** — add `"cabs": create_cabs_agent` to `SPECIALISTS` in
   `coordinator.py`. The orchestrator automatically gets a `transfer_to_cabs`
   handoff; nothing else changes.

Then add "cabs" to the orchestrator prompt's list so it knows to route there.

## What is deliberately not here yet

- **Memory across restarts.** `coordinator.py` uses `MemorySaver` — an
  in-process checkpointer. A restart loses every in-flight conversation. Swap it
  for a Redis or Postgres checkpointer when calls need to survive a deploy.
- **More than one agent process per set of calls.** `main.py` keeps running
  sessions in a dict in memory, so a `/stop` must reach the process that owns
  that call. Scale with more containers addressed individually, not with more
  uvicorn workers.

## Two rules about what it says

- **Keep spoken replies short.** `prompts/global_constraints.py` is prepended to
  every agent and enforces this. Don't let an agent read out lists or prices
  with symbols — it is going through TTS.
- **Never invent results.** Agents may only say what a tool returned. Keep it
  that way, or the bot will confidently offer a train that does not exist.

## What's in the repo

```
main.py                         joins the room, builds the pipeline, runs the worker
config/
  env.py                        loads .env from the project root; fails fast on gaps
  settings.py                   the greeting and the bot's display name
src/bot/
  core/
    llm.py                      Claude Haiku / Sonnet
    state.py                    the graph's shared state
    tools.py                    async search_trains/flights/stays -> the backend
    coordinator.py              orchestrator -> specialist routing, compiled graph
  agents/specialists.py         the trains / flights / stays agents
  prompts/                      each agent's brief + global spoken-output rules
  infrastructure/
    transport.py                LiveKit transport + token (AIC noise filter optional)
    events.py                   greeting on join + idle nudges
  voice/
    greeting.py                 the spoken hello for the welcome screen
  processors/
    pipeline.py                 create_services (STT+TTS) + create_pipeline
    langgraph_processor.py      turn -> graph -> streamed TTS + cards  (the bridge)
    text_input.py               typed turns / device-STT in through the data channel
    elevenlabs_v3_tts.py        the ElevenLabs voice (eleven_v3)
    word_interceptor.py         pushes the transcript to the app (data channel)
    card_dispatcher.py          pushes result cards to the app (RTVI)
    tts_text_filter.py          cleans text before TTS — and before the transcript
    jitter_buffer.py            smooths audio right after TTS
scripts/
  chat_brain.py                 terminal chat with the brain (no voice)
  call_agent.py                 a full call with no phone — the acceptance test
```

## Verified

Against the live services, not just compiled:

- The brain routes correctly, calls the real backend, resolves spoken dates
  ("the second of October") to real ISO dates, and remembers across turns.
- A full call works end to end: speech in → Azure STT → Claude → tool call →
  ElevenLabs → audio back, with the transcript and a result card on the data
  channel. `make call` exits 0.
- Typed turns drive the same loop (`make call --type`).

Worth doing once on a device and not possible from a laptop: **a human on a real
phone** — particularly barge-in, and echo behaviour on a speaker rather than
headphones.
