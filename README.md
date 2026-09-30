# BackPAC-Agent

The voice brain. Pipecat pipeline (STT → LangGraph → ElevenLabs TTS) with a
multi-agent Claude graph (orchestrator + trains/flights/stays specialists).

- New here? Read **AGENT_GUIDE.md** — the mental model, how a turn flows, how to
  add a specialist, and how to test the brain with no microphone.
- Run: `cp .env.example .env` → fill it → `uv pip install -r requirements.txt`
  → `python main.py` (:8080).

BackPAC-BE calls this service's `POST /start` to put a bot into a room.

---

Part of **backPAC**, a voice travel planner — start at
[BackPAC-Fe](https://github.com/TanishSen/BackPAC-Fe) for the overview and how the
three repositories fit together. MIT licensed; see [LICENSE](LICENSE).
