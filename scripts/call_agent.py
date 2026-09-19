"""Have a full voice conversation with the agent — no phone, no microphone.

This is the acceptance test for "does the assistant actually work". It does
exactly what the app does, so if this passes, the app will too:

  1. asks BackPAC-BE for a session (which mints a LiveKit room and puts the
     agent in it),
  2. joins that room as an ordinary participant,
  3. publishes real speech as its microphone — the text below is synthesised
     with ElevenLabs, so the agent's Azure STT hears a genuine human voice, not
     a text shortcut,
  4. records everything that comes back: the transcript and the result cards
     over the data channel, and the agent's own audio.

Run both services first, then:

    make call                                  # the default script
    python -m scripts.call_agent "trains from delhi to jaipur on friday"

Exit status is 0 only if the agent transcribed the speech, answered, and spoke
back — so it is usable in CI.
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request

from config.env import load_env, require

load_env()
require("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")

import os  # noqa: E402  (after load_env, so .env is in scope)

from livekit import rtc  # noqa: E402

# 24 kHz mono PCM: what ElevenLabs can emit directly and LiveKit accepts, so no
# resampling in between.
SAMPLE_RATE = 24_000
FRAME_MS = 20

DEFAULT_SCRIPT = [
    "Hello! I want to take a train from Delhi to Jaipur on the second of "
    "October, for two people.",
]


def _http_json(url: str, payload: dict, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_session(backend: str) -> dict:
    """Ask the backend for a room, exactly as the app does."""
    try:
        return _http_json(
            f"{backend}/api/v1/sessions",
            {"agentId": "trip-planner", "participantName": "call-agent-script"},
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(
            f"POST /sessions failed ({exc.code}): {body}\n"
            "503 means LiveKit keys are missing from BackPAC-BE/.env; "
            "502 means BackPAC-Agent is not running."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach the backend at {backend}: {exc.reason}\n"
            "Start it with: uvicorn app.main:app --port 8000"
        ) from exc


def synthesize(text: str) -> bytes:
    """Turn the line into speech, so the agent's STT has real audio to hear.

    A different voice from the agent's own, so the two are easy to tell apart
    in a recording.
    """
    voice = os.getenv("CALLER_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Julia
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        f"?output_format=pcm_{SAMPLE_RATE}",
        data=json.dumps({"text": text, "model_id": "eleven_turbo_v2_5"}).encode(),
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


class Call:
    """One conversation, and what came back from it."""

    def __init__(self) -> None:
        self.transcript: list[dict] = []
        self.cards: list[dict] = []
        self.agent_audio_bytes = 0

    # --- what the agent sends us -----------------------------------------
    def on_data(self, packet: rtc.DataPacket) -> None:
        try:
            message = json.loads(packet.data.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return

        if packet.topic == "transcription":
            self.transcript.append(message)
            if message.get("text"):
                print(message["text"], end="", flush=True)
            return

        # Cards come wrapped in an RTVI envelope — the payload is one level in.
        body = message.get("data") if message.get("type") == "server-message" else message
        if isinstance(body, dict) and body.get("type") == "agent-card":
            self.cards.append(body)
            rows = body.get("payload")
            count = len(rows) if isinstance(rows, list) else "?"
            print(f"\n   [card] {body.get('cardType')}: {count} result(s)")

    @property
    def user_text(self) -> str:
        return " ".join(
            m["text"] for m in self.transcript if m.get("role") == "user" and m.get("text")
        )

    @property
    def agent_text(self) -> str:
        """Everything the agent said, one line per turn.

        `speech_final` marks the end of a turn, so the greeting and the answer
        that follows it are separate utterances — joining them blind produces
        "…travel?I'll search…" and makes it look like a formatting bug that
        isn't there.
        """
        turns: list[str] = []
        current = ""
        for message in self.transcript:
            if message.get("role") != "agent":
                continue
            current += message.get("text") or ""
            if message.get("speech_final"):
                if current.strip():
                    turns.append(current.strip())
                current = ""
        if current.strip():
            turns.append(current.strip())
        return "\n---\n".join(turns)


async def speak_into(source: rtc.AudioSource, pcm: bytes, tail_silence_s: float = 2.0):
    """Push audio in at real speed, then a little silence.

    Paced against a monotonic deadline rather than a fixed sleep: sleeping a
    flat 20ms per frame drifts, and a tight loop starves the event loop badly
    enough to trip LiveKit's connection keepalive. The trailing silence is what
    tells the agent's end-of-turn detection that the sentence is over.
    """
    chunk = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono
    frames = [pcm[i : i + chunk] for i in range(0, len(pcm) - chunk, chunk)]
    frames += [b"\x00" * chunk] * int(tail_silence_s * 1000 / FRAME_MS)

    started = time.monotonic()
    step = FRAME_MS / 1000
    for n, buf in enumerate(frames):
        await source.capture_frame(rtc.AudioFrame(buf, SAMPLE_RATE, 1, chunk // 2))
        await asyncio.sleep(max(0.0, started + (n + 1) * step - time.monotonic()))


async def type_into(room: rtc.Room, text: str) -> None:
    """Send a typed turn over the data channel, as the app's keyboard does.

    The agent handles it identically to speech (see infrastructure/text_input.py),
    which is the whole point — one conversation, two ways in.
    """
    await room.local_participant.publish_data(
        json.dumps({"type": "user-text", "text": text}).encode(),
        reliable=True,
        topic="user-text",
    )


async def run(lines: list[str], backend: str, wait_s: int, typed: bool) -> Call:
    info = start_session(backend)
    livekit = info["livekit"]
    print(f"session {info['sessionId']} -> room {livekit['roomName']}")

    call = Call()
    room = rtc.Room()
    room.on("data_received")(call.on_data)

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):  # noqa: ARG001
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_drain(track, call))

    await room.connect(livekit["url"], livekit["token"], rtc.RoomOptions(auto_subscribe=True))
    print("joined the room")

    source = rtc.AudioSource(SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    # The agent greets whoever joins; let it finish before talking over it.
    print("waiting for the greeting…")
    await asyncio.sleep(10)

    for line in lines:
        print(f"\n\nyou  > {line}{'  (typed)' if typed else ''}\nbot  > ", end="", flush=True)
        if typed:
            await type_into(room, line)
        else:
            await speak_into(source, synthesize(line))
        await asyncio.sleep(wait_s)

    await room.disconnect()
    return call


async def _drain(track, call: Call) -> None:
    """Count the agent's audio, so 'did it actually speak?' is answerable."""
    async for event in rtc.AudioStream(track):
        call.agent_audio_bytes += len(event.frame.data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lines", nargs="*", default=None, help="what to say")
    parser.add_argument(
        "--backend",
        default=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=45,
        help="seconds to wait for a reply after each line",
    )
    parser.add_argument(
        "--type",
        dest="typed",
        action="store_true",
        help="send the lines as typed text instead of speech — exercises the "
        "app's keyboard path (and STT_MODE=device) rather than the microphone",
    )
    args = parser.parse_args()

    call = asyncio.run(
        run(args.lines or DEFAULT_SCRIPT, args.backend, args.wait, args.typed)
    )

    print("\n\n--- result ---")
    print(f"heard from you : {call.user_text or '(nothing — STT did not transcribe)'}")
    print(f"agent replied  : {call.agent_text or '(nothing)'}")
    print(f"result cards   : {len(call.cards)}")
    print(f"agent audio    : {call.agent_audio_bytes:,} bytes")

    ok = bool(call.user_text) and bool(call.agent_text) and call.agent_audio_bytes > 0
    print("\nPASS — the full voice loop works." if ok else "\nFAIL — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
