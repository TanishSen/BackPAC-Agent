"""The spoken hello the app plays while the orb is still arriving.

The welcome screen wants two things at once: the agent's actual voice, and a
level for every moment of it so the orb's mouth moves with the words instead of
near them. Both come from one ElevenLabs call:

  * ask for raw PCM rather than MP3,
  * measure it to produce the level track,
  * wrap the same bytes in a WAV header so the app can play them.

One request, no second encode, no audio library. WAV is 44 bytes of header in
front of the samples ElevenLabs already sent.

This is deliberately **not** a LiveKit call. Greeting someone should not cost a
room, an agent session and a microphone permission prompt before they have even
tapped anything — it is one short audio file, cached after the first request.
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import struct
import wave
from dataclasses import dataclass

import httpx
from loguru import logger

# ElevenLabs can emit PCM at this rate directly, so nothing is resampled.
SAMPLE_RATE = 24_000

# One level per this many milliseconds. 50ms is about as coarse as you can go
# before the orb stops looking like it is forming words.
FRAME_MS = 50

# Everything the orb says on the welcome screen.
#
# All of it is kept short on purpose: each line also appears in the speech
# bubble above the orb's head, and much more than this wraps past three lines on
# a phone. Short also means quick to synthesise and small to send.

# What it opens with. Warm, and a question, so the screen reads as the start of
# a conversation rather than a splash.
GREETINGS = [
    "Hey! Where are we off to?",
    "Hey there! Got somewhere in mind?",
    "Hey! Where would you like to go?",
    "Hey — ready to plan something?",
]

# When you poke it. Short — these land right after a tap, and anything longer
# than a breath stops feeling like a reaction and starts feeling like a speech.
# Surprised, ticklish, a bit indignant, and always steering back to the point,
# because the joke is that it would much rather be planning your trip.
POKE_LINES = [
    "Hey! Stop it!",
    "What are you doing?",
    "Okay, okay — I'm ticklish!",
    "Don't poke me!",
    "Again? Let's plan instead.",
    "Hey! That tickles.",
    "Oi! Trip first.",
    "Boop. Now, where to?",
]

# When you leave it alone. A nudge, never a nag: curious rather than pushy, and
# the app spaces these further and further apart (see the Flutter side).
IDLE_LINES = [
    "So… where are we going?",
    "Psst. Beach, or mountains?",
    "Still thinking? I've got ideas.",
    "Come on, let's plan something!",
    "Name a city. Any city.",
    "I'm ready when you are.",
    "What are you thinking?",
]

# The welcome screen fetches all of these in one go and holds them, so a tap
# answers instantly instead of waiting on a round trip and a synthesis.
WELCOME_SETS = {
    "greeting": GREETINGS,
    "poke": POKE_LINES,
    "idle": IDLE_LINES,
}


@dataclass(frozen=True)
class Greeting:
    """A spoken line: the words, the audio, and how loud it is over time."""

    text: str
    wav: bytes
    levels: list[float]  # 0..1, one per FRAME_MS


# Synthesis costs money and takes a second, and there are only a handful of
# lines, so each is generated once per process. Keyed by (voice, text) because
# changing the voice must not serve the old one back.
_cache: dict[tuple[str, str], Greeting] = {}

# In-flight work, so two callers asking for the same line share one request
# instead of paying for it twice. A task per line rather than one global lock:
# a global lock would serialise a whole batch, turning a 5-second fetch of
# twenty lines into half a minute.
_inflight: dict[tuple[str, str], asyncio.Task[Greeting]] = {}


def _levels_from_pcm(pcm: bytes, *, frame_ms: int = FRAME_MS) -> list[float]:
    """Loudness per frame, normalised to 0..1.

    Peak amplitude rather than RMS: RMS of speech sits low and flat, and the
    orb reads much better when quiet passages actually look quiet. The track is
    normalised against its own maximum so a softly-recorded voice still animates
    fully.
    """
    samples_per_frame = SAMPLE_RATE * frame_ms // 1000
    step = samples_per_frame * 2  # 16-bit mono

    peaks: list[float] = []
    for start in range(0, len(pcm) - step + 1, step):
        chunk = pcm[start : start + step]
        values = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        peaks.append(max(abs(v) for v in values) / 32768.0)

    if not peaks:
        return []

    ceiling = max(peaks) or 1.0
    # A gentle curve: loud parts stay loud, but quiet consonants still show,
    # which linear scaling loses entirely.
    return [round(min(1.0, (p / ceiling) ** 0.65), 3) for p in peaks]


def _to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV container — a header, not a re-encode."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)  # 16-bit
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


async def _synthesize(text: str, voice_id: str) -> Greeting:
    api_key = os.environ["ELEVENLABS_API_KEY"]
    model = os.getenv("ELEVENLABS_MODEL", "eleven_v3")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            params={"output_format": f"pcm_{SAMPLE_RATE}"},
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": model},
        )
        response.raise_for_status()
        pcm = response.content

    logger.info(f"greeting synthesised: {len(pcm)} bytes of PCM for {text!r}")
    return Greeting(text=text, wav=_to_wav(pcm), levels=_levels_from_pcm(pcm))


async def get_greeting(text: str | None = None) -> Greeting:
    """One spoken line, synthesised on first use and cached after that.

    Pass `text` to say something specific; leave it out for a random greeting.
    """
    voice_id = os.environ["ELEVENLABS_VOICE_ID"]
    line = text or random.choice(GREETINGS)
    key = (voice_id, line)

    if key in _cache:
        return _cache[key]

    task = _inflight.get(key)
    if task is None:
        task = asyncio.create_task(_synthesize(line, voice_id))
        _inflight[key] = task
    try:
        greeting = await task
    finally:
        _inflight.pop(key, None)
    _cache[key] = greeting
    return greeting


async def get_welcome_lines() -> dict[str, list[Greeting]]:
    """Every line the welcome screen might need, synthesised in parallel.

    The app asks for these once and keeps them, so poking the orb answers
    immediately rather than after a round trip and a second of synthesis. Cold,
    this takes a few seconds; warm, it is instant. The app fetches it in the
    background after the opening greeting, so neither case is ever visible.

    If these lines ever stop changing, pre-render them and ship them as app
    assets — this endpoint exists so they can keep changing.
    """
    results = await asyncio.gather(
        *(
            asyncio.gather(*(get_greeting(line) for line in lines))
            for lines in WELCOME_SETS.values()
        )
    )
    return dict(zip(WELCOME_SETS.keys(), results))
