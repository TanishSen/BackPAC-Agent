"""Builds the voice services (STT + TTS) and assembles the Pipecat pipeline.

This is where the ElevenLabs voice you picked in ajimganj plugs in, and where
the STT decision lives.

STT — two options, chosen by the `STT_MODE` env var:

  azure   (default, ported from ajimganj)  server-side Azure STT on the LiveKit
          audio stream. Needs AZURE_SPEECH_API_KEY + AZURE_SPEECH_REGION.

  device  the phone does STT (the Flutter app already has a speech-to-text
          seam) and sends text over the LiveKit data channel. No Azure key, no
          server STT — this is the lighter path you said you wanted. In this
          mode `build_stt()` returns None and the pipeline takes text frames
          straight in. Wiring the data-channel-text input is the one TODO
          marked below.

TTS is always ElevenLabs (eleven_v3), the voice + model + tuning all from env.
"""

import os
import logging

import aiohttp

from src.bot.processors.elevenlabs_v3_tts import ElevenLabsV3TTSService

logger = logging.getLogger(__name__)


def _opt_float(name: str) -> float | None:
    """ElevenLabs tuning knobs are all optional — return None when unset so the
    service uses its own defaults rather than 0.0."""
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


def build_stt():
    """Return an STT service, or None in device mode.

    Kept as its own function so the STT choice is one obvious place, and so a
    unit test can build the pipeline without any speech backend.
    """
    mode = os.getenv("STT_MODE", "azure").lower()
    if mode == "device":
        logger.info("STT_MODE=device — the phone transcribes; no server STT.")
        return None

    from pipecat.services.azure.stt import AzureSTTService

    logger.info("STT_MODE=azure — server-side Azure STT.")
    return AzureSTTService(
        api_key=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
    )


def build_tts(aiohttp_session: aiohttp.ClientSession):
    """The ElevenLabs voice. Voice id, model and tuning all come from env, so
    changing the voice is a config change, not a code change."""
    model_name = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
    logger.info("TTS: ElevenLabs %s, voice %s", model_name, os.getenv("ELEVENLABS_VOICE_ID"))
    return ElevenLabsV3TTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        aiohttp_session=aiohttp_session,
        model=model_name,
        params=ElevenLabsV3TTSService.InputParams(
            speed=_opt_float("ELEVENLABS_SPEED"),
            stability=_opt_float("ELEVENLABS_STABILITY"),
            similarity_boost=_opt_float("ELEVENLABS_SIMILARITY_BOOST"),
            style=_opt_float("ELEVENLABS_STYLE"),
        ),
    )


def create_pipeline(transport, stt, tts, langgraph_processor):
    """Wire the stages into a Pipecat pipeline.

    Order is the data flow: audio in from the room -> STT -> our LangGraph
    brain -> ElevenLabs TTS -> audio back out to the room.

    In device mode `stt` is None and that stage is dropped; text arrives on the
    transport's data channel instead (TODO below).
    """
    from pipecat.pipeline.pipeline import Pipeline

    stages = [transport.input()]
    if stt is not None:
        stages.append(stt)
    else:
        # TODO(agent): in device mode, add a processor that reads text frames
        # from the LiveKit data channel here. See AGENT_GUIDE.md "Device STT".
        pass
    stages += [langgraph_processor, tts, transport.output()]

    return Pipeline(stages)
