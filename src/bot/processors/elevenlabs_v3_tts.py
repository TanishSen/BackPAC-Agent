"""ElevenLabs TTS, patched for the eleven_v3 model.

Ported verbatim from the ajimganj concierge — this is the one non-obvious piece
of the voice stack. `eleven_v3` rejects the `previous_text` field that Pipecat
normally sends for prosody continuity, so we clear it before every call. The
trailing space after sentence-final punctuation is ajimganj's fix for the model
clipping the last word.

Nothing here holds a key: the key is passed in when the service is constructed
(see pipeline.py), read from ELEVENLABS_API_KEY there.
"""

from typing import AsyncGenerator

from pipecat.frames.frames import Frame
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService


class ElevenLabsV3TTSService(ElevenLabsHttpTTSService):
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self._previous_text = ""
        if text and text[-1] in ".?!":
            text = text + " "
        async for frame in super().run_tts(text, context_id):
            yield frame
