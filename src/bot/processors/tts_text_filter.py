from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSSpeakFrame,
)

from .tts_text_sanitize import has_speakable_tts_text, sanitize_tts_text


class TTSTextFilter(FrameProcessor):
    """Sanitises text frames before they reach the ElevenLabs Eleven v3 TTS service.

    Strips any HTML/SSML tags the LLM accidentally emits (v3 has no SSML support)
    and removes stray markdown characters. Square-bracket Eleven v3 audio tags
    ([warmly], [short pause], [laughs], [curious], etc.) are buffered until they
    can be sent with spoken text, avoiding ElevenLabs' empty-input validation.
    """

    def __init__(self):
        super().__init__()
        self._pending_prefix = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame, CancelFrame)):
            self._pending_prefix = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TextFrame, TTSSpeakFrame)):
            cleaned = sanitize_tts_text(frame.text)
            text = self._pending_prefix + cleaned
            if not has_speakable_tts_text(text):
                self._pending_prefix = text
                return
            self._pending_prefix = ""
            cleaned = text
            if cleaned != frame.text:
                if isinstance(frame, TTSSpeakFrame):
                    frame = TTSSpeakFrame(
                        cleaned,
                        append_to_context=getattr(frame, "append_to_context", True),
                    )
                else:
                    frame = TextFrame(text=cleaned)

        if isinstance(frame, LLMFullResponseEndFrame):
            self._pending_prefix = ""

        await self.push_frame(frame, direction)
