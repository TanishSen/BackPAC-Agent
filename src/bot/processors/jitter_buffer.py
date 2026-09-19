"""Prebuffer chunked HTTPS TTS audio before transport playout."""

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TTSAudioJitterBuffer(FrameProcessor):
    """Hold the first ~200ms of TTS audio, then pass frames through."""

    def __init__(self, *, target_ms: int = 200):
        super().__init__()
        target_ms = max(150, min(target_ms, 500))
        self._target_secs = target_ms / 1000.0
        self._pending: list[TTSAudioRawFrame] = []
        self._pending_secs = 0.0
        self._released = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSStartedFrame, InterruptionFrame, CancelFrame)):
            self._clear()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSAudioRawFrame):
            if self._released:
                await self.push_frame(frame, direction)
                return

            self._pending.append(frame)
            self._pending_secs += self._duration_secs(frame)
            if self._pending_secs >= self._target_secs:
                self._released = True
                await self._emit_pending(direction)
            return

        if isinstance(frame, TTSStoppedFrame):
            await self._emit_pending(direction)
            self._clear()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _emit_pending(self, direction: FrameDirection):
        for pending in self._pending:
            await self.push_frame(pending, direction)
        self._pending.clear()
        self._pending_secs = 0.0

    def _clear(self):
        self._pending.clear()
        self._pending_secs = 0.0
        self._released = False

    @staticmethod
    def _duration_secs(frame: TTSAudioRawFrame) -> float:
        audio = frame.audio
        sample_rate = frame.sample_rate or 24000
        channels = frame.num_channels or 1
        return len(audio) / (sample_rate * channels * 2)
