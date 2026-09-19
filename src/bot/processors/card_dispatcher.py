from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from .langgraph_processor import ToolResultFrame

class CardDispatcherProcessor(FrameProcessor):
    """
    Intercepts ToolResultMessage frames and converts them into 
    RTVIServerMessageFrame payloads for the frontend to render rich UI cards.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, ToolResultFrame):
            # Transform the LangGraph tool result into an RTVI Server Message
            rtvi_frame = RTVIServerMessageFrame(
                data={
                    "type": "agent-card",
                    "cardType": frame.card_type,
                    "payload": frame.result,
                }
            )
            await self.push_frame(rtvi_frame, direction)

        # Always forward the original frame downstream
        await self.push_frame(frame, direction)
