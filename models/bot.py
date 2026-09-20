"""Request / response shapes for the agent's own HTTP surface."""

from pydantic import BaseModel, ConfigDict, Field


class StartRequest(BaseModel):
    """Body for `POST /start`, sent by BackPAC-BE after it mints the room."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(alias="roomName")
    session_id: str = Field(alias="sessionId")
    agent_id: str = Field(alias="agentId")
    #: The LangGraph thread to think in. Equal to the room name today, but its
    #: own field so the two can diverge later without a wire change on both
    #: sides at once. Defaults to the room name when the backend omits it.
    thread_id: str | None = Field(default=None, alias="threadId")
    #: True when this call continues an earlier conversation. The graph loads
    #: whatever state the thread has either way; this only decides whether to
    #: open with a greeting or pick up mid-sentence.
    is_resuming: bool = Field(default=False, alias="isResuming")


class StartResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    status: str = "running"
    session_id: str = Field(alias="sessionId")


class StopRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(alias="sessionId")


class SpokenLine(BaseModel):
    """One thing the orb can say.

    `levels` is one loudness value per `frameMs`, so the orb's mouth follows the
    actual waveform rather than a generic wobble — it is a few hundred bytes and
    travels inline.

    The audio does **not**: it is fetched from `/voice-line.wav?text=…`. Inlining
    it as base64 made the welcome set a 2.5 MB JSON body that no cache could
    reuse. As a separate file it is cached by the browser and the OS, so it is
    downloaded once per line, ever.
    """

    model_config = ConfigDict(populate_by_name=True)

    text: str
    levels: list[float]
    frame_ms: int = Field(alias="frameMs")


class WelcomeLinesResponse(BaseModel):
    """Everything the welcome screen might say, fetched in one go.

    The app holds these so poking the orb answers instantly instead of waiting
    on a request and a synthesis.
    """

    model_config = ConfigDict(populate_by_name=True)

    greeting: list[SpokenLine]
    poke: list[SpokenLine]
    idle: list[SpokenLine]
