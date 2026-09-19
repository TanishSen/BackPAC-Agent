"""Request / response shapes for the agent's own HTTP surface."""

from pydantic import BaseModel, ConfigDict, Field


class StartRequest(BaseModel):
    """Body for `POST /start`, sent by BackPAC-BE after it mints the room."""

    model_config = ConfigDict(populate_by_name=True)

    room_name: str = Field(alias="roomName")
    session_id: str = Field(alias="sessionId")
    agent_id: str = Field(alias="agentId")


class StartResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    status: str = "running"
    session_id: str = Field(alias="sessionId")


class StopRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(alias="sessionId")
