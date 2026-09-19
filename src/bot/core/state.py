"""The shared state that flows through the multi-agent graph.

Every node reads and writes this one dict-like object. `messages` is the
conversation so far (LangGraph appends to it automatically via `add_messages`);
`active_agent` is which specialist currently holds the conversation, so the
next user turn resumes at the right place instead of always re-routing through
the orchestrator.

Add a field here when a piece of information must survive across nodes/turns
(e.g. a chosen origin city). Keep it small — this is hot state, not a database.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class TripState(TypedDict):
    # The running transcript. `add_messages` merges new messages in rather than
    # overwriting, so each node can just return {"messages": [reply]}.
    messages: Annotated[list, add_messages]

    # Which specialist is in control. The orchestrator sets it when it hands
    # off; a specialist clears it back to "orchestrator" when its job is done.
    active_agent: str

    # Room id / conversation id, so tools can scope their calls if needed.
    room_name: str
