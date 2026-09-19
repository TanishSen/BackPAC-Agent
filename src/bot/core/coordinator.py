"""Builds the multi-agent graph and routes turns through it.

Shape (same idea as the ajimganj concierge, trimmed to three specialists):

    START
      └─► orchestrator ──(picks a specialist)──► trains | flights | stays
                                                      │
                                                 (answers the user, or
                                                  hands control back)
                                                      ▼
                                                    END (waits for next turn)

How routing works, and why it's cheap: the orchestrator doesn't "decide" in
prose. It calls a `transfer_to_*` tool, and we read which tool it called to
pick the next node. No second LLM call to route. `active_agent` is saved in
state so the *next* user turn skips the orchestrator and resumes with the same
specialist — the checkpointer (MemorySaver) makes that survive across turns,
keyed by the LiveKit room.

This module has no Pipecat in it on purpose: it's a pure LangGraph you could
unit-test from a script. The voice layer (processors/) drives it.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.bot.agents.specialists import (
    create_flights_agent,
    create_stays_agent,
    create_trains_agent,
)
from src.bot.core.llm import get_llm
from src.bot.core.state import TripState
from src.bot.prompts.orchestrator_prompt import ORCHESTRATOR_PROMPT

logger = logging.getLogger(__name__)

# The specialists the orchestrator can hand off to. Add a row to grow the graph.
SPECIALISTS = {
    "trains": create_trains_agent,
    "flights": create_flights_agent,
    "stays": create_stays_agent,
}


def build_graph():
    """Compile the graph once per session. Returns something with `.invoke` /
    `.astream` that the voice processor calls each user turn."""
    specialists = {name: factory() for name, factory in SPECIALISTS.items()}

    # The orchestrator is the model bound to one handoff tool per specialist.
    # The tools do nothing but exist to be *called* — we inspect the call.
    from langchain_core.tools import tool

    handoffs = []
    for name in SPECIALISTS:
        @tool(f"transfer_to_{name}")
        def _handoff(reason: str = "", _name: str = name) -> str:
            """Hand the conversation to a specialist."""
            return _name

        handoffs.append(_handoff)

    orchestrator = get_llm().bind_tools(handoffs)

    def orchestrator_node(state: TripState) -> dict:
        from langchain_core.messages import SystemMessage

        reply = orchestrator.invoke(
            [SystemMessage(content=ORCHESTRATOR_PROMPT), *state["messages"]]
        )
        # Which specialist did it choose? Read the tool call, don't parse prose.
        target = "orchestrator"
        for call in getattr(reply, "tool_calls", []) or []:
            if call["name"].startswith("transfer_to_"):
                target = call["name"].removeprefix("transfer_to_")
        return {"messages": [reply], "active_agent": target}

    def make_specialist_node(name: str):
        agent = specialists[name]

        def node(state: TripState) -> dict:
            result = agent.invoke({"messages": state["messages"]})
            # Specialist answered; hand control back to the desk for next turn.
            return {"messages": result["messages"], "active_agent": "orchestrator"}

        return node

    graph = StateGraph(TripState)
    graph.add_node("orchestrator", orchestrator_node)
    for name in SPECIALISTS:
        graph.add_node(name, make_specialist_node(name))

    # Entry: resume with whoever held the conversation last turn; first turn
    # (active_agent unset) starts at the orchestrator.
    def entry(state: TripState) -> str:
        current = state.get("active_agent", "orchestrator")
        return current if current in SPECIALISTS else "orchestrator"

    graph.add_conditional_edges(START, entry,
                                {**{n: n for n in SPECIALISTS}, "orchestrator": "orchestrator"})

    # After the orchestrator, go to the chosen specialist (or end if it just
    # talked without handing off).
    graph.add_conditional_edges(
        "orchestrator",
        lambda s: s["active_agent"] if s["active_agent"] in SPECIALISTS else END,
        {**{n: n for n in SPECIALISTS}, END: END},
    )
    for name in SPECIALISTS:
        graph.add_edge(name, END)

    # MemorySaver keeps each room's conversation in memory across turns. Swap for
    # a Redis/Postgres checkpointer to survive a restart (see AGENT_GUIDE.md).
    return graph.compile(checkpointer=MemorySaver())
