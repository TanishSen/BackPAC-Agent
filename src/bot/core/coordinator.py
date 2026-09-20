"""Builds the multi-agent graph and routes each turn through it.

Shape:

    START ──► orchestrator ──(calls transfer_to_X)──► trains | flights | stays
                    │                                         │
              (no handoff: it                            (answers, then)
               answered itself)                                │
                    └───────────────► END ◄───────────────────┘

How routing works, and why it's cheap: the orchestrator doesn't "decide" in
prose. It calls a `transfer_to_*` tool and we read *which* tool it called. No
second LLM call, no parsing English.

Three decisions worth knowing about:

1. **Every turn re-enters at the orchestrator.** Travellers change subject
   ("...and somewhere to stay") far more often than they stay in one lane, and
   the orchestrator sees the whole history so it routes follow-ups correctly.
   One cheap Haiku call per turn buys that. `active_agent` is therefore
   this-turn routing, not sticky state.

2. **The orchestrator's handoff message never enters the transcript.** A
   handoff is an AIMessage whose only content is a tool call nobody answers.
   Keeping it would (a) break the specialist — `create_react_agent` rejects a
   history with an unanswered tool call — and (b) put routing plumbing in the
   transcript the app displays. So a routing turn contributes no message; only
   real speech does.

3. **Every node is async.** LangGraph runs a synchronous node on a thread from
   the default executor. Inside a realtime voice pipeline that pool is already
   contended — 50 audio frames a second, Silero VAD and end-of-turn inference
   all want it — and an LLM node measured over a minute of queueing before it
   made its first API call, while the caller sat listening to silence. Async
   nodes do their waiting as ordinary non-blocking I/O and start immediately.
   Keep it that way: one `.invoke()` in here reintroduces the stall.

This module contains no Pipecat: it's a pure LangGraph you can drive from a
terminal (`make brain`). The voice layer in processors/ drives it in a call.
"""

import logging
from datetime import date

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
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
from src.bot.prompts.global_constraints import today_line

logger = logging.getLogger(__name__)

# The specialists the orchestrator can hand off to. Add a row to grow the graph.
SPECIALISTS = {
    "trains": create_trains_agent,
    "flights": create_flights_agent,
    "stays": create_stays_agent,
}


def _make_handoff_tools() -> list:
    """One no-op tool per specialist. They are never executed — the orchestrator
    *calling* one is the routing signal, and we read the call directly."""
    tools = []
    for name in SPECIALISTS:

        @tool(f"transfer_to_{name}")
        def _handoff(reason: str = "", _name: str = name) -> str:
            """Hand the conversation to a specialist."""
            return _name

        tools.append(_handoff)
    return tools


def build_graph(checkpointer=None):
    """Compile the graph. Returns something with `.invoke` / `.astream_events`
    that the voice processor calls once per user turn.

    Called once per session, which also means the date injected below is fresh
    for every call.

    `checkpointer` is where the conversation's memory lives, keyed by
    thread_id. Pass an `AsyncPostgresSaver` (see `make_checkpointer`) and a
    conversation survives a restart, a redeploy and a second container, which
    is what makes "resume this chat" mean anything. Omit it and you get
    `MemorySaver`: fine for a terminal harness or a test, useless in
    production, because the state lives in one process's RAM and dies with it.
    """
    specialists = {name: factory() for name, factory in SPECIALISTS.items()}
    orchestrator = get_llm().bind_tools(_make_handoff_tools())

    # Today's date, resolved when the graph is built. Without it the model has
    # no way to turn "next Friday" or "the 2nd of October" into the YYYY-MM-DD
    # the search tools require, and it will quietly guess a year.
    system = SystemMessage(content=f"{today_line()}\n{ORCHESTRATOR_PROMPT}")

    async def orchestrator_node(state: TripState) -> dict:
        reply = await orchestrator.ainvoke([system, *state["messages"]])

        target = None
        for call in getattr(reply, "tool_calls", []) or []:
            if call["name"].startswith("transfer_to_"):
                target = call["name"].removeprefix("transfer_to_")
                break

        if target in SPECIALISTS:
            logger.info("routing to %s", target)
            # Routing only — see note 2 in the module docstring. No message.
            return {"active_agent": target}

        # No handoff: the orchestrator answered the traveller itself (a greeting,
        # or a question it needed to ask). That IS speech, so it goes in.
        return {"messages": [reply], "active_agent": "orchestrator"}

    def make_specialist_node(name: str):
        agent = specialists[name]

        async def node(state: TripState) -> dict:
            history = state["messages"]
            result = await agent.ainvoke({"messages": history})
            # A react agent returns the whole conversation back. Only the tail is
            # new; returning the head as well would re-send messages the reducer
            # already holds.
            produced = result["messages"][len(history):]
            return {"messages": produced, "active_agent": "orchestrator"}

        return node

    graph = StateGraph(TripState)
    graph.add_node("orchestrator", orchestrator_node)
    for name in SPECIALISTS:
        graph.add_node(name, make_specialist_node(name))

    graph.add_edge(START, "orchestrator")

    # After the orchestrator: to the chosen specialist, or end if it just spoke.
    graph.add_conditional_edges(
        "orchestrator",
        lambda s: s["active_agent"] if s["active_agent"] in SPECIALISTS else END,
        {**{n: n for n in SPECIALISTS}, END: END},
    )
    for name in SPECIALISTS:
        graph.add_edge(name, END)

    # The checkpointer keeps each room's conversation across turns, keyed by
    # thread_id (the LiveKit room name — see LangGraphProcessor).
    return graph.compile(checkpointer=checkpointer or MemorySaver())
