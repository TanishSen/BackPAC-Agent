"""The specialist agents: trains, flights, stays.

Each is a prebuilt LangGraph "react" agent — a loop of (call the model, run any
tools it asked for, repeat until it answers). We use `create_react_agent`
because every specialist is the same shape: one model, a set of tools, a system
prompt. ajimganj hand-builds each as a StateGraph; react agents are the shorter
way to the same thing and are the right default until an agent needs custom
control flow.

To ADD a specialist (say "cabs"): write a prompt, add a tool in core/tools.py,
add one `create_*_agent` here, and register it in coordinator.py. That's it.
See AGENT_GUIDE.md.
"""

from langgraph.prebuilt import create_react_agent

from src.bot.core.llm import get_llm, get_llm_heavy
from src.bot.core.tools import search_flights, search_stays, search_trains
from src.bot.prompts.flights_prompt import FLIGHTS_PROMPT
from src.bot.prompts.global_constraints import GLOBAL_CONSTRAINTS
from src.bot.prompts.stays_prompt import STAYS_PROMPT
from src.bot.prompts.trains_prompt import TRAINS_PROMPT


def _prompt(specific: str) -> str:
    # Global spoken-output rules first, then the specialist's own brief.
    return f"{GLOBAL_CONSTRAINTS}\n\n{specific}"


def create_trains_agent():
    return create_react_agent(
        get_llm(), tools=[search_trains], prompt=_prompt(TRAINS_PROMPT)
    )


def create_flights_agent():
    return create_react_agent(
        get_llm(), tools=[search_flights], prompt=_prompt(FLIGHTS_PROMPT)
    )


def create_stays_agent():
    # Stays involve more trade-offs (area vs price vs rating) — worth the
    # stronger model. Swap to get_llm() if latency matters more than nuance.
    return create_react_agent(
        get_llm_heavy(), tools=[search_stays], prompt=_prompt(STAYS_PROMPT)
    )
