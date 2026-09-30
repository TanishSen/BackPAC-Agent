"""The Claude models the agents run on.

Two tiers, exactly as the ajimganj concierge does it:
  - Haiku  — fast + cheap, for routing and the simple specialist agents.
  - Sonnet — for anything that has to reason hard (multi-stop itineraries,
    price/constraint trade-offs).

The API key is read from the environment (`ANTHROPIC_API_KEY`) and never
written here. Whoever deploys this supplies the key; this file doesn't care
whose it is.
"""

import os

from langchain_anthropic import ChatAnthropic

#: Seconds without a byte before a Claude request is abandoned. langchain passes
#: `timeout=None` to the SDK unless told otherwise, which disables the SDK's own
#: default — so a hung stream was silence on the line with no end. Applies per
#: read, so a long streamed answer is unaffected; only a stalled one is cut.
TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT", "30"))
#: One retry covers a transient 529/5xx without doubling a caller's wait twice.
MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "1"))

# One id, one place. If Anthropic ships a newer Haiku, change it here.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"


def get_llm(temperature: float = 0, streaming: bool = True) -> ChatAnthropic:
    """Fast model — the default for routing and simple agents."""
    return ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name=HAIKU_MODEL,
        temperature=temperature,
        streaming=streaming,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def get_llm_heavy(temperature: float = 0, streaming: bool = True) -> ChatAnthropic:
    """Slower, stronger model — for agents that plan or weigh options."""
    return ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name=SONNET_MODEL,
        temperature=temperature,
        streaming=streaming,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
