"""Naming a conversation, once it is over.

The history list needs a label a person can scan — "Goa in December", "Trains
to Varanasi". Until now that label was the user's first sentence, truncated,
which gives you "I need an overnight train to Varanasi arriving bef…" across
every row: technically the right conversation, practically unreadable.

So when a call ends, Haiku reads the first few turns and writes a title. One
call, a handful of tokens, once per conversation — cheap enough not to think
about, and it happens after the line is dead so it costs the user nothing.

**Why only the first few turns.** A title describes what someone came for, and
they say that at the start. Feeding in a twenty-minute transcript costs more
and produces worse titles, because the model starts summarising the ending
("Confirming the booking") rather than the subject.

**It is allowed to fail.** No title is a perfectly good outcome — the
placeholder from the user's first sentence is already there and already
truthful. Nothing here raises.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.bot.core.llm import get_llm

logger = logging.getLogger(__name__)

#: How many turns to show the model. Four is a question and an answer, twice —
#: enough to know the destination and the mode.
TURNS_USED = 4

#: Below this, there is nothing to name. A call where someone said "hello" and
#: hung up should keep its placeholder rather than get an invented title.
MIN_TURNS = 2

MAX_CHARS = 42

_PROMPT = SystemMessage(
    content=(
        "You name travel-planning conversations for a history list.\n\n"
        "Reply with the title and nothing else. No quotes, no full stop, no "
        "preamble.\n\n"
        "Rules:\n"
        "- At most 5 words.\n"
        "- Name the destination if there is one, and the kind of trip: "
        "'Trains to Varanasi', 'Goa in December', 'Hotel in Jaipur'.\n"
        "- Title case, as a heading would be.\n"
        "- Describe what they wanted, not what happened. Not 'Booking "
        "Confirmed', not 'Flight Search'.\n"
        "- If the conversation never settled on anything, say what was "
        "discussed: 'Weekend Trip Ideas'."
    )
)


async def name_conversation(turns: list[tuple[str, str]]) -> str | None:
    """Return a short title for these (role, text) turns, or None.

    None whenever a title would be a guess — too little was said, the model
    was unreachable, or it answered with something that is not a title.
    """
    usable = [(r, t.strip()) for r, t in turns if t and t.strip()]
    if len(usable) < MIN_TURNS:
        logger.info("not enough was said to name this conversation")
        return None

    transcript = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}"
        for role, text in usable[:TURNS_USED]
    )

    try:
        # streaming off: we want one short string, and streaming a five-word
        # answer is pure overhead.
        llm = get_llm(temperature=0, streaming=False)
        reply = await llm.ainvoke(
            [_PROMPT, HumanMessage(content=transcript)]
        )
        return _clean(str(reply.content))
    except Exception as exc:  # noqa: BLE001 — a missing title is not a failure
        logger.warning("could not name the conversation: %s", exc)
        return None


def _clean(raw: str) -> str | None:
    """Take the model at its best and ignore the rest.

    Models are chatty in predictable ways — wrapping the answer in quotes,
    prefixing "Title:", adding a trailing full stop, or occasionally returning
    a whole sentence when asked for five words. Stripping the first three is
    trivial; the last is caught by the length check, which drops the answer
    rather than putting a paragraph in the history list.
    """
    text = " ".join(raw.split())
    text = re.sub(r"^(title|suggested title)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip("\"'“”").rstrip(".").strip()

    if not text or len(text) > MAX_CHARS:
        logger.info("ignoring an unusable title: %r", text[:80])
        return None
    return _ease_title_case(text)


#: Words that stay lowercase inside a title. Asking the model for this in the
#: prompt works most of the time, which is exactly the problem: "Goa In August"
#: turning up one row in five looks like a bug rather than a style. Doing it
#: here is free and always right.
_SMALL = frozenset(
    "a an and as at but by for from in into nor of on or the to via with".split()
)


def _ease_title_case(text: str) -> str:
    """Lowercase the small words, except the first.

    Only touches words the model capitalised as a whole word — "In", "To". A
    word already lowercase is left alone, and anything with inner capitals or
    digits (IndiGo, 6E-2043, AC) is never touched, because those are names.
    """
    words = text.split()
    out: list[str] = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _SMALL and w[:1].isupper() and w[1:].islower():
            out.append(w.lower())
        else:
            out.append(w)
    return " ".join(out)
