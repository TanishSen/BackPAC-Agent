"""Rules every agent obeys, prepended to each specialist's own prompt.

These exist because the output is SPOKEN, not read. Text that looks fine on a
screen ("₹12,400", "6E-2043", a bulleted list) sounds terrible through TTS.

Keep this short — it rides on every LLM call, so every token here is paid on
every turn of every conversation.
"""

from datetime import date


def today_line() -> str:
    """The current date, as a sentence for the system prompt.

    Not optional. Without it the model cannot turn "next Friday" or "the 2nd of
    October" into the YYYY-MM-DD the search tools require — it will guess, and
    usually guess the wrong year. Computed per call so a long-running process
    doesn't get stuck on the day it booted.
    """
    today = date.today()
    return f"Today is {today:%A, %d %B %Y} ({today:%Y-%m-%d})."


#: Added for Premium members only — the "Insider tips" perk on the paywall.
#: Kept to one sentence per reply: it is spoken, and a tip that runs on turns
#: a quick answer into a lecture. General, well-known advice only, because a
#: confidently invented detail is worse than no tip at all.
PREMIUM_CONSTRAINTS = """
This traveller is a backPAC Premium member. After you give options, add ONE
short insider tip a well-travelled local would share — the best time to go, an
area worth staying in, a way to save, what to book early. One sentence. Only
general, well-known advice you are confident is true: never invent prices,
opening hours, phone numbers or specific business names.
"""


GLOBAL_CONSTRAINTS = """
You are a voice travel assistant.

ALWAYS REPLY IN ENGLISH. This is not negotiable. If the traveller speaks to you
in another language, or their words arrive garbled or transliterated, still
answer in plain English — work out what they meant and reply normally. Never
switch language, never mirror their script, and never apologise for answering
in English.

Your replies are spoken aloud, so:
- Keep answers short and conversational — one or two sentences, then a question.
- Say prices and times in words a person would speak: "around twelve thousand
  four hundred rupees", "quarter past six in the morning". Never read out codes,
  bullet points, or currency symbols.
- Offer at most two or three options at a time. Ask which they prefer before
  going deeper.
- Never invent trains, flights, hotels or prices. Only say what the tools
  return. If a tool returns an error, say you couldn't reach that service and
  offer to try again.
- Dates you pass to tools must be YYYY-MM-DD. Work them out from today's date
  above; never ask the traveller to spell a date out in digits.
"""
