"""Talk to the trip brain in your terminal — no microphone, no LiveKit, no TTS.

This is the fastest way to develop and debug the conversation. It runs the exact
same LangGraph the voice agent runs; only the voice plumbing is skipped. If a
reply reads well here, it will sound right in a real call.

    make brain          (or:  python -m scripts.chat_brain)

It reads the same .env the voice agent reads, so there is nothing to export.
Type your side of the conversation; Ctrl-C to quit.

Tool calls are printed as they happen, so you can see when the brain actually
searched and what came back — start BackPAC-BE first or every search will come
back as an error.
"""

import asyncio

from config.env import load_env, require

load_env()
require("ANTHROPIC_API_KEY")

from src.bot.core.coordinator import build_graph  # noqa: E402


async def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "terminal"}}
    print("Trip brain ready. Say something (Ctrl-C to quit).\n")

    while True:
        try:
            user = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not user:
            continue

        # Stream tokens so it feels like the real voice agent.
        print("bot  > ", end="", flush=True)
        async for event in graph.astream_events(
            {
                "messages": [{"role": "user", "content": user}],
                "active_agent": "orchestrator",
                "room_name": "terminal",
            },
            config=config,
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, str):
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    # Some responses stream a list of content blocks.
                    print(
                        "".join(
                            b.get("text", "") for b in content if isinstance(b, dict)
                        ),
                        end="",
                        flush=True,
                    )
            elif kind == "on_tool_end":
                name = event.get("name", "")
                # transfer_to_* are internal routing, not searches — skip them.
                if not name.startswith("transfer_to_"):
                    print(f"\n     [{name} -> {event['data'].get('output')}]")
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
