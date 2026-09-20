"""Loads `.env` and fails fast on missing credentials.

Import this **first**, before anything that reads `os.getenv`:

    from config.env import load_env
    load_env()

Why a module instead of a bare `load_dotenv()` call: `load_dotenv()` with no
argument searches upward from *the file that called it*, so it silently finds
nothing when the caller lives somewhere else (a script in scripts/, a test, a
notebook). That failure is invisible — the key is just absent, and you get an
authentication error three layers deep in LangChain instead of "you forgot the
key". Here the path is pinned to the project root, so it works from anywhere.

`require()` turns the same class of problem into one clear line at startup.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# config/env.py -> config/ -> the agent root, where .env lives.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

_loaded = False


def load_env() -> None:
    """Load `.env` from the project root. Safe to call more than once."""
    global _loaded
    if _loaded:
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        # Normal in a container, where .dockerignore keeps .env out of the
        # image and compose supplies the values instead. Worth saying either
        # way, but not as a warning that looks like something is broken —
        # `require()` below is what actually catches a missing key.
        logger.info(
            "no .env at %s; using the ambient environment "
            "(expected in Docker, where compose supplies the values)",
            ENV_FILE,
        )
    _loaded = True


def require(*names: str) -> None:
    """Exit with one readable message if any of `names` is unset or empty.

    Called at startup rather than at first use, so a misconfigured deployment
    dies immediately and says why, instead of accepting a call and failing
    mid-conversation with a provider error.
    """
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + f"\nSet them in {ENV_FILE} (see .env.example)."
        )


def voice_requirements() -> tuple[str, ...]:
    """What a *voice call* needs on top of the brain.

    STT_MODE=device means the phone transcribes and posts text, so the Azure
    keys aren't needed; STT_MODE=azure means they are.
    """
    base = (
        "ANTHROPIC_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
    )
    if os.getenv("STT_MODE", "azure").lower() == "azure":
        return base + ("AZURE_SPEECH_API_KEY", "AZURE_SPEECH_REGION")
    return base
