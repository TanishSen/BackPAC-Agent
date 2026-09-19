_MARKDOWN_CHARS = frozenset("*_`#")


def sanitize_tts_text(text: str) -> str:
    """Remove HTML/SSML and stray markdown from TTS text."""
    output: list[str] = []
    in_angle_tag = False

    for ch in text or "":
        if in_angle_tag:
            if ch == ">":
                in_angle_tag = False
            continue
        if ch == "<":
            in_angle_tag = True
            continue
        if ch in _MARKDOWN_CHARS:
            continue
        output.append(ch)

    return "".join(output)


def has_speakable_tts_text(text: str) -> bool:
    """True when text has spoken content outside ElevenLabs delivery tags."""
    in_audio_tag = False

    for ch in sanitize_tts_text(text):
        if in_audio_tag:
            if ch == "]":
                in_audio_tag = False
            continue
        if ch == "[":
            in_audio_tag = True
            continue
        if not ch.isspace():
            return True

    return False
