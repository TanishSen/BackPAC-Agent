# BackPAC agent — the Pipecat voice pipeline and the Claude trip brain.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries this image genuinely needs:
#   libssl3, ca-certificates — the Azure Speech SDK links against OpenSSL and
#                              will not start without a cert store.
#   libasound2              — the same SDK links ALSA even when, as here, it
#                              only ever reads from a memory stream.
#   git                     — some pipecat extras install from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libasound2 \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 backpac && chown -R backpac:backpac /app
USER backpac

EXPOSE 8080

# "/" reports the process and how many calls it is carrying. It deliberately
# does not probe LiveKit or Claude — a health check that fans out is a health
# check that flaps and restarts a healthy process mid-call.
# A generous start period and timeout on purpose: this process imports
# Pipecat and loads the Silero VAD and end-of-turn ONNX models before it binds,
# and once it is up a single-threaded asyncio loop is also carrying live calls.
# Twenty seconds and a three-second timeout marked a perfectly healthy agent
# unhealthy on a cold start, which in an orchestrator means being restarted
# mid-call.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/', timeout=2).status==200 else 1)"

# One worker on purpose. Sessions live in this process's memory (`running` in
# main.py), so a second worker would take /stop calls for calls it has never
# heard of. Scale by running more containers, not more workers — each call is
# independent, and the LiveKit room is what ties a caller to their agent.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
