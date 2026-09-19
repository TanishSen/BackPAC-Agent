# BackPAC agent — common tasks.
.PHONY: install run brain call
install:            ## install deps into .venv (uv)
	uv venv --python 3.12 .venv && uv pip install -r requirements.txt

run:                ## run the agent service on :8080
	.venv/bin/python main.py

brain:              ## talk to the trip brain in the terminal (no voice, no LiveKit)
	.venv/bin/python -m scripts.chat_brain

call:               ## a full voice call with no phone — the acceptance test.
                    ## Needs BackPAC-BE running and this agent running.
	.venv/bin/python -m scripts.call_agent
