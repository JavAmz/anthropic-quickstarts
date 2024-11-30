import asyncio
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from anthropic.types.beta import BetaMessageParam
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult

app = FastAPI()

CONFIG_DIR = os.path.expanduser("~/.anthropic")
API_KEY_FILE = os.path.join(CONFIG_DIR, "api_key")


class AuthRequest(BaseModel):
    api_key: str
    provider: APIProvider = APIProvider.ANTHROPIC


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    custom_system_prompt: str = ""
    only_n_most_recent_images: Optional[int] = 3


class State(BaseModel):
    messages: list[BetaMessageParam] = []
    api_key: str = ""
    provider: APIProvider = APIProvider.ANTHROPIC
    auth_validated: bool = False
    tools: dict = {}
    only_n_most_recent_images: int = 3
    custom_system_prompt: str = ""
    in_sampling_loop: bool = False


responses = {}


# In-memory state to replace `st.session_state`
state = State()

# Endpoint to set up state


@contextmanager
def track_sampling_loop():
    state.in_sampling_loop = True
    yield
    state.in_sampling_loop = False


@app.post("/setup_state")
async def setup_state(auth_request: AuthRequest):
    state.provider = auth_request.provider
    state.api_key = auth_request.api_key

    # Validate authentication
    if auth_request.provider == APIProvider.ANTHROPIC and not auth_request.api_key:
        raise HTTPException(
            status_code=400, detail="API key required for Anthropic provider")

    state.auth_validated = True
    return {"detail": "State setup complete"}

# Endpoint for reset state


@app.post("/reset")
async def reset_state():
    global state
    if state.in_sampling_loop:
        raise HTTPException(
            409, detail="Can not reset while the loop is running"
        )
    state = State()
    await setup_state(AuthRequest(provider=APIProvider.ANTHROPIC, api_key=""))
    for cmd in ["pkill Xvfb; pkill tint2", "./start_all.sh"]:
        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()
        await asyncio.sleep(1)

    return {"detail": "State reset complete"}

# Endpoint for chat messages


@app.post("/chat")
async def chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    # Append user message
    user_message = BetaMessageParam(
        content=chat_request.message,
        role="user",
    )

    state.messages.append(user_message)

    # Add background task for sampling loop
    background_tasks.add_task(run_sampling_loop, chat_request)
    return {"detail": "Message received", "messages": state.messages}


async def run_sampling_loop(chat_request: ChatRequest):
    state.in_sampling_loop = True
    try:
        # Run the agent sampling loop
        state.messages = await sampling_loop(
            system_prompt_suffix=chat_request.custom_system_prompt,
            model=chat_request.model or PROVIDER_TO_DEFAULT_MODEL_NAME[state.provider],
            provider=state.provider,
            messages=state.messages,
            output_callback=output_callback,
            tool_output_callback=store_tool_output,
            api_response_callback=lambda request, response, error: store_api_response(
                request, response, error),
            api_key=state.api_key,
            only_n_most_recent_images=chat_request.only_n_most_recent_images,
        )
    finally:
        state.in_sampling_loop = False


def output_callback(message):
    pass


def store_tool_output(tool_output: ToolResult, tool_id: str):
    state.tools[tool_id] = tool_output


def store_api_response(request, response, error):
    response_id = datetime.now().isoformat()
    responses[response_id] = (request, response)
    if error:
        responses[response_id] = {"error": str(error)}

# Endpoint to get the current state (for debugging or to see conversation history)


@app.get("/state")
async def get_state():
    return state
