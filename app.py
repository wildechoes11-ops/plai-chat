```python id="r5m2qp"
import os
import json
import uuid
import time
import asyncio
import traceback

from quart import Quart, request, Response, jsonify
import httpx

from plai_chat import ChatbotChatApp
from plai_solver import _solve

COOKIE_FILE = "plai_cookies.json"

FREE_MODELS = {
    "nano": "nvidia/nemotron-3-nano-30b-a3b:free",
    "super": "nvidia/nemotron-3-super-120b-a12b:free",
    "vision": "nvidia/nemotron-nano-12b-v2-vl:free"
}

COOKIE_LIFETIME = 60 * 60 * 6
REFRESH_BEFORE = 60 * 10

cookie_created_at = 0

chat_app = None

app = Quart(__name__)


# =========================
# COOKIE HELPERS
# =========================

def cookies_expired():

    global cookie_created_at

    if not os.path.exists(COOKIE_FILE):
        return True

    age = time.time() - cookie_created_at

    return age >= (
        COOKIE_LIFETIME - REFRESH_BEFORE
    )


def save_cookies(cookies):

    global cookie_created_at

    cookie_created_at = time.time()

    with open(COOKIE_FILE, "w") as f:

        json.dump({
            "cookies": cookies,
            "created_at": cookie_created_at
        }, f)


def load_cookie_timestamp():

    global cookie_created_at

    if not os.path.exists(COOKIE_FILE):
        return

    try:

        with open(COOKIE_FILE, "r") as f:
            data = json.load(f)

        cookie_created_at = data.get(
            "created_at",
            0
        )

    except:
        cookie_created_at = 0


# =========================
# INIT
# =========================

async def initialize_app(
    force_refresh=False
):

    global chat_app

    try:

        print(
            "\n========== INITIALIZING ==========",
            flush=True
        )

        load_cookie_timestamp()

        # =========================
        # EXISTING COOKIES
        # =========================

        if (
            os.path.exists(COOKIE_FILE)
            and not cookies_expired()
            and not force_refresh
        ):

            print(
                "Using existing cookies...",
                flush=True
            )

            chat_app = (
                ChatbotChatApp
                .from_cookie_file(
                    COOKIE_FILE
                )
            )

            print(
                "Loaded existing session",
                flush=True
            )

            return

        # =========================
        # REFRESH COOKIES
        # =========================

        print(
            "\n========== REFRESHING COOKIES ==========",
            flush=True
        )

        print(
            "Calling solver...",
            flush=True
        )

        token = await _solve(
            "https://plai.chat",
            "0x4AAAAAAC-lISmo_bU02UL9"
        )

        print(
            "Solver finished",
            flush=True
        )

        if not token:

            raise Exception(
                "Solver returned empty token"
            )

        print(
            "Opening HTTP client...",
            flush=True
        )

        with httpx.Client(
            timeout=120
        ) as cx:

            print(
                "Sending verify request...",
                flush=True
            )

            verify = cx.post(
                "https://plai.chat/api/web/auth/anonymous-verify",
                json={
                    "turnstileToken": token
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64)"
                    ),
                    "Origin": "https://plai.chat",
                    "Referer": "https://plai.chat/"
                }
            )

            print(
                "Verify status:",
                verify.status_code,
                flush=True
            )

            print(
                "Verify body:",
                verify.text[:300],
                flush=True
            )

            if verify.status_code != 200:

                raise Exception(
                    f"Verification failed: "
                    f"{verify.status_code}"
                )

            cookies = {
                k: v
                for k, v
                in cx.cookies.items()
            }

        print(
            "Cookies received:",
            cookies,
            flush=True
        )

        if "web_anon_session" not in cookies:

            raise Exception(
                "web_anon_session missing"
            )

        save_cookies(cookies)

        print(
            "Creating chat app...",
            flush=True
        )

        chat_app = (
            ChatbotChatApp
            .from_cookie_file(
                COOKIE_FILE
            )
        )

        print(
            "SUCCESSFULLY INITIALIZED",
            flush=True
        )

    except Exception as e:

        print(
            "\n========== INIT ERROR ==========",
            flush=True
        )

        print(
            "ERROR:",
            str(e),
            flush=True
        )

        traceback.print_exc()

        chat_app = None


# =========================
# ENSURE APP
# =========================

async def ensure_app():

    global chat_app

    if chat_app is None:

        await initialize_app()

    if chat_app is None:

        raise Exception(
            "Failed to initialize app"
        )


# =========================
# BACKGROUND REFRESH LOOP
# =========================

async def cookie_refresh_loop():

    while True:

        try:

            await asyncio.sleep(300)

            if cookies_expired():

                print(
                    "\n========== AUTO REFRESH ==========",
                    flush=True
                )

                await initialize_app(
                    force_refresh=True
                )

        except Exception:

            print(
                "\n========== REFRESH LOOP ERROR ==========",
                flush=True
            )

            traceback.print_exc()


# =========================
# STARTUP
# =========================

@app.before_serving
async def startup():

    print(
        "\n========== SERVER STARTUP ==========",
        flush=True
    )

    # Run init in background
    asyncio.create_task(
        initialize_app()
    )

    # Background refresh loop
    asyncio.create_task(
        cookie_refresh_loop()
    )

    print(
        "Startup tasks launched",
        flush=True
    )


# =========================
# MODEL HELPER
# =========================

def normalize_model(model: str):

    if model in FREE_MODELS:
        return FREE_MODELS[model]

    if model in FREE_MODELS.values():
        return model

    return FREE_MODELS["nano"]


# =========================
# ROUTES
# =========================

@app.route("/")
async def home():

    return {
        "status": "running",
        "models": list(
            FREE_MODELS.values()
        )
    }


@app.route("/v1/models")
async def models():

    data = []

    for model in FREE_MODELS.values():

        data.append({
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "plai"
        })

    return {
        "object": "list",
        "data": data
    }


@app.route(
    "/v1/chat/completions",
    methods=["POST"]
)
async def chat_completions():

    await ensure_app()

    body = await request.get_json()

    model = normalize_model(
        body.get("model", "nano")
    )

    stream = body.get(
        "stream",
        False
    )

    messages = body.get(
        "messages",
        []
    )

    if not messages:

        return jsonify({
            "error": "messages required"
        }), 400

    api_history = []

    for msg in messages[:-1]:

        role = msg.get("role")
        content = msg.get("content")

        if role and content:

            api_history.append({
                "role": role,
                "content": content
            })

    latest = messages[-1]["content"]

    completion_id = (
        f"chatcmpl-{uuid.uuid4().hex}"
    )

    # =========================
    # STREAMING
    # =========================

    if stream:

        async def generate():

            try:

                gen = chat_app.send_message(
                    latest,
                    model=model,
                    history=api_history
                )

                while True:

                    try:
                        chunk = next(gen)

                    except StopIteration:
                        break

                    payload = {
                        "id": completion_id,
                        "object": (
                            "chat.completion.chunk"
                        ),
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }

                    yield (
                        "data: "
                        + json.dumps(payload)
                        + "\n\n"
                    )

                    await asyncio.sleep(0)

                done_payload = {
                    "id": completion_id,
                    "object": (
                        "chat.completion.chunk"
                    ),
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }

                yield (
                    "data: "
                    + json.dumps(done_payload)
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

            except Exception as e:

                traceback.print_exc()

                yield (
                    "data: "
                    + json.dumps({
                        "error": str(e)
                    })
                    + "\n\n"
                )

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # =========================
    # NORMAL RESPONSE
    # =========================

    response_text = ""

    gen = chat_app.send_message(
        latest,
        model=model,
        history=api_history
    )

    while True:

        try:
            chunk = next(gen)

        except StopIteration:
            break

        response_text += chunk

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }]
    }


# =========================
# START
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )
```
