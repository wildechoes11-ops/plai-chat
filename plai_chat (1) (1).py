"""
PLAI.chat unofficial Python client.
=========================================

Reverse-engineered from https://plai.chat/ (JS bundle in /js/*.js,
particularly chat.js, auth.js and models.js — see ``plai_chat_findings.md``
for a full write-up of the network protocol).

Quick reference of what we discovered
-------------------------------------
* Single chat endpoint:   ``POST https://plai.chat/api/web/chat/send``
* Request body (JSON)::

      {
        "message":  "<the user prompt>",
        "history":  [ {"role": "user"|"assistant", "content": "..."}, ... ],
        "model":    "free" | "balanced" | "premium" | "@gpt" | "openai/gpt-5.5" | ...,
        "attachments":           [],
        "conversationStartedAt": "2026-01-01T00:00:00.000Z",
        "zdr":                   false
      }

* Response is **Server-Sent Events** (one ``data: {...}\\n`` line per
  event, blank line between events).  Event payloads we care about::

      {"type":"start",   "modelPrefix":"openai/gpt-5.5"}
      {"type":"model",   "model":"openai/gpt-5.5"}
      {"type":"content", "text":"Hello world"}        # cumulative text
      {"type":"images",  "images":[{"url":"..."}]}
      {"type":"info",    "message":"..."}
      {"type":"usage",   "balance":12.34}
      {"type":"error",   "error":"..."}
      {"type":"done",    "model":"..."}

* Anti-bot:  the server only accepts requests that carry a
  ``web_anon_session`` cookie (or, for paying users, a magic-link login
  cookie).  The cookie is minted by the server in exchange for a
  Cloudflare Turnstile token which the page solves invisibly via the
  WASM challenge bundled at
  ``challenges.cloudflare.com/turnstile/v0/api.js``.

* If the cookie is missing/expired the server replies::

      HTTP/2 403
      {"error":"anon_verification_required"}

  The browser then loads the Turnstile widget, gets a token, posts it
  to ``POST /api/web/auth/anonymous-verify`` (body ``{"turnstileToken":"..."}``)
  and the server sets a fresh ``web_anon_session`` cookie.

Because Turnstile is essentially impossible to solve without a real
browser, this client does **not** try to forge cookies on its own.
Instead it offers two ways to obtain one:

1.  ``ChatbotChatApp.bootstrap_cookies(...)`` — opens a Chromium window
    with Playwright, lets Cloudflare auto-solve, and returns the
    cookies.  Works on any normal desktop machine.
2.  Manual paste — open https://plai.chat/, send any message, then
    copy the value of the ``web_anon_session`` cookie from DevTools
    (Application → Cookies) and pass it as ``cookie="..."``.

Usage example
-------------
::

    from plai_chat import ChatbotChatApp

    # First-time bootstrap (only needed once per ~hour or until the
    # cookie expires); persists cookies to ./plai_cookies.json.
    ChatbotChatApp.bootstrap_cookies("plai_cookies.json")

    bot = ChatbotChatApp.from_cookie_file("plai_cookies.json")
    history = []
    for chunk in bot.send_message("Say hello in 3 words.", model="free", history=history):
        print(chunk, end="", flush=True)
    print()
    # history is updated in-place, ready for the next turn.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import time
import typing as _t
from dataclasses import dataclass, field

import httpx


# ---------------------------------------------------------------------------
# Constants taken from the live frontend
# ---------------------------------------------------------------------------

BASE_URL = "https://plai.chat"
CHAT_ENDPOINT = "/api/web/chat/send"
MODELS_ALIAS_ENDPOINT = "/api/web/models/aliases"
SESSION_ENDPOINT = "/api/web/auth/session"
ANON_VERIFY_ENDPOINT = "/api/web/auth/anonymous-verify"

# A modern desktop UA — the site does not care about UA itself, but Cloudflare
# will reject obviously botty UAs (python-requests/x.y).  Keeping a real-looking
# UA is the cheapest anti-bot mitigation we can apply.
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Preset aliases the site exposes to anonymous and paying users.  The actual
# underlying OpenRouter ids per preset are returned by /api/web/models/aliases
# at runtime — call ``ChatbotChatApp.list_models()`` to grab them.
PRESETS = ("free", "balanced", "premium", "image", "vision", "auto")

# Friendly @keyword aliases the UI accepts in the input box.  Each maps to a
# fallback chain of OpenRouter model ids on the server side.
KEYWORD_ALIASES = (
    "gpt", "chatgpt", "openai",
    "claude", "anthropic",
    "gemini",
    "grok",
    "deepseek",
    "mistral",
    "kimi",
    "glm",
    "minimax",
)

# How long the web_anon_session cookie typically lasts before /chat/send 403s.
# The server sets it server-side so the exact lifetime isn't published, but in
# practice it survives well over an hour.  We treat "older than 30 minutes"
# as a hint to refresh proactively.
ANON_COOKIE_REFRESH_AFTER = 30 * 60  # seconds


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlaiChatError(RuntimeError):
    """Base class for all plai.chat client errors."""


class AnonVerificationRequired(PlaiChatError):
    """Raised when the server returns ``{"error":"anon_verification_required"}``.

    The caller must run :meth:`ChatbotChatApp.bootstrap_cookies` (or paste a
    fresh ``web_anon_session`` cookie) and retry.
    """


class PaymentRequired(PlaiChatError):
    """Raised on HTTP 402 — premium models with empty balance."""


class ServerError(PlaiChatError):
    """Any other non-OK HTTP status from /chat/send."""


# ---------------------------------------------------------------------------
# Conversation history — minimal helper so callers do not have to remember
# the exact dict shape.
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation."""

    role: _t.Literal["user", "assistant", "system"]
    content: str

    def to_payload(self) -> dict:
        return {"role": self.role, "content": self.content}


History = _t.List[_t.Union[Message, dict]]


def _normalise_history(history: _t.Optional[History]) -> list[dict]:
    """Coerce an iterable of Message-or-dict into the wire format."""
    if not history:
        return []
    out: list[dict] = []
    for item in history:
        if isinstance(item, Message):
            out.append(item.to_payload())
        elif isinstance(item, dict):
            # Accept anything that has role+content; copy to avoid surprises.
            if "role" not in item or "content" not in item:
                raise ValueError(f"history item missing role/content: {item!r}")
            out.append({"role": item["role"], "content": item["content"]})
        else:
            raise TypeError(f"unsupported history item: {item!r}")
    return out


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


@dataclass
class ChatbotChatApp:
    """Synchronous, streaming client for plai.chat.

    Parameters
    ----------
    cookies:
        A dict of cookies.  At minimum needs ``web_anon_session`` (anonymous
        users) or the magic-link session cookie (paying users).  Use
        :meth:`from_cookie_file` / :meth:`bootstrap_cookies` to populate it.
    base_url:
        Override only for local testing.
    user_agent:
        Override the User-Agent header.  Keep it desktop-y or Cloudflare
        will challenge you.
    timeout:
        Per-event read timeout for the SSE stream.  ``None`` = no timeout
        which is what httpx defaults to anyway.
    """

    cookies: dict = field(default_factory=dict)
    base_url: str = BASE_URL
    user_agent: str = DEFAULT_UA
    timeout: _t.Optional[float] = 120.0
    cookie_file: _t.Optional[str] = None  # if set, refresh writes here

    # ------------------------------------------------------------------ ctor

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._default_headers(),
            cookies=self.cookies,
            follow_redirects=True,
            http2=False,  # plai.chat is HTTP/2 but httpx works fine over 1.1.
        )

    def __enter__(self) -> "ChatbotChatApp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -------------------------------------------------------- alt constructors

    @classmethod
    def from_cookie_file(
        cls, path: str | os.PathLike, **kwargs
    ) -> "ChatbotChatApp":
        """Load a cookie jar previously saved by :meth:`save_cookies` /
        :meth:`bootstrap_cookies`."""
        path = pathlib.Path(path)
        data = json.loads(path.read_text())
        return cls(cookies=data.get("cookies", {}), cookie_file=str(path), **kwargs)

    @classmethod
    def from_cookie_string(cls, cookie_str: str, **kwargs) -> "ChatbotChatApp":
        """Build a client from a raw ``Cookie:`` header string (e.g. copied
        from DevTools)."""
        cookies: dict = {}
        for kv in cookie_str.split(";"):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, _, v = kv.partition("=")
            cookies[k.strip()] = v.strip()
        return cls(cookies=cookies, **kwargs)

    # -------------------------------------------------------------- bootstrap

    @staticmethod
    def bootstrap_cookies(
        out_path: str | os.PathLike = "plai_cookies.json",
        *,
        headless: bool = True,
        timeout: float = 90.0,
        channel: str = "chrome",
    ) -> dict:
        """Solve Cloudflare Turnstile and obtain a ``web_anon_session`` cookie.

        Strategy (validated against the live site 2026-05-07):

        1.  Launch Chromium via :mod:`patchright` (a Playwright fork that
            patches the obvious ``navigator.webdriver`` / CDP fingerprints
            Cloudflare looks for).
        2.  Use ``page.route`` to intercept the navigation to ``plai.chat/``
            and serve a tiny HTML stub that mounts the Turnstile widget
            with the real sitekey.  Because the request still goes to the
            plai.chat origin, the resulting token is valid for that domain.
        3.  Wait for the widget to fill ``[name=cf-turnstile-response]``,
            then POST the token to ``/api/web/auth/anonymous-verify``.
        4.  Read the resulting cookie jar.

        ``channel="chrome"`` (the default) uses the system-installed real
        Google Chrome.  Run ``patchright install chrome`` once first.  In
        practice the bundled chromium-headless-shell does **not** pass
        Cloudflare's bot check on this site; only real Chrome does.
        """
        try:
            from patchright.sync_api import sync_playwright
        except ImportError as e:
            raise PlaiChatError(
                "patchright is required for bootstrap_cookies(); install with"
                " `pip install patchright && patchright install chrome`"
            ) from e

        sitekey = "0x4AAAAAAC-lISmo_bU02UL9"
        landing = f"{BASE_URL}/"
        stub = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<script src='https://challenges.cloudflare.com/turnstile/v0/api.js' async></script>"
            "</head><body><form>"
            f"<div class='cf-turnstile' data-sitekey='{sitekey}'></div>"
            "</form></body></html>"
        )

        launch_kwargs: dict = {
            "headless": headless,
            "args": ["--no-sandbox", "--disable-setuid-sandbox"],
        }
        if channel:
            launch_kwargs["channel"] = channel

        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            # Important: the UA string MUST line up with the actual Chrome
            # binary major version (Cloudflare cross-checks the JA3/H2/UA
            # tuple).  Don't override DEFAULT_UA here \u2014 we use a UA that
            # matches the Chrome installed via ``patchright install chrome``.
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
                ),
            )
            page = ctx.new_page()

            # One-shot interception of the landing URL so the verify call
            # afterwards still hits the real backend.
            handled = {"done": False}

            def _route(route):
                if handled["done"]:
                    route.continue_()
                    return
                handled["done"] = True
                route.fulfill(status=200, content_type="text/html", body=stub)

            page.route(landing, _route)
            page.goto(landing, wait_until="domcontentloaded")

            deadline = time.monotonic() + timeout
            token: _t.Optional[str] = None
            while time.monotonic() < deadline:
                try:
                    val = page.input_value(
                        '[name="cf-turnstile-response"]', timeout=1500
                    )
                    if val:
                        token = val
                        break
                except Exception:
                    pass
                page.wait_for_timeout(500)

            if not token:
                browser.close()
                raise PlaiChatError(
                    "Timed out waiting for Turnstile token. Cloudflare's bot "
                    "check probably blocked us. Try `channel='chrome'` (default) "
                    "after running `patchright install chrome`, or paste a "
                    "cookie manually from your browser."
                )

            verify = page.evaluate(
                """async (tok) => {
                    const r = await fetch('/api/web/auth/anonymous-verify', {
                        method: 'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({turnstileToken: tok}),
                    });
                    return {status: r.status, body: (await r.text()).slice(0, 400)};
                }""",
                token,
            )
            if verify["status"] != 200:
                browser.close()
                raise PlaiChatError(
                    f"anonymous-verify failed: {verify['status']} {verify['body']}"
                )

            cookies = {c["name"]: c["value"] for c in ctx.cookies(BASE_URL)}
            browser.close()

        if "web_anon_session" not in cookies and "anon-session" not in cookies:
            raise PlaiChatError(
                f"verify succeeded but no session cookie was set; jar={list(cookies)}"
            )

        out = {
            "obtained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "cookies": cookies,
        }
        pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
        return cookies

    @staticmethod
    def bootstrap_via_solver(
        solver_url: str,
        out_path: str | os.PathLike = "plai_cookies.json",
        *,
        sitekey: str = "0x4AAAAAAC-lISmo_bU02UL9",
        landing: str = f"{BASE_URL}/",
        timeout: float = 90.0,
        user_agent: str | None = None,
    ) -> dict:
        """Mint a ``web_anon_session`` cookie using a remote Turnstile-solver
        microservice (e.g. the one in ``cf/app.py``, designed for Hugging Face
        Spaces).  No browser, no Patchright on the client side \u2014 only
        :mod:`httpx`.

        The solver service must expose ``GET {solver_url}/solve?url=...&sitekey=...``
        returning ``{"success": true, "token": "...", ...}`` (the same contract
        as the Theyka/Turnstile-Solver project).

        Flow:
            1. Ask the solver for a fresh Turnstile token bound to the
               ``plai.chat`` origin.
            2. POST ``{"turnstileToken": <token>}`` to
               ``/api/web/auth/anonymous-verify`` with plain :mod:`httpx` \u2014
               plai.chat replies ``200 {"success":true,"expiresAt":...}`` and
               sets ``web_anon_session`` via ``Set-Cookie``.
            3. Persist the cookie jar to ``out_path``.

        Args:
            solver_url:  Base URL of a Turnstile solver (no trailing ``/solve``).
                         Example: ``https://your-space.hf.space``.
            out_path:    Where to write the JSON cookie jar.
            sitekey:     Turnstile sitekey for plai.chat (defaults to the live
                         value).
            landing:     ``url`` parameter passed to the solver (must be the
                         plai.chat origin so the token is valid for it).
            timeout:     Total budget in seconds for the solve + verify.
            user_agent:  Optional UA override for the verify call.  Defaults
                         to a recent-Chrome string \u2014 plai.chat does not
                         strictly require it but a desktop UA avoids flags.

        Returns the cookie jar (``{"web_anon_session": "..."}``).
        """
        if not solver_url:
            raise PlaiChatError("solver_url is required")
        solver_url = solver_url.rstrip("/")

        ua = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
        )

        with httpx.Client(timeout=timeout) as cx:
            # 1. Ask the solver for a token.
            r = cx.get(
                f"{solver_url}/solve",
                params={"url": landing, "sitekey": sitekey},
            )
            try:
                payload = r.json()
            except ValueError as e:
                raise PlaiChatError(
                    f"solver returned non-JSON ({r.status_code}): {r.text[:200]}"
                ) from e
            if r.status_code != 200 or not payload.get("success"):
                raise PlaiChatError(
                    f"solver failed ({r.status_code}): {payload}"
                )
            token = payload.get("token")
            if not token:
                raise PlaiChatError(f"solver returned no token: {payload}")

            # 2. POST the token to plai.chat's verify endpoint.  No browser
            #    needed \u2014 the Turnstile token alone is enough.
            verify = cx.post(
                f"{BASE_URL}/api/web/auth/anonymous-verify",
                json={"turnstileToken": token},
                headers={
                    "User-Agent": ua,
                    "Accept": "application/json",
                    "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/",
                    "Content-Type": "application/json",
                },
            )
            if verify.status_code != 200:
                raise PlaiChatError(
                    f"anonymous-verify failed: {verify.status_code} "
                    f"{verify.text[:300]}"
                )

            cookies = {k: v for k, v in cx.cookies.items()}

        if "web_anon_session" not in cookies:
            raise PlaiChatError(
                f"verify succeeded but no web_anon_session cookie was set; "
                f"jar={list(cookies)}"
            )

        out = {
            "obtained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "via": f"solver:{solver_url}",
            "expiresAt": payload.get("elapsed_time"),
            "cookies": cookies,
        }
        pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
        return cookies

    def save_cookies(self, path: str | os.PathLike) -> None:
        """Persist the current cookie jar to disk."""
        out = {
            "obtained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "cookies": dict(self._client.cookies),
        }
        pathlib.Path(path).write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- helpers

    def _default_headers(self) -> dict:
        # Cloudflare and the Express backend both look at Origin/Referer.
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/event-stream, application/json;q=0.9, */*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------- discovery

    def list_models(self) -> dict[str, str]:
        """Fetch the live alias map (preset → comma-separated OpenRouter
        ids) from the server.  Useful if you want to know which models a
        given preset will route to today."""
        r = self._client.get(MODELS_ALIAS_ENDPOINT)
        r.raise_for_status()
        return r.json().get("aliases", {})

    def session_info(self) -> dict:
        """Return the current session info ``{"authenticated": bool, "user":
        {...}}`` — handy to verify your magic-link cookie is still valid."""
        r = self._client.get(SESSION_ENDPOINT)
        r.raise_for_status()
        return r.json()

    # ----------------------------------------------------------------- chat

    def send_message(
        self,
        prompt: str,
        model: str = "free",
        history: _t.Optional[History] = None,
        *,
        zdr: bool = False,
        attachments: _t.Optional[list[dict]] = None,
        conversation_started_at: _t.Optional[str] = None,
        update_history: bool = True,
        return_events: bool = False,
    ) -> _t.Generator[str, None, dict]:
        """Stream a chat completion as text chunks.

        Yields incremental **delta** strings (not the cumulative text, even
        though the wire format sends cumulative text — we diff it for
        you).  Returns a metadata dict at the end::

            {"model": "openai/gpt-5.5", "balance": 12.3, "full_text": "..."}

        Parameters
        ----------
        prompt:
            The user message.
        model:
            One of the presets (``"free"``, ``"balanced"``, ``"premium"``,
            ``"image"``, ``"vision"``, ``"auto"``), a friendly keyword
            (``"gpt"``, ``"claude"``, ``"deepseek"`` …) or a fully-qualified
            OpenRouter id (``"openai/gpt-5.5"``).  You can also send a
            comma-separated chain — the server will fall back through them.
        history:
            Conversation history.  Each item is a :class:`Message` or a
            ``{"role": ..., "content": ...}`` dict.  If ``update_history``
            is True (the default) the user prompt and the assistant reply
            are appended to the **same list** you passed in.
        zdr:
            "Zero Data Retention" toggle — only honoured for paying users.
        attachments:
            Vision/PDF attachments.  Each item should be
            ``{"name": str, "type": str, "dataUrl": str}`` with a base64
            data URL — same shape the website uses internally.
        conversation_started_at:
            ISO-8601 timestamp.  Auto-populated on the first turn.
        update_history:
            See ``history`` above.  Set False if you want to manage the
            list yourself.
        return_events:
            If True, yield raw decoded SSE events (dicts) instead of text
            chunks.  Useful for advanced consumers who want to see image
            payloads, usage updates, etc.

        Raises
        ------
        AnonVerificationRequired
            Cookie expired/missing.  Call :meth:`bootstrap_cookies` and try
            again.
        PaymentRequired
            HTTP 402 from a premium model on a $0 balance.
        ServerError
            Any other non-OK HTTP status.
        """
        history = history if history is not None else []
        wire_history = _normalise_history(history)

        if not conversation_started_at:
            conversation_started_at = (
                _dt.datetime.now(_dt.timezone.utc)
                .replace(tzinfo=None)
                .isoformat(timespec="milliseconds")
                + "Z"
            )

        body = {
            "message": prompt,
            "history": wire_history,
            "model": model,
            "attachments": attachments or [],
            "conversationStartedAt": conversation_started_at,
            "zdr": zdr,
        }

        # NB: httpx.stream() opens a context manager that auto-closes the
        # underlying TCP connection when the generator is exhausted or GC'd.
        with self._client.stream("POST", CHAT_ENDPOINT, json=body) as resp:
            self._raise_for_chat_status(resp)

            full_text = ""
            chosen_model: str = ""
            balance: _t.Optional[float] = None

            for event in _iter_sse(resp):
                if return_events:
                    yield event  # type: ignore[misc]
                    if event.get("type") == "done":
                        break
                    continue

                etype = event.get("type")
                if etype == "content":
                    new = event.get("text") or ""
                    # The wire format sends *cumulative* text; emit deltas.
                    if new.startswith(full_text):
                        delta = new[len(full_text):]
                        full_text = new
                    else:
                        # Server reset the buffer — treat the whole payload
                        # as the new text and emit it as a delta.
                        delta = new
                        full_text = new
                    if delta:
                        yield delta
                elif etype == "model":
                    chosen_model = event.get("model") or chosen_model
                elif etype == "usage":
                    balance = event.get("balance", balance)
                elif etype == "info":
                    # Non-fatal informational message (e.g. "model X
                    # skipped, no vision support").  Surface it as a delta
                    # the same way the website would.
                    msg = event.get("message")
                    if msg:
                        prefix = f"*ℹ {msg}*\n\n"
                        yield prefix
                        full_text += prefix
                elif etype == "images":
                    # Append generated images as markdown — matches website
                    # behaviour.
                    md = _images_to_markdown(event.get("images") or [])
                    if md:
                        yield "\n\n" + md
                        full_text += "\n\n" + md
                elif etype == "error":
                    err = event.get("error", "unknown error")
                    raise ServerError(err)
                elif etype == "done":
                    chosen_model = event.get("model") or chosen_model
                    break
                # "start" events have no user-visible payload.

        if update_history:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": full_text})

        return {
            "model": chosen_model,
            "balance": balance,
            "full_text": full_text,
        }

    # ----------------------------------------------------------- internal io

    def _raise_for_chat_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 200:
            return
        # Try to read a JSON error body — chat/send always sends JSON for
        # non-200s.
        try:
            payload = resp.read()
            data = json.loads(payload)
        except Exception:
            data = {}

        if resp.status_code == 403 and data.get("error") == "anon_verification_required":
            raise AnonVerificationRequired(
                "Cloudflare Turnstile cookie expired/missing. "
                "Call ChatbotChatApp.bootstrap_cookies() to mint a new one."
            )
        if resp.status_code == 402 or data.get("paymentRequired"):
            raise PaymentRequired(data.get("error") or "Payment required")
        raise ServerError(
            f"HTTP {resp.status_code}: {data.get('error') or resp.text[:200]}"
        )


# ---------------------------------------------------------------------------
# SSE parser — small enough that pulling in sseclient-py is overkill.
# ---------------------------------------------------------------------------


def _iter_sse(resp: httpx.Response) -> _t.Iterator[dict]:
    """Yield decoded ``data:`` JSON events from an SSE httpx stream."""
    buf = ""
    for raw in resp.iter_text():
        buf += raw
        # SSE separates events by *blank* lines, but plai.chat uses a
        # simpler one-event-per-line format — we accept either.
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                # Skip malformed events rather than crashing the stream.
                continue


def _images_to_markdown(images: list) -> str:
    """Mirror the JS chat.js logic for converting OpenRouter image payloads
    into a markdown image block."""
    out: list[str] = []
    for img in images:
        url = ""
        if isinstance(img, str):
            url = img
        elif isinstance(img, dict):
            if img.get("url"):
                url = img["url"]
            elif (img.get("image_url") or {}).get("url"):
                url = img["image_url"]["url"]
            elif img.get("b64_json"):
                url = "data:image/png;base64," + img["b64_json"]
            elif (img.get("image") or {}).get("url"):
                url = img["image"]["url"]
            elif img.get("data"):
                d = img["data"]
                url = d if d.startswith("data:") else "data:image/png;base64," + d
        if url:
            out.append(f"![Generated Image]({url})")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Tiny CLI that mints a cookie if needed, then streams two turns."""
    import argparse

    ap = argparse.ArgumentParser(description="plai.chat unofficial CLI demo")
    ap.add_argument(
        "--cookies",
        default="plai_cookies.json",
        help="Path to a JSON cookie jar (will be created on first run).",
    )
    ap.add_argument(
        "--cookie",
        default=None,
        help="Raw 'name=value; name2=value2' cookie header string "
        "(skips the file).",
    )
    ap.add_argument("--model", default="free")
    ap.add_argument(
        "--prompt",
        default="Say hello in 3 words.",
        help="First-turn prompt to send.",
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="Force re-running the Playwright Turnstile bootstrap.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run the bootstrap browser headless (often blocked by CF).",
    )
    args = ap.parse_args()

    if args.cookie:
        bot = ChatbotChatApp.from_cookie_string(args.cookie)
    else:
        cookie_path = pathlib.Path(args.cookies)
        if args.bootstrap or not cookie_path.exists():
            print(f"[bootstrap] minting web_anon_session cookie → {cookie_path}…")
            ChatbotChatApp.bootstrap_cookies(cookie_path, headless=args.headless)
        bot = ChatbotChatApp.from_cookie_file(cookie_path)

    history: list[dict] = []

    def _stream(prompt: str) -> None:
        print(f"\n>>> {prompt}\n--- ", end="", flush=True)
        try:
            gen = bot.send_message(prompt, model=args.model, history=history)
            meta: dict = {}
            # Iterate manually so we can capture the generator's return value
            # (a Generator returns via StopIteration.value, which `for` swallows).
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as stop:
                    meta = stop.value or {}
                    break
                print(chunk, end="", flush=True)
            print(f"\n[model={meta.get('model')} balance={meta.get('balance')}]")
        except AnonVerificationRequired as e:
            print(f"\n[!] {e}")
            print("    Re-run with --bootstrap to refresh the cookie.")

    _stream(args.prompt)
    _stream("Now translate that to French.")

    bot.close()


if __name__ == "__main__":
    _demo()
