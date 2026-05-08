"""Minimal Cloudflare-Turnstile solver service for plai.chat.

Exposes ``GET /solve?url=<url>&sitekey=<sitekey>`` using the same JSON contract
as the Theyka / Hugging-Face Turnstile-Solver projects (the ``cf/app.py`` the
user supplied).  It can be deployed once (locally, on a VPS, on a Hugging Face
Space, etc.) and reused by ``plai_cli.py --solver <url>`` forever \u2014 the
plai-chat client itself stays browser-free.

Endpoints
---------
* ``GET /health``                          \u2192 ``{"status":"healthy"}``
* ``GET /solve?url=<url>&sitekey=<key>``   \u2192 ``{"success":true, "token": "...", "elapsed_time": 1.23}``
                                              or ``{"success":false, "error": "..."}``

Run locally
-----------
::

    pip install plai-solver-deps     # quart + patchright
    patchright install chrome
    python plai_solver.py            # listens on :7860

Why this exists
---------------
We tried using the user-supplied generic ``cf/app.py``: even after patching
it to use real Chrome (``channel='chrome'``) and dropping the manual click,
its tokens were rejected by ``plai.chat/api/web/auth/anonymous-verify`` with
``403 Bot verification failed`` \u2014 the default ``HeadlessChrome`` UA gets
flagged.  This file bakes in the exact knobs that make Cloudflare's invisible
sitekey configuration on plai.chat happy:

* real Chrome via ``channel='chrome'`` (not ``chromium-headless-shell``);
* a non-headless-looking UA pinned to the Chrome major version installed by
  ``patchright install chrome``;
* a *one-shot* route interception of the landing URL (subsequent requests
  fall through to the real origin so the verify call works);
* poll-only, no click \u2014 plai.chat's sitekey is invisible/non-interactive.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from quart import Quart, jsonify, request
from patchright.async_api import async_playwright


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] -> %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plai-solver")

app = Quart(__name__)


# Use a Windows UA by default as it's more common and often less flagged
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)

STUB_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async></script>
</head><body>
<form><div class="cf-turnstile" data-sitekey="{sitekey}"></div></form>
</body></html>
"""


async def _solve(url: str, sitekey: str, timeout: float = 120.0) -> str | None:
    """Mint a fresh Turnstile token bound to the *url* origin.

    Returns ``None`` on failure.
    """
    landing = url if url.endswith("/") else url + "/"
    body = STUB_HTML.format(sitekey=sitekey)
    handled = {"v": False}

    async def _route(route):
        if handled["v"]:
            await route.continue_()
            return
        handled["v"] = True
        await route.fulfill(status=200, content_type="text/html", body=body)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            channel="chrome",
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        try:
            ctx = await browser.new_context(
                user_agent=CHROME_UA,
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
            )
            page = await ctx.new_page()
            await page.route(landing, _route)
            await page.goto(landing, wait_until="domcontentloaded")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    val = await page.input_value(
                        '[name="cf-turnstile-response"]', timeout=1500
                    )
                    if val:
                        return val
                except Exception:
                    pass
                await asyncio.sleep(0.3)
            return None
        finally:
            await browser.close()


@app.route("/health")
async def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/solve", methods=["GET", "POST"])
async def solve():
    if request.method == "POST":
        data = await request.get_json(force=True, silent=True) or {}
        target_url = data.get("url")
        sitekey = data.get("sitekey")
    else:
        target_url = request.args.get("url")
        sitekey = request.args.get("sitekey")

    if not target_url or not sitekey:
        return (
            jsonify(
                {"success": False, "error": "url and sitekey are required"}
            ),
            400,
        )

    log.info(f"solving {target_url} sitekey={sitekey[:12]}…")
    t0 = time.time()
    try:
        token = await _solve(target_url, sitekey)
    except Exception as e:
        log.exception("solver crashed")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_time": round(time.time() - t0, 3),
                }
            ),
            500,
        )

    elapsed = round(time.time() - t0, 3)
    if not token:
        log.warning(f"  no token after {elapsed}s")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Timed out waiting for Turnstile token",
                    "elapsed_time": elapsed,
                }
            ),
            422,
        )
    log.info(f"  solved in {elapsed}s (token len={len(token)})")
    return (
        jsonify(
            {"success": True, "token": token, "elapsed_time": elapsed}
        ),
        200,
    )


@app.route("/")
async def index():
    return (
        "<h1>plai-solver</h1>"
        "<p>GET <code>/solve?url=&lt;url&gt;&amp;sitekey=&lt;sitekey&gt;</code>"
        " or <code>POST /solve</code> with JSON body.</p>"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
