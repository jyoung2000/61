"""AuthMiddleware — resolves the session cookie before routers run.

Lifecycle per request:

  1. Check request path against allow-lists (static / public / auth
     endpoints themselves) — these short-circuit before any lookup.
  2. Read the ``clipai_session`` cookie.
  3. Look up the session; verify the IP+UA fingerprint hasn't changed.
     A changed fingerprint means the user is coming from a new IP or
     browser — force re-login by responding 401. The frontend catches
     this and redirects to ``/login``.
  4. Look up the user; skip the request when the user is deactivated.
  5. Attach ``request.state.user`` + ``request.state.session`` and
     continue the ASGI chain.
  6. Periodically ``touch_session`` so long-running use doesn't expire.

Endpoints that should be reachable without auth (e.g. ``/api/auth/login``
or the SPA's index page) bypass the middleware via
``_is_public_path``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Iterable

from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.app.auth.security import compute_fingerprint
from backend.app.auth.store import get_session, get_user, touch_session

logger = logging.getLogger(__name__)


SESSION_COOKIE = "clipai_session"
REMEMBER_COOKIE = "clipai_remember"
REMEMBER_MAX_AGE = 365 * 24 * 3600


def _cookie_secure() -> bool:
    """Return True when the instance is served over HTTPS.

    Set ``CLIPAI_HTTPS=true`` in the environment when running behind an
    SSL-terminating reverse proxy so cookies receive the ``Secure`` flag.
    Defaults to ``false`` so plain-HTTP / localhost installs work without
    any configuration.
    """
    return os.environ.get("CLIPAI_HTTPS", "false").lower() not in (
        "0", "false", "no", "off",
    )


# Paths that never require auth. SPA routes fall through to index.html
# which is public; the frontend then mounts AuthProvider and fetches
# /api/auth/me — a 401 there triggers the redirect to /login.
#
# Note: /api/auth/me is NOT in this list. It MUST flow through this
# middleware so the router handler can read ``request.state.user``.
# Unauthenticated callers still receive a 401 from the middleware —
# same signal the frontend needs to redirect to /login.
_PUBLIC_EXACT = {
    "/",
    "/login",
    "/api/site-config",
    "/api/auth/login",
    "/api/auth/bootstrap",
    "/health",
    "/healthz",
    # GPU Companion self-update: the Companion is a non-browser LAN peer with no
    # session cookie, so it 401'd on the installer manifest ("answered HTTP 401
    # Unauthorized for the installer manifest") and its Update button never
    # worked. These are READ-ONLY installer artifacts (version metadata + the
    # public app binary, sha256-verified by the Companion) — safe to serve
    # unauthenticated over the LAN. The admin refresh (POST /companion/refresh)
    # is deliberately NOT here, so it stays behind auth.
    "/api/downloads/companion/manifest",
    "/api/downloads/companion/windows",
    "/api/downloads/companion/windows_msi",
    "/api/downloads/companion/mac",
    # The YOLO-World weight the Companion streams during its from-source
    # vision-offload install — same class of read-only public artifact, and
    # the Companion (a cookieless LAN peer) 401'd on it otherwise. The
    # install/status control routes (POST vision-install, GET .../status) are
    # deliberately NOT here — they stay behind auth like /companion/refresh.
    "/api/downloads/companion/vision-model",
}

_PUBLIC_PREFIXES = (
    "/static/",
    "/assets/",
    "/favicon",
    "/api/site-uploads/",
    "/api/auth/login",   # handle query params
    "/api/auth/logout",
    # Signed share-link API: the token IS the credential. No session
    # cookie required so recipients can view the linked content
    # without needing an account. Issuance / revocation lives at
    # /api/share/links/* and DOES require auth (handled per-route).
    "/api/share/public/",
    # OG unfurl / preview pages live under /share/...
    "/share/",
    # SPA is a single index.html; any UI route is public — frontend
    # enforces the redirect.
)


_SPA_ROUTE_RE = re.compile(
    r"^/(login|signin|signup|upload|clips|media|logs|settings|analysis|seo|share)(/|$)"
)


def _is_public_path(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    for p in _PUBLIC_PREFIXES:
        if path.startswith(p):
            return True
    if _SPA_ROUTE_RE.match(path):
        return True
    # Anything not under /api/ is a SPA route (index.html).
    if not path.startswith("/api/"):
        return True
    return False


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else ""


class AuthMiddleware(BaseHTTPMiddleware):
    """Attach the authenticated user to ``request.state`` or respond 401."""

    # In-memory throttle: we only hit touch_session once per session
    # per ``_TOUCH_INTERVAL`` seconds of real time so we don't re-write
    # the sessions file on every single API call.
    _TOUCH_INTERVAL = 60.0

    def __init__(self, app, touch_cache: dict | None = None):
        super().__init__(app)
        self._touch_cache: dict = touch_cache or {}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_public_path(path):
            return await call_next(request)

        ip = _client_ip(request)
        ua = request.headers.get("user-agent", "")
        token = request.cookies.get(SESSION_COOKIE)
        session = None
        new_remember_token = None

        # --- Primary: validate the session cookie ---
        if token:
            session = await get_session(token)
            if session is not None:
                if compute_fingerprint(ip, ua) != session.fingerprint:
                    # Fingerprint mismatch — kill the old session and fall
                    # through to the remember-token fallback below rather
                    # than immediately rejecting. A remember token can
                    # transparently issue a fresh fingerprinted session.
                    from backend.app.auth.store import delete_session
                    await delete_session(token)
                    session = None
                    token = None

        # --- Fallback: exchange a remember-me token for a new session ---
        if session is None:
            remember_cookie = request.cookies.get(REMEMBER_COOKIE)
            if remember_cookie:
                from backend.app.auth.store import exchange_remember_token
                result = await exchange_remember_token(
                    remember_cookie, ip=ip, user_agent=ua
                )
                if result is not None:
                    session, new_remember_token = result
                    token = session.token

        if session is None:
            return self._clear_and_reject("not authenticated")

        user = await get_user(session.user_id)
        if user is None or not user.active:
            return self._clear_and_reject("user not found or deactivated")

        # Attach to request.state for downstream deps.
        request.state.user = user
        request.state.session = session

        # Throttled touch_session — extends both the server-side
        # ``expires_at`` AND (via ``response.set_cookie`` below) the
        # browser-side cookie max-age, so an active user never has to
        # sign in again as long as they hit the app at least once
        # within the rolling window.
        now = time.monotonic()
        last = self._touch_cache.get(token, 0.0)
        refreshed_cookie = new_remember_token is not None  # always refresh after exchange
        if now - last > self._TOUCH_INTERVAL:
            try:
                await touch_session(token)
                self._touch_cache[token] = now
                refreshed_cookie = True
            except Exception:
                pass

        response = await call_next(request)
        if refreshed_cookie:
            try:
                # Honor the per-session ``remember`` flag — never
                # accidentally promote a session-scoped cookie to a
                # persistent one when the user explicitly opted out
                # of "Remember me" at login.
                set_session_cookie(
                    response, token,
                    remember=getattr(session, "remember", True),
                )
            except Exception:
                pass
        if new_remember_token is not None:
            try:
                response.set_cookie(
                    key=REMEMBER_COOKIE,
                    value=new_remember_token.token,
                    httponly=True,
                    secure=_cookie_secure(),
                    samesite="lax",
                    path="/",
                    max_age=REMEMBER_MAX_AGE,
                )
            except Exception:
                pass
        return response

    def _clear_and_reject(self, detail: str) -> Response:
        resp = JSONResponse({"detail": detail}, status_code=401)
        resp.delete_cookie(SESSION_COOKIE, path="/")
        resp.delete_cookie(REMEMBER_COOKIE, path="/")
        return resp


def set_session_cookie(
    response: Response,
    token: str,
    *,
    max_age_seconds: int = 30 * 24 * 3600,
    remember: bool = True,
) -> None:
    """Attach the session cookie with safe defaults.

    ``remember=True`` (default) writes a persistent cookie with
    ``max_age``: the browser keeps it across restarts and the user
    stays signed in for up to ``max_age_seconds``.

    ``remember=False`` writes a SESSION cookie (no ``Max-Age`` /
    ``Expires``): browsers drop it on quit, so the next time the
    user opens ClipAI on that device they have to sign in again.
    The server-side session record is unchanged either way; the
    auto-extending touch loop on the middleware still rolls the
    expiry forward, but no cookie persists past the browser tab.
    """
    cookie_kwargs = dict(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )
    if remember:
        cookie_kwargs["max_age"] = max_age_seconds
    response.set_cookie(**cookie_kwargs)


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
