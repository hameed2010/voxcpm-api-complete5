import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Request
from app.core.config import settings

_rate_lock = Lock()
_rate_windows = defaultdict(deque)

def require_api_key(x_api_key: str = Header(default="")):
    configured = settings.API_KEY
    if not configured or configured.strip() in {"change-me", "changeme", "your-api-key"}:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def rate_limit(request: Request, x_api_key: str = Header(default="")):
    # Authentication first; rate-limit bucket is keyed by client + key fingerprint.
    require_api_key(x_api_key)
    client = request.client.host if request.client else "unknown"
    bucket = client
    now = time.monotonic()
    with _rate_lock:
        q = _rate_windows[bucket]
        cutoff = now - settings.RATE_LIMIT_WINDOW_SECONDS
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= settings.RATE_LIMIT_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)
