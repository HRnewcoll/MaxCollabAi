"""
minigrok.utils
Bounded caches, circuit breakers, rate limiter
"""


# Cross-module imports
import os, json, re, logging, threading, time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from collections import OrderedDict
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

from .base import log
from .config import Config


# ════════════════════════════════════════════════════════════════════
# § 4  BOUNDED CACHES + CIRCUIT BREAKERS + RATE LIMITER
# ════════════════════════════════════════════════════════════════════

class LRUCache:
    """Thread-safe LRU cache with optional TTL."""
    def __init__(self, maxsize=200, ttl=0):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if self._ttl and (datetime.now() - entry["ts"]).total_seconds() > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry["v"]

    def set(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = {"v": value, "ts": datetime.now()}
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        with self._lock:
            return len(self._cache)

_tool_cache     = LRUCache(Config.TOOL_CACHE_SIZE, ttl=Config.TOOL_CACHE_TTL)
_response_cache = LRUCache(Config.LRU_CACHE_SIZE, ttl=0)
_token_cache    = LRUCache(Config.TOKEN_CACHE_SIZE, ttl=600)


class CircuitBreaker:
    """Prevents repeated calls to failing services."""
    def __init__(self, name, max_fail=3, cooldown=300):
        self.name = name
        self.max_fail = max_fail
        self.cooldown = cooldown
        self.fails = 0
        self.opened = None
        self._lock = threading.Lock()

    def is_open(self):
        with self._lock:
            if not self.opened:
                return False
            if (datetime.now() - self.opened).total_seconds() > self.cooldown:
                self.fails = 0
                self.opened = None
                return False
            return True

    def fail(self):
        with self._lock:
            self.fails += 1
            if self.fails >= self.max_fail:
                self.opened = datetime.now()
                log.warning(f"Circuit opened: {self.name}")

    def ok(self):
        with self._lock:
            self.fails = 0
            self.opened = None

_breakers = {k: CircuitBreaker(k) for k in ["ddg", "wikipedia", "arxiv", "playwright", "crawl4ai"]}


class RateLimiter:
    """Simple sliding-window rate limiter."""
    def __init__(self, max_per_minute=30):
        self._times: List[float] = []
        self._max = max_per_minute
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._times = [t for t in self._times if now - t < 60]
            if len(self._times) >= self._max:
                return False
            self._times.append(now)
            return True

_rate_limiter = RateLimiter(Config.RATE_LIMIT_RPM)


def with_retry(fn, breaker_name="", retries=3):
    """Execute fn with retry logic and circuit breaker support."""
    breaker = _breakers.get(breaker_name)
    if breaker and breaker.is_open():
        return f"Service '{breaker_name}' temporarily unavailable."
    last_error = None
    for attempt in range(retries):
        try:
            result = fn()
            if breaker:
                breaker.ok()
            return result
        except Exception as e:
            if breaker:
                breaker.fail()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            last_error = e
    return f"Error: {last_error}"

log.info("Caches + circuit breakers + rate limiter ready")




def pii_filter(text: str) -> str:
    """Redact common PII patterns."""
    for pattern in [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{16}\b',
                    r'\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b',
                    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def injection_check(text: str) -> bool:
    """Check for prompt injection attempts."""
    patterns = [r"ignore\s+(all\s+)?previous", r"you\s+are\s+now",
                r"act\s+as\s+(?!a\s+tutor)", r"disregard\s+(all\s+)?instructions",
                r"reveal\s+.*system\s+prompt", r"jailbreak"]
    return any(re.search(p, text.lower()) for p in patterns)
