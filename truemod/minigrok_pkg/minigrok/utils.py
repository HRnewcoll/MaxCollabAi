"""
minigrok.utils
Bounded caches, circuit breakers, rate limiter
"""


import json, shutil, threading, tempfile, re, traceback, warnings, logging, sqlite3, socket
import wave, hashlib, io, base64, uuid, copy, queue, math, difflib
import ast as _ast, asyncio, nest_asyncio, secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Generator, List, Dict, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from enum import Enum
from filelock import FileLock

try:
    nest_asyncio.apply()
except Exception:
    pass
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"]           = "true"

import numpy as np, torch, networkx as nx, schedule as schedule_lib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jsonschema import validate, ValidationError

from transformers import (AutoProcessor, BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from datasets import load_dataset as hf_load, Dataset
from huggingface_hub import snapshot_download

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import wikipedia as wikipedia_lib, arxiv as arxiv_lib
from duckduckgo_search import DDGS
import trafilatura, feedparser
import PyPDF2, pandas as pd
from PIL import Image
# cv2 and whisper are lazy-imported on first use (saves ~300MB VRAM at boot)
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
import gradio as gr

try:
    from google.colab import drive as _cd
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── Logging ─────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit log records as one-line JSON — machine-queryable."""
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts":    self.formatTime(record, datefmt="%H:%M:%S"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)

# Single handler: JSON for INFO+, with a readable fallback line for WARNING+
# This prevents the triple-print issue (JSON × 2 + plain text) seen in Colab
if not log.handlers:   # Guard against re-run in same notebook session
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JSONFormatter())
    log.addHandler(_handler)

# v13: Load .env file if present (graceful if missing or python-dotenv not installed)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # .env values don't override existing env vars
    log.info("Loaded .env file")
except ImportError:
    pass  # python-dotenv not installed — that's fine

log.info("Imports OK")



# Cross-module imports
from .base import log, APP_NAME, APP_VERSION, IN_COLAB
from .config import Config

# ────────────────────────────────────────────────────────────
# § 4  BOUNDED CACHES + CIRCUIT BREAKERS + RATE LIMITER
# ────────────────────────────────────────────────────────────
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


