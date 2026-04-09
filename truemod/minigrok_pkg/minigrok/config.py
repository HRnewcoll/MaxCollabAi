"""
minigrok.config
Configuration, system paths, exceptions
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
from .models import model

# ────────────────────────────────────────────────────────────
# § 1  IMPORTS
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 1  IMPORTS
# ════════════════════════════════════════════════════════════════════

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
log = logging.getLogger("MiniGrok")
log.setLevel(logging.INFO)
log.propagate = False   # Prevent duplicate output from root logger

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



# ────────────────────────────────────────────────────────────
# § 2  CONFIGURATION
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 2  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

# ── Config as frozen dataclass (immutable after definition) ─────
# All constants live here. IDE will type-check them. Can't be
# accidentally mutated at runtime. Override via env vars if needed.

from dataclasses import dataclass, field
from typing import ClassVar

@dataclass(frozen=False)  # frozen=True would break Colab cell re-runs; keep False but treat as read-only
class _Config:
    """All tunable constants — single source of truth. DO NOT mutate after startup."""

    # ── Model ────────────────────────────────────────────────────────
    MODEL_ID: str        = "Qwen/Qwen2.5-VL-7B-Instruct"          # v11: fixed model ID
    MODEL_FALLBACKS: ClassVar[list] = ["Qwen/Qwen2-VL-7B-Instruct"]
    MIN_VRAM_GB: float     = 5.5

    # ── Feature Toggles (set False to save VRAM / skip slow installs) ─
    ENABLE_IMAGE_GEN: bool   = True   # Stable Diffusion (uses ~4GB extra VRAM when active)
    ENABLE_VOICE: bool       = True   # TTS + Whisper STT
    ENABLE_BROWSER: bool     = True   # Playwright browser agent
    ENABLE_BG_LEARNING: bool = True   # Background arXiv/RSS learner
    ENABLE_REFLECTION: bool  = True   # Self-critique loop after complex answers
    USE_LARGE_EMBEDDER: bool = False  # True=BAAI/bge-m3 (better), False=MiniLM (faster, less VRAM)

    # ── Security ─────────────────────────────────────────────────────
    # Set GRADIO_AUTH to ("username","password") or leave None for no auth.
    # WARNING: if GRADIO_SHARE=True and GRADIO_AUTH=None, anyone with the link
    # can access your workspace, run code, and view memory.
    GRADIO_AUTH: object  = None
    GRADIO_SHARE: bool = True

    # ── Caches ────────────────────────────────────────────────────────
    LRU_CACHE_SIZE: int   = 200
    TOKEN_CACHE_SIZE: int = 500
    TOOL_CACHE_SIZE: int  = 100
    TOOL_CACHE_TTL: int   = 300
    CACHE_TTLS: ClassVar[dict] = {"factual":3600,"news":300,"code":1800,"creative":0,"personal":0,"math":7200}

    # ── RAG + Memory ─────────────────────────────────────────────────
    DEDUP_THRESHOLD: float       = 0.95
    CONF_THRESHOLD: float        = 0.65
    MEMORY_HALF_LIFE: int      = 30
    MEMORY_MIN_IMP: float        = 0.05
    MAX_SAVED_CONVS: int       = 60
    # Memory compression: summarise conversation when history > this many turns
    COMPRESSION_THRESHOLD: int = 20
    COMPRESSION_KEEP: int      = 8

    # ── Training ─────────────────────────────────────────────────────
    KERNEL_VAR_LIMIT: int  = 500
    RATE_LIMIT_RPM: int    = 30
    BG_LEARN_INTERVAL: int = 7200
    LORA_R: int            = 16
    LORA_ALPHA: int        = 32
    LORA_DROPOUT: float      = 0.05
    LORA_TARGETS: ClassVar[list] = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    # DPO: corrections are queued to SQLite, never run inline (prevents UI freeze)
    DPO_BATCH_MIN: int     = 10

    # ── Model fallback chain (v12.1) ─────────────────────────────
    # Primary: Qwen2.5-VL-7B (vision+language, ~6.2GB VRAM)
    # Fallback 1: Qwen2.5-7B text-only (faster, no vision, ~5.5GB)
    # Fallback 2: Qwen2-VL-7B (older VL model)
    # Fallback 3: Qwen2.5-3B text-only (tiny, fits in 4GB VRAM)
    # Fallback 4: microsoft/Phi-3.5-mini-instruct (3.8B, last resort)
    # The system tries each in order until one loads.
    MODEL_FALLBACK_CHAIN: ClassVar[list] = [
        {"id": "Qwen/Qwen2.5-VL-7B-Instruct",      "vision": True,  "size_gb": 6.2, "cls": "Qwen2_5_VLForConditionalGeneration"},
        {"id": "Qwen/Qwen2.5-7B-Instruct",          "vision": False, "size_gb": 5.5, "cls": "AutoModelForCausalLM"},
        {"id": "Qwen/Qwen2-VL-7B-Instruct",         "vision": True,  "size_gb": 6.0, "cls": "Qwen2VLForConditionalGeneration"},
        {"id": "Qwen/Qwen2.5-3B-Instruct",          "vision": False, "size_gb": 2.8, "cls": "AutoModelForCausalLM"},
        {"id": "microsoft/Phi-3.5-mini-instruct",   "vision": False, "size_gb": 2.4, "cls": "AutoModelForCausalLM"},
    ]
    # Fast model for simple queries (classification, short answers)
    # Set to None to always use the primary model
    FAST_MODEL_ID: str     = "Qwen/Qwen2.5-3B-Instruct"
    FAST_MODEL_THRESH: int = 30   # words — queries under this use fast model

    DPO_QUEUE_FILE: object    = None   # Set dynamically after DR is known (see § 3)

Config = _Config()  # Singleton instance



# ────────────────────────────────────────────────────────────
# § 3  SYSTEM + PATHS
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 3  SYSTEM + PATHS
# ════════════════════════════════════════════════════════════════════

if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU — Runtime → Change runtime type → T4 GPU")

GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB  = torch.cuda.get_device_properties(0).total_memory / 1e9
log.info(f"GPU: {GPU_NAME}  VRAM: {VRAM_GB:.1f} GB")

if IN_COLAB:
    _cd.mount("/content/drive", force_remount=False)

DRIVE_ROOT = "/content/drive/MyDrive"
MODEL_SEARCH = (
    # Always check current version's fine-tuned model first
    [f"{DRIVE_ROOT}/minigrok_v{APP_VERSION.split(".")[0]}/model"] +
    [f"{DRIVE_ROOT}/minigrok_v{v}/model" for v in ["13","12","11","10","9","8","7"]] +
    [f"{DRIVE_ROOT}/mini_grok_model"]
)
DR        = f"{DRIVE_ROOT}/minigrok_v{APP_VERSION.split(".")[0]}"  # Dynamic: uses APP_VERSION major
WORKSPACE = f"{DR}/workspace"

DIRS = {
    "model": "/content/model", "memory": "/content/memory", "voices": "/content/voices",
    "tmp_train": "/content/tmp_train", "outputs": "/content/outputs", "uploads": "/content/uploads",
    "chroma": "/content/chroma", "moe": f"{DR}/moe_adapters", "sd_cache": f"{DR}/sd_cache",
    "checkpts": f"{DR}/checkpoints", "benchmarks": f"{DR}/benchmarks",
    "ws_projects": f"{WORKSPACE}/projects", "ws_research": f"{WORKSPACE}/research",
    "ws_homework": f"{WORKSPACE}/homework", "ws_downloads": f"{WORKSPACE}/downloads",
    "ws_plugins": f"{WORKSPACE}/plugins", "ws_exports": f"{WORKSPACE}/exports",
}

for d in list(DIRS.values()) + [DR, f"{DR}/model", f"{DR}/memory",
                                  f"{DR}/voices", f"{DR}/chats", WORKSPACE]:
    os.makedirs(d, exist_ok=True)

MODEL_DIR = DIRS["model"]

WS_NOTES = Path(WORKSPACE) / "memory.md"
if not WS_NOTES.exists():
    WS_NOTES.write_text(f"# {APP_NAME} Workspace Memory\nCreated: {datetime.now():%Y-%m-%d}\n\n## Notes\n")

AUDIT_LOG = "/content/audit.jsonl"
_DB_FILE  = f"{DR}/memory/minigrok.db"


# ── Forward definitions: SQLite helpers needed at module init time ──────
# (Full implementations appear later in § 5e; these run at § 3 init)
def _get_db() -> "sqlite3.Connection":
    if not _DB_PATH:
        raise StorageError("DB path not set yet.")
    conn = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def _init_db(db_path: str):
    global _DB_PATH
    _DB_PATH = db_path
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dpo_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL,
            chosen TEXT NOT NULL, rejected TEXT NOT NULL,
            source TEXT DEFAULT 'user', ts TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
            bad_ans TEXT, good_ans TEXT, reason TEXT DEFAULT '', ts TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS bench_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL,
            overall REAL NOT NULL, by_type TEXT NOT NULL, ts TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_dpo_ts ON dpo_pairs(ts);
        CREATE INDEX IF NOT EXISTS idx_corr_ts ON corrections(ts);
        CREATE INDEX IF NOT EXISTS idx_bench_ts ON bench_history(ts);
    """)
    conn.commit()
    conn.close()

# Init SQLite DB now that DR is defined
_init_db(_DB_FILE)

def audit(event: str, data: Optional[dict] = None):
    """Structured JSON audit log — machine-queryable unlike plain text."""
    """Append structured event to audit log."""
    record = {"ts": datetime.now().isoformat(), "event": event, **(data or {})}
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Never crash on audit failure

log.info(f"Workspace: {WORKSPACE}")



# ────────────────────────────────────────────────────────────
# § 5C  UNIFIED EXCEPTION HIERARCHY (v12.1)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5c  UNIFIED EXCEPTION HIERARCHY (v12.1)
# ════════════════════════════════════════════════════════════════════

class MiniGrokError(Exception):
    """Base exception for all MiniGrok errors. Carries a user-friendly message."""
    def __init__(self, msg: str, detail: str = ""):
        super().__init__(msg)
        self.detail = detail

class ToolError(MiniGrokError):
    """Raised when a tool call fails (web search, browser, code exec, etc.)."""

class ModelError(MiniGrokError):
    """Raised when model inference fails (OOM, corrupted input, etc.)."""

class StorageError(MiniGrokError):
    """Raised when reading/writing persistent storage fails."""

class SecurityError(MiniGrokError):
    """Raised when a security check blocks execution."""


