"""
minigrok.chain
Multi-step tool chaining
"""


# Cross-module imports
import os, json, re, logging, threading
import warnings
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

from .base import log
from .config import Config, DIRS
from .models import quick


# ════════════════════════════════════════════════════════════════════
# § 17  MULTI-STEP TOOL CHAINING
# ════════════════════════════════════════════════════════════════════

def plan_tool_chain(goal: str, max_steps: int = 8) -> list:
    raw = quick(
        f"Break this goal into tool call sequence (max {max_steps} steps).\nGOAL: {goal}\n\n"
        f"Return JSON array. Each: {{\"step\":1,\"tool\":\"name\",\"params\":{{}},\"depends_on\":[],\"description\":\"...\"}}\n"
        f"Use ONLY these tools: {', '.join(TOOL_REGISTRY.keys())}\n"
        f"Return ONLY the JSON array.",
        max_tokens=600, temp=0.3
    )
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return [s for s in json.loads(m.group())[:max_steps] if s.get("tool") in TOOL_REGISTRY]
    except Exception:
        pass
    return []

def execute_tool_chain(goal: str, log_fn=None) -> str:
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n🔗 Planning: {goal}")
    plan = plan_tool_chain(goal)
    if not plan:
        return quick(goal, max_tokens=600)
    _log(f"  📋 {len(plan)} steps")
    step_outputs: Dict[int, str] = {}
    for step in plan:
        sn = step["step"]
        name = step["tool"]
        params = step.get("params", {})
        deps = step.get("depends_on", [])
        _log(f"  ▶ {sn}: {name}")
        if deps:
            dep_ctx = "\n".join(f"[Step {d}]: {step_outputs.get(d, '')[:400]}" for d in deps)
            for pk in ["query", "question", "topic", "code", "content", "goal"]:
                if pk in params:
                    params[pk] = f"{params[pk]}\nContext: {dep_ctx}"
                    break
        try:
            result = TOOL_REGISTRY[name]["fn"](**params) if name in TOOL_REGISTRY else f"Unknown: {name}"
        except Exception as e:
            result = f"Error: {e}"
        step_outputs[sn] = str(result)[:2000]
        _log(f"    ✓ {str(result)[:80]}")
    all_out = "\n\n".join(f"Step {k}: {v[:500]}" for k, v in step_outputs.items())
    return quick(f"Goal:{goal}\n\nResults:\n{all_out[:3000]}\n\nSynthesise comprehensive final answer.",
                 max_tokens=800, temp=0.4)

def _is_complex(msg: str) -> bool:
    indicators = ["and then", "after that", r"compare.*and", r"research.*and.*write",
                   r"find.*and.*summarise", r"build.*and.*test"]
    return any(re.search(p, msg.lower()) for p in indicators) and len(msg.split()) > 15

log.info("Tool chaining ready")


"""
minigrok.chain
Multi-step tool chaining
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
from .generation import quick
from .registry import TOOL_REGISTRY

# ────────────────────────────────────────────────────────────
# § 17  MULTI-STEP TOOL CHAINING
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 17  MULTI-STEP TOOL CHAINING
# ════════════════════════════════════════════════════════════════════

def plan_tool_chain(goal: str, max_steps: int = 8) -> list:
    raw = quick(
        f"Break this goal into tool call sequence (max {max_steps} steps).\nGOAL: {goal}\n\n"
        f"Return JSON array. Each: {{\"step\":1,\"tool\":\"name\",\"params\":{{}},\"depends_on\":[],\"description\":\"...\"}}\n"
        f"Use ONLY these tools: {', '.join(TOOL_REGISTRY.keys())}\n"
        f"Return ONLY the JSON array.",
        max_tokens=600, temp=0.3
    )
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return [s for s in json.loads(m.group())[:max_steps] if s.get("tool") in TOOL_REGISTRY]
    except Exception:
        pass
    return []

def execute_tool_chain(goal: str, log_fn=None) -> str:
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n🔗 Planning: {goal}")
    plan = plan_tool_chain(goal)
    if not plan:
        return quick(goal, max_tokens=600)
    _log(f"  📋 {len(plan)} steps")
    step_outputs: Dict[int, str] = {}
    for step in plan:
        sn = step["step"]
        name = step["tool"]
        params = step.get("params", {})
        deps = step.get("depends_on", [])
        _log(f"  ▶ {sn}: {name}")
        if deps:
            dep_ctx = "\n".join(f"[Step {d}]: {step_outputs.get(d, '')[:400]}" for d in deps)
            for pk in ["query", "question", "topic", "code", "content", "goal"]:
                if pk in params:
                    params[pk] = f"{params[pk]}\nContext: {dep_ctx}"
                    break
        try:
            result = TOOL_REGISTRY[name]["fn"](**params) if name in TOOL_REGISTRY else f"Unknown: {name}"
        except Exception as e:
            result = f"Error: {e}"
        step_outputs[sn] = str(result)[:2000]
        _log(f"    ✓ {str(result)[:80]}")
    all_out = "\n\n".join(f"Step {k}: {v[:500]}" for k, v in step_outputs.items())
    return quick(f"Goal:{goal}\n\nResults:\n{all_out[:3000]}\n\nSynthesise comprehensive final answer.",
                 max_tokens=800, temp=0.4)

def _is_complex(msg: str) -> bool:
    indicators = ["and then", "after that", r"compare.*and", r"research.*and.*write",
                   r"find.*and.*summarise", r"build.*and.*test"]
    return any(re.search(p, msg.lower()) for p in indicators) and len(msg.split()) > 15

log.info("Tool chaining ready")


