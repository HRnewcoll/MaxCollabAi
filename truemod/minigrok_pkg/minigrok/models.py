"""
minigrok.models
Model loading, VRAM juggler, model router
"""


# Cross-module imports
import os, json, re, logging, threading, shutil, time
import warnings
from pathlib import Path
from typing import Optional, Generator, List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

import torch
from transformers import (AutoProcessor, BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer)
from peft import PeftModel
from huggingface_hub import snapshot_download

from .base import log, APP_NAME, APP_VERSION
from .config import Config, DIRS, DR, MODEL_SEARCH, MODEL_DIR


# ════════════════════════════════════════════════════════════════════
# § 5  MODEL LOADING (thread-safe)
# ════════════════════════════════════════════════════════════════════

_model_lock = threading.RLock()  # RLock: reentrant, prevents nested deadlocks
def _vram_free():
    return (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9

# ═══════════════════════════════════════════════════════════════════
# Multi-model system state (v12.1)
# ═══════════════════════════════════════════════════════════════════
_active_model_info: dict = {}   # Tracks which model is currently loaded
_fast_model        = None        # Optional small model for simple queries
_fast_processor    = None
_fast_model_info: dict = {}

def _make_bnb_config() -> BitsAndBytesConfig:
    """
    Dynamic quantization based on free VRAM (Kimi's recommendation).
    • >12GB free → 4-bit NF4 bfloat16  (best quality)
    • 8-12GB free → 4-bit NF4 float16  (good balance)
    • <8GB free  → 4-bit NF4 float16 + double quant (maximum compression)
    """
    free = _vram_free()
    if free > 12:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False
        )
    elif free > 8:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
        )
    else:
        log.warning(f"Low VRAM ({free:.1f}GB) — using double quantization (may affect quality)")
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )

bnb_cfg = _make_bnb_config()

processor = None
model     = None

def _load_single_model(model_id: str, model_cls_name: str, model_dir: str):
    """
    Load one model by class name. Supports:
      Qwen2_5_VLForConditionalGeneration  (VL, needs qwen_vl_utils)
      Qwen2VLForConditionalGeneration     (older VL)
      AutoModelForCausalLM                (text-only fallback)
    Returns (processor, model) or raises.
    """
    from transformers import AutoModelForCausalLM
    # Processor: always try AutoProcessor first; it handles all Qwen variants
    proc = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, padding_side="left",
        extra_special_tokens={}
    )
    tok = getattr(proc, "tokenizer", proc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pick the right model class
    if model_cls_name == "Qwen2_5_VLForConditionalGeneration":
        ModelClass = Qwen2_5_VLForConditionalGeneration
    elif model_cls_name == "Qwen2VLForConditionalGeneration":
        try:
            from transformers import Qwen2VLForConditionalGeneration
            ModelClass = Qwen2VLForConditionalGeneration
        except ImportError:
            ModelClass = AutoModelForCausalLM
    else:
        ModelClass = AutoModelForCausalLM

    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            torch.cuda.empty_cache()
            mdl = ModelClass.from_pretrained(
                model_dir, device_map="auto", quantization_config=bnb_cfg,
                dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn
            )
            mdl.config.use_cache = False
            log.info(f"  Loaded {model_id} | attn={attn}")
            return proc, mdl
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:
            log.debug(f"  {model_id} attn={attn} failed: {e}")
    raise RuntimeError(f"Could not load {model_id} with any attention backend.")


def _load_model(model_dir: str):
    """
    Load the primary model from model_dir.
    Called at startup — doesn't try fallbacks (fallback download loop does that).
    """
    free = _vram_free()
    if free < Config.MIN_VRAM_GB:
        torch.cuda.empty_cache()
        if _vram_free() < Config.MIN_VRAM_GB:
            raise RuntimeError(f"❌ Only {_vram_free():.1f}GB free, need {Config.MIN_VRAM_GB}GB.")
    # Determine model class from Config.MODEL_ID
    cls_name = "Qwen2_5_VLForConditionalGeneration"
    for entry in Config.MODEL_FALLBACK_CHAIN:
        if entry["id"] == Config.MODEL_ID:
            cls_name = entry["cls"]
            break
    proc, mdl = _load_single_model(Config.MODEL_ID, cls_name, model_dir)
    _active_model_info.update({
        "id": Config.MODEL_ID, "vision": True, "dir": model_dir
    })
    for entry in Config.MODEL_FALLBACK_CHAIN:
        if entry["id"] == Config.MODEL_ID:
            _active_model_info["vision"] = entry["vision"]
            break
    return proc, mdl


def _try_load_fast_model():
    """
    Lazily load the fast small model for simple queries.
    Called on first use, not at startup.
    Only loaded if enough VRAM is free after primary model.
    """
    global _fast_model, _fast_processor, _fast_model_info
    if _fast_model is not None:
        return True
    if not Config.FAST_MODEL_ID:
        return False
    # Find in fallback chain
    fast_entry = next(
        (e for e in Config.MODEL_FALLBACK_CHAIN if e["id"] == Config.FAST_MODEL_ID), None
    )
    if fast_entry is None:
        return False
    needed = fast_entry["size_gb"]
    if _vram_free() < needed + 0.5:
        log.info(f"Not enough VRAM for fast model ({_vram_free():.1f}GB free, need {needed}GB)")
        return False
    try:
        fast_dir = DIRS["model"] + "_fast"
        if not any(Path(fast_dir).rglob("*.safetensors")):
            log.info(f"Downloading fast model {Config.FAST_MODEL_ID}…")
            from huggingface_hub import snapshot_download
            snapshot_download(Config.FAST_MODEL_ID, local_dir=fast_dir,
                              ignore_patterns=["*.pt","*.bin"])
        _fast_processor, _fast_model = _load_single_model(
            Config.FAST_MODEL_ID, fast_entry["cls"], fast_dir
        )
        _fast_model.eval()
        _fast_model_info = fast_entry
        log.info(f"✅ Fast model ready: {Config.FAST_MODEL_ID}")
        return True
    except Exception as e:
        log.warning(f"Fast model load failed: {e}")
        return False

# Find and load model
_found = next((p for p in MODEL_SEARCH if Path(p).exists() and any(Path(p).rglob("*.safetensors"))), None)
_local = any(Path(MODEL_DIR).rglob("*.safetensors"))

if _local:
    log.info("Model found locally")
elif _found:
    log.info(f"Copying model from {_found}…")
    shutil.copytree(_found, MODEL_DIR, dirs_exist_ok=True)
else:
    for mid in [Config.MODEL_ID] + Config.MODEL_FALLBACKS:
        try:
            log.info(f"Downloading {mid}…")
            snapshot_download(mid, local_dir=MODEL_DIR, ignore_patterns=["*.pt", "*.bin"])
            shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
            break
        except Exception as e:
            log.warning(f"  {e}")

log.info("Loading model…")
_MODEL_LOADED = False
for _fb_entry in Config.MODEL_FALLBACK_CHAIN:
    _fb_id  = _fb_entry["id"]
    _fb_cls = _fb_entry["cls"]
    _fb_gb  = _fb_entry["size_gb"]
    # Check if we have it locally or can use the existing MODEL_DIR for primary
    _try_dir = MODEL_DIR if _fb_id == Config.MODEL_ID else (DIRS["model"] + f"_fb_{_fb_id.replace('/','_')}")
    _have_local = any(Path(_try_dir).rglob("*.safetensors"))
    if not _have_local and _vram_free() < _fb_gb + 0.5:
        log.warning(f"Skipping {_fb_id}: need {_fb_gb}GB VRAM, have {_vram_free():.1f}GB")
        continue
    if not _have_local:
        try:
            log.info(f"Downloading {_fb_id}…")
            snapshot_download(_fb_id, local_dir=_try_dir, ignore_patterns=["*.pt","*.bin"])
            shutil.copytree(_try_dir, f"{DR}/model", dirs_exist_ok=True)
        except Exception as _e:
            log.warning(f"Download {_fb_id} failed: {_e}")
            continue
    try:
        processor, model = _load_single_model(_fb_id, _fb_cls, _try_dir)
        model.eval()
        Config.MODEL_ID = _fb_id   # Update active model ID
        _active_model_info.update({"id": _fb_id, "vision": _fb_entry["vision"], "dir": _try_dir})
        log.info(f"✅ Active model: {_fb_id} (vision={_fb_entry['vision']})")
        _MODEL_LOADED = True
        break
    except Exception as _e:
        log.warning(f"Load {_fb_id} failed: {_e}")
        torch.cuda.empty_cache()

if not _MODEL_LOADED:
    raise RuntimeError("❌ All models in fallback chain failed to load. Check VRAM and network.")

MODEL_MAX_TOKENS = getattr(model.config, "max_position_embeddings", 32768)
log.info(f"Model ready | {Config.MODEL_ID} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB | Context: {MODEL_MAX_TOKENS:,} tokens")

# v13: Pre-load fast model in background so first simple query has no cold-start lag
def _preload_fast_model_bg():
    """Background thread: load fast model after primary settles."""
    time.sleep(30)   # Wait for system to stabilise
    if _fast_model is None and Config.FAST_MODEL_ID:
        log.info("Background: pre-loading fast model…")
        _try_load_fast_model()

threading.Thread(target=_preload_fast_model_bg, daemon=True, name="FastModelPreload").start()



# ────────────────────────────────────────────────────────────
# § 5B  VRAM JUGGLER  (v11 — prevents OOM when switching between models)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5b  VRAM JUGGLER  (v11 — prevents OOM when switching between models)
#
#  On a T4 (16GB), you cannot have Qwen-VL (6GB) + SD-Turbo (4GB) +
#  Whisper (1GB) + embedder (0.5GB) all in VRAM simultaneously.
#  The Juggler offloads inactive models to CPU RAM automatically.
#
#  Usage:
#    with VRAMJuggler.use("qwen"):    # Qwen is active, others offloaded
#    with VRAMJuggler.use("sd"):      # SD is active, Qwen offloaded
# ════════════════════════════════════════════════════════════════════

class _VRAMJuggler:
    """Manages VRAM by offloading models when not in use."""
    MODELS = {}          # name → (model_object, is_in_vram)
    _current = None
    _lock = threading.RLock()  # RLock: reentrant

    def register(self, name: str, model_obj, priority: int = 1):
        """Register a model with the juggler."""
        with self._lock:
            self.MODELS[name] = {"obj": model_obj, "in_vram": True, "priority": priority}

    def offload(self, name: str):
        """Move a model from VRAM to CPU RAM."""
        entry = self.MODELS.get(name)
        if entry and entry["in_vram"] and hasattr(entry["obj"], "to"):
            try:
                entry["obj"].to("cpu")
                entry["in_vram"] = False
                torch.cuda.empty_cache()
                log.debug(f"VRAM Juggler: offloaded '{name}' to CPU")
            except Exception as e:
                log.warning(f"VRAM Juggler: could not offload '{name}': {e}")

    def load_to_gpu(self, name: str):
        """Move a model from CPU back to VRAM."""
        entry = self.MODELS.get(name)
        if entry and not entry["in_vram"] and hasattr(entry["obj"], "to"):
            try:
                entry["obj"].to("cuda")
                entry["in_vram"] = True
                log.debug(f"VRAM Juggler: loaded '{name}' to VRAM")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Force offload lowest priority models then retry
                self._emergency_offload(exclude=name)
                entry["obj"].to("cuda")
                entry["in_vram"] = True

    def _emergency_offload(self, exclude: str):
        """Offload all models except the one being loaded."""
        for name, entry in self.MODELS.items():
            if name != exclude and entry["in_vram"]:
                self.offload(name)

    def use(self, name: str):
        """Context manager: ensure named model is in VRAM."""
        juggler = self
        class _Ctx:
            def __enter__(self_):
                with juggler._lock:
                    if juggler._current != name:
                        # Offload the previous active model
                        if juggler._current and juggler._current in juggler.MODELS:
                            juggler.offload(juggler._current)
                        juggler.load_to_gpu(name)
                        juggler._current = name
                return self
            def __exit__(self_, *_):
                pass  # Stay in VRAM until next use()
        return _Ctx()

    def free_all(self):
        """Release all non-main models from VRAM."""
        with self._lock:
            for name in list(self.MODELS.keys()):
                if name != "qwen":
                    self.offload(name)

VRAMJuggler = _VRAMJuggler()
# Register main model immediately
VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}

log.info("VRAM Juggler ready")



# ────────────────────────────────────────────────────────────
# § 5D  MODEL ROUTER (v12.1)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5d  MODEL ROUTER (v12.1)
#
# Routes queries to the fast small model (Qwen-3B) when the query is
# simple (short, conversational, no tool use needed).
# Falls back to primary model (VL-7B) for everything else.
# Saves ~400ms per simple response on T4.
# ════════════════════════════════════════════════════════════════════

_COMPLEX_PATTERNS = re.compile(
    r'research|explain|analyse|analyze|compare|write|code|debug|solve|calculate|'
    r'essay|homework|summarise|summarize|paper|generate|create|build|implement|'
    r'what is .{20,}|how does|why does|tell me about',
    re.IGNORECASE
)

def _is_simple_query(query: str) -> bool:
    """
    Returns True if the query is simple enough for the fast small model.
    Criteria: short (< FAST_MODEL_THRESH words) AND no complex patterns.
    """
    if not Config.FAST_MODEL_ID:
        return False
    words = query.split()
    if len(words) > Config.FAST_MODEL_THRESH:
        return False
    if _COMPLEX_PATTERNS.search(query):
        return False
    # Never use fast model if image is being processed (no vision support)
    return True


def quick_routed(prompt: str, system: str = "", max_tokens: int = 512,
                  temp: float = 0.5, image_path: Optional[str] = None) -> str:
    """
    v12.1: Route to fast model if query is simple, primary model otherwise.
    Transparent to callers — same signature as quick().
    """
    use_fast = (image_path is None) and _is_simple_query(prompt) and _try_load_fast_model()

    if use_fast and _fast_model is not None:
        try:
            msgs = ([{"role":"system","content":system}] if system else []) +                    [{"role":"user","content":prompt}]
            with _model_lock:
                text = _fast_processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                inputs = _fast_processor(text=[text], return_tensors="pt").to(_fast_model.device)
                tok = getattr(_fast_processor, "tokenizer", _fast_processor)
                out = _fast_model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=max(float(temp), 0.01), top_p=0.9,
                    do_sample=temp > 0.05, repetition_penalty=1.1,
                    pad_token_id=tok.eos_token_id
                )
                ids = out[0][inputs["input_ids"].shape[1]:]
                return tok.decode(ids, skip_special_tokens=True)
        except Exception as e:
            log.debug(f"Fast model failed, falling back: {e}")

    # Default: use primary model
    msgs = ([{"role":"system","content":system}] if system else []) +            [{"role":"user","content":prompt}]
    return "".join(stream_gen(msgs, max_new_tokens=max_tokens,
                               temperature=temp, image_path=image_path))




# ── Inference helpers (live here to avoid circular imports) ────────────
def stream_gen(messages, max_new_tokens=1024, temperature=0.7, image_path=None):
    """Stream tokens from the model with thread-safe inference."""
    from typing import Generator
    with _model_lock:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image_path and Path(image_path).exists():
            try:
                from qwen_vl_utils import process_vision_info
                vmsg = [{"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": messages[-1]["content"]}
                ]}]
                imgs, _ = process_vision_info(vmsg)
                inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt").to(model.device)
            except Exception:
                inputs = processor(text=[text], return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
        tok = getattr(processor, "tokenizer", processor)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=120)
        thread = threading.Thread(target=model.generate, kwargs=dict(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(float(temperature), 0.01),
            top_p=0.9, do_sample=temperature > 0.05,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id, streamer=streamer
        ), daemon=True)
        thread.start()
        tokens = []
        for token in streamer:
            tokens.append(token)
        thread.join(timeout=20)
    for token in tokens:
        yield token


def quick(prompt: str, system: str = "", max_tokens: int = 512,
          temp: float = 0.5, image_path=None) -> str:
    """Quick non-streaming generation."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    return "".join(stream_gen(msgs, max_new_tokens=max_tokens, temperature=temp, image_path=image_path))
"""
minigrok.models
Model loading, VRAM juggler, model router
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
from .config import Config, DIRS
from .generation import quick, stream_gen

# ────────────────────────────────────────────────────────────
# § 5  MODEL LOADING (thread-safe)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5  MODEL LOADING (thread-safe)
# ════════════════════════════════════════════════════════════════════

_model_lock = threading.RLock()  # RLock: reentrant, prevents nested deadlocks
def _vram_free():
    return (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9

# ═══════════════════════════════════════════════════════════════════
# Multi-model system state (v12.1)
# ═══════════════════════════════════════════════════════════════════
_active_model_info: dict = {}   # Tracks which model is currently loaded
_fast_model        = None        # Optional small model for simple queries
_fast_processor    = None
_fast_model_info: dict = {}

def _make_bnb_config() -> BitsAndBytesConfig:
    """
    Dynamic quantization based on free VRAM (Kimi's recommendation).
    • >12GB free → 4-bit NF4 bfloat16  (best quality)
    • 8-12GB free → 4-bit NF4 float16  (good balance)
    • <8GB free  → 4-bit NF4 float16 + double quant (maximum compression)
    """
    free = _vram_free()
    if free > 12:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False
        )
    elif free > 8:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
        )
    else:
        log.warning(f"Low VRAM ({free:.1f}GB) — using double quantization (may affect quality)")
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )

bnb_cfg = _make_bnb_config()

processor = None
model     = None

def _load_single_model(model_id: str, model_cls_name: str, model_dir: str):
    """
    Load one model by class name. Supports:
      Qwen2_5_VLForConditionalGeneration  (VL, needs qwen_vl_utils)
      Qwen2VLForConditionalGeneration     (older VL)
      AutoModelForCausalLM                (text-only fallback)
    Returns (processor, model) or raises.
    """
    from transformers import AutoModelForCausalLM
    # Processor: always try AutoProcessor first; it handles all Qwen variants
    proc = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, padding_side="left",
        extra_special_tokens={}
    )
    tok = getattr(proc, "tokenizer", proc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pick the right model class
    if model_cls_name == "Qwen2_5_VLForConditionalGeneration":
        ModelClass = Qwen2_5_VLForConditionalGeneration
    elif model_cls_name == "Qwen2VLForConditionalGeneration":
        try:
            from transformers import Qwen2VLForConditionalGeneration
            ModelClass = Qwen2VLForConditionalGeneration
        except ImportError:
            ModelClass = AutoModelForCausalLM
    else:
        ModelClass = AutoModelForCausalLM

    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            torch.cuda.empty_cache()
            mdl = ModelClass.from_pretrained(
                model_dir, device_map="auto", quantization_config=bnb_cfg,
                dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn
            )
            mdl.config.use_cache = False
            log.info(f"  Loaded {model_id} | attn={attn}")
            return proc, mdl
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:
            log.debug(f"  {model_id} attn={attn} failed: {e}")
    raise RuntimeError(f"Could not load {model_id} with any attention backend.")


def _load_model(model_dir: str):
    """
    Load the primary model from model_dir.
    Called at startup — doesn't try fallbacks (fallback download loop does that).
    """
    free = _vram_free()
    if free < Config.MIN_VRAM_GB:
        torch.cuda.empty_cache()
        if _vram_free() < Config.MIN_VRAM_GB:
            raise RuntimeError(f"❌ Only {_vram_free():.1f}GB free, need {Config.MIN_VRAM_GB}GB.")
    # Determine model class from Config.MODEL_ID
    cls_name = "Qwen2_5_VLForConditionalGeneration"
    for entry in Config.MODEL_FALLBACK_CHAIN:
        if entry["id"] == Config.MODEL_ID:
            cls_name = entry["cls"]
            break
    proc, mdl = _load_single_model(Config.MODEL_ID, cls_name, model_dir)
    _active_model_info.update({
        "id": Config.MODEL_ID, "vision": True, "dir": model_dir
    })
    for entry in Config.MODEL_FALLBACK_CHAIN:
        if entry["id"] == Config.MODEL_ID:
            _active_model_info["vision"] = entry["vision"]
            break
    return proc, mdl


def _try_load_fast_model():
    """
    Lazily load the fast small model for simple queries.
    Called on first use, not at startup.
    Only loaded if enough VRAM is free after primary model.
    """
    global _fast_model, _fast_processor, _fast_model_info
    if _fast_model is not None:
        return True
    if not Config.FAST_MODEL_ID:
        return False
    # Find in fallback chain
    fast_entry = next(
        (e for e in Config.MODEL_FALLBACK_CHAIN if e["id"] == Config.FAST_MODEL_ID), None
    )
    if fast_entry is None:
        return False
    needed = fast_entry["size_gb"]
    if _vram_free() < needed + 0.5:
        log.info(f"Not enough VRAM for fast model ({_vram_free():.1f}GB free, need {needed}GB)")
        return False
    try:
        fast_dir = DIRS["model"] + "_fast"
        if not any(Path(fast_dir).rglob("*.safetensors")):
            log.info(f"Downloading fast model {Config.FAST_MODEL_ID}…")
            from huggingface_hub import snapshot_download
            snapshot_download(Config.FAST_MODEL_ID, local_dir=fast_dir,
                              ignore_patterns=["*.pt","*.bin"])
        _fast_processor, _fast_model = _load_single_model(
            Config.FAST_MODEL_ID, fast_entry["cls"], fast_dir
        )
        _fast_model.eval()
        _fast_model_info = fast_entry
        log.info(f"✅ Fast model ready: {Config.FAST_MODEL_ID}")
        return True
    except Exception as e:
        log.warning(f"Fast model load failed: {e}")
        return False

# Find and load model
_found = next((p for p in MODEL_SEARCH if Path(p).exists() and any(Path(p).rglob("*.safetensors"))), None)
_local = any(Path(MODEL_DIR).rglob("*.safetensors"))

if _local:
    log.info("Model found locally")
elif _found:
    log.info(f"Copying model from {_found}…")
    shutil.copytree(_found, MODEL_DIR, dirs_exist_ok=True)
else:
    for mid in [Config.MODEL_ID] + Config.MODEL_FALLBACKS:
        try:
            log.info(f"Downloading {mid}…")
            snapshot_download(mid, local_dir=MODEL_DIR, ignore_patterns=["*.pt", "*.bin"])
            shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
            break
        except Exception as e:
            log.warning(f"  {e}")

log.info("Loading model…")
_MODEL_LOADED = False
for _fb_entry in Config.MODEL_FALLBACK_CHAIN:
    _fb_id  = _fb_entry["id"]
    _fb_cls = _fb_entry["cls"]
    _fb_gb  = _fb_entry["size_gb"]
    # Check if we have it locally or can use the existing MODEL_DIR for primary
    _try_dir = MODEL_DIR if _fb_id == Config.MODEL_ID else (DIRS["model"] + f"_fb_{_fb_id.replace('/','_')}")
    _have_local = any(Path(_try_dir).rglob("*.safetensors"))
    if not _have_local and _vram_free() < _fb_gb + 0.5:
        log.warning(f"Skipping {_fb_id}: need {_fb_gb}GB VRAM, have {_vram_free():.1f}GB")
        continue
    if not _have_local:
        try:
            log.info(f"Downloading {_fb_id}…")
            snapshot_download(_fb_id, local_dir=_try_dir, ignore_patterns=["*.pt","*.bin"])
            shutil.copytree(_try_dir, f"{DR}/model", dirs_exist_ok=True)
        except Exception as _e:
            log.warning(f"Download {_fb_id} failed: {_e}")
            continue
    try:
        processor, model = _load_single_model(_fb_id, _fb_cls, _try_dir)
        model.eval()
        Config.MODEL_ID = _fb_id   # Update active model ID
        _active_model_info.update({"id": _fb_id, "vision": _fb_entry["vision"], "dir": _try_dir})
        log.info(f"✅ Active model: {_fb_id} (vision={_fb_entry['vision']})")
        _MODEL_LOADED = True
        break
    except Exception as _e:
        log.warning(f"Load {_fb_id} failed: {_e}")
        torch.cuda.empty_cache()

if not _MODEL_LOADED:
    raise RuntimeError("❌ All models in fallback chain failed to load. Check VRAM and network.")

MODEL_MAX_TOKENS = getattr(model.config, "max_position_embeddings", 32768)
log.info(f"Model ready | {Config.MODEL_ID} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB | Context: {MODEL_MAX_TOKENS:,} tokens")

# v13: Pre-load fast model in background so first simple query has no cold-start lag
def _preload_fast_model_bg():
    """Background thread: load fast model after primary settles."""
    time.sleep(30)   # Wait for system to stabilise
    if _fast_model is None and Config.FAST_MODEL_ID:
        log.info("Background: pre-loading fast model…")
        _try_load_fast_model()

threading.Thread(target=_preload_fast_model_bg, daemon=True, name="FastModelPreload").start()



# ────────────────────────────────────────────────────────────
# § 5B  VRAM JUGGLER  (v11 — prevents OOM when switching between models)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5b  VRAM JUGGLER  (v11 — prevents OOM when switching between models)
#
#  On a T4 (16GB), you cannot have Qwen-VL (6GB) + SD-Turbo (4GB) +
#  Whisper (1GB) + embedder (0.5GB) all in VRAM simultaneously.
#  The Juggler offloads inactive models to CPU RAM automatically.
#
#  Usage:
#    with VRAMJuggler.use("qwen"):    # Qwen is active, others offloaded
#    with VRAMJuggler.use("sd"):      # SD is active, Qwen offloaded
# ════════════════════════════════════════════════════════════════════

class _VRAMJuggler:
    """Manages VRAM by offloading models when not in use."""
    MODELS = {}          # name → (model_object, is_in_vram)
    _current = None
    _lock = threading.RLock()  # RLock: reentrant

    def register(self, name: str, model_obj, priority: int = 1):
        """Register a model with the juggler."""
        with self._lock:
            self.MODELS[name] = {"obj": model_obj, "in_vram": True, "priority": priority}

    def offload(self, name: str):
        """Move a model from VRAM to CPU RAM."""
        entry = self.MODELS.get(name)
        if entry and entry["in_vram"] and hasattr(entry["obj"], "to"):
            try:
                entry["obj"].to("cpu")
                entry["in_vram"] = False
                torch.cuda.empty_cache()
                log.debug(f"VRAM Juggler: offloaded '{name}' to CPU")
            except Exception as e:
                log.warning(f"VRAM Juggler: could not offload '{name}': {e}")

    def load_to_gpu(self, name: str):
        """Move a model from CPU back to VRAM."""
        entry = self.MODELS.get(name)
        if entry and not entry["in_vram"] and hasattr(entry["obj"], "to"):
            try:
                entry["obj"].to("cuda")
                entry["in_vram"] = True
                log.debug(f"VRAM Juggler: loaded '{name}' to VRAM")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Force offload lowest priority models then retry
                self._emergency_offload(exclude=name)
                entry["obj"].to("cuda")
                entry["in_vram"] = True

    def _emergency_offload(self, exclude: str):
        """Offload all models except the one being loaded."""
        for name, entry in self.MODELS.items():
            if name != exclude and entry["in_vram"]:
                self.offload(name)

    def use(self, name: str):
        """Context manager: ensure named model is in VRAM."""
        juggler = self
        class _Ctx:
            def __enter__(self_):
                with juggler._lock:
                    if juggler._current != name:
                        # Offload the previous active model
                        if juggler._current and juggler._current in juggler.MODELS:
                            juggler.offload(juggler._current)
                        juggler.load_to_gpu(name)
                        juggler._current = name
                return self
            def __exit__(self_, *_):
                pass  # Stay in VRAM until next use()
        return _Ctx()

    def free_all(self):
        """Release all non-main models from VRAM."""
        with self._lock:
            for name in list(self.MODELS.keys()):
                if name != "qwen":
                    self.offload(name)

VRAMJuggler = _VRAMJuggler()
# Register main model immediately
VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}

log.info("VRAM Juggler ready")



# ────────────────────────────────────────────────────────────
# § 5D  MODEL ROUTER (v12.1)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5d  MODEL ROUTER (v12.1)
#
# Routes queries to the fast small model (Qwen-3B) when the query is
# simple (short, conversational, no tool use needed).
# Falls back to primary model (VL-7B) for everything else.
# Saves ~400ms per simple response on T4.
# ════════════════════════════════════════════════════════════════════

_COMPLEX_PATTERNS = re.compile(
    r'research|explain|analyse|analyze|compare|write|code|debug|solve|calculate|'
    r'essay|homework|summarise|summarize|paper|generate|create|build|implement|'
    r'what is .{20,}|how does|why does|tell me about',
    re.IGNORECASE
)

def _is_simple_query(query: str) -> bool:
    """
    Returns True if the query is simple enough for the fast small model.
    Criteria: short (< FAST_MODEL_THRESH words) AND no complex patterns.
    """
    if not Config.FAST_MODEL_ID:
        return False
    words = query.split()
    if len(words) > Config.FAST_MODEL_THRESH:
        return False
    if _COMPLEX_PATTERNS.search(query):
        return False
    # Never use fast model if image is being processed (no vision support)
    return True


def quick_routed(prompt: str, system: str = "", max_tokens: int = 512,
                  temp: float = 0.5, image_path: Optional[str] = None) -> str:
    """
    v12.1: Route to fast model if query is simple, primary model otherwise.
    Transparent to callers — same signature as quick().
    """
    use_fast = (image_path is None) and _is_simple_query(prompt) and _try_load_fast_model()

    if use_fast and _fast_model is not None:
        try:
            msgs = ([{"role":"system","content":system}] if system else []) +                    [{"role":"user","content":prompt}]
            with _model_lock:
                text = _fast_processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                inputs = _fast_processor(text=[text], return_tensors="pt").to(_fast_model.device)
                tok = getattr(_fast_processor, "tokenizer", _fast_processor)
                out = _fast_model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=max(float(temp), 0.01), top_p=0.9,
                    do_sample=temp > 0.05, repetition_penalty=1.1,
                    pad_token_id=tok.eos_token_id
                )
                ids = out[0][inputs["input_ids"].shape[1]:]
                return tok.decode(ids, skip_special_tokens=True)
        except Exception as e:
            log.debug(f"Fast model failed, falling back: {e}")

    # Default: use primary model
    msgs = ([{"role":"system","content":system}] if system else []) +            [{"role":"user","content":prompt}]
    return "".join(stream_gen(msgs, max_new_tokens=max_tokens,
                               temperature=temp, image_path=image_path))


