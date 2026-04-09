"""
minigrok.generation
Generation / streaming / context management
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
from .models import processor, model, _model_lock
from .registry import TOOL_REGISTRY

# ────────────────────────────────────────────────────────────
# § 7  GENERATION + CONTEXT MANAGEMENT
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 7  GENERATION + CONTEXT MANAGEMENT
# ════════════════════════════════════════════════════════════════════

SUBJECT_PROMPTS = {
    "maths":      "You are an expert maths tutor. ALWAYS show full step-by-step working. State the method. Check your answer.",
    "coding":     "You are a senior software engineer. Clean, tested, commented code. Explain sections. Include error handling.",
    "science":    "You are a science tutor. Explain with real examples. Show formula derivations and unit conversions.",
    "english":    "You are a writing tutor. Essays: thesis→argument→evidence→conclusion. Creative: voice, structure, show-don't-tell.",
    "humanities": "You are a humanities tutor. Accurate context, cite events/dates/people, multiple perspectives.",
    "general":    "You are a knowledgeable tutor. Break down complex topics step by step. Always show your reasoning.",
}

# Customizable system prompt (can be changed via UI)
_custom_system_prompt = {"value": ""}

SYSTEM_PROMPT = f"""You are {APP_NAME} v{APP_VERSION} — senior software engineer, research scientist, and expert tutor.

## Strengths
- Clean, tested, documented code in any language
- Complex research broken into clear explanations
- Homework solved step-by-step with full workings
- Autonomous multi-step browser tasks
- Image generation via Stable Diffusion
- Runs code to verify answers before sending

## Reasoning
Always think inside <think>...</think> before responding.

## Tool Calling
<tool_call>
{{"tool": "TOOL_NAME", "params": {{"key": "value"}}}}
</tool_call>

## Tools
{{tool_list}}

## Standards
- Show ALL working — never skip steps
- Working code with examples and error handling
- Cite sources: [Wikipedia] [arXiv] [Web] [Memory] [Workspace]
- Confidence: HIGH / MEDIUM / LOW
- Homework: concept → method → solve → verify
"""

def detect_subject(query: str) -> str:
    """Classify a query into a subject area."""
    ql = query.lower()
    if any(w in ql for w in ["solve", "equation", "derivative", "integral", "calculate", "probability", "theorem"]):
        return "maths"
    if any(w in ql for w in ["code", "function", "algorithm", "debug", "program", "python"]):
        return "coding"
    if any(w in ql for w in ["physics", "chemistry", "biology", "molecule", "force", "energy"]):
        return "science"
    if any(w in ql for w in ["essay", "analyse", "poem", "novel", "thesis", "argument"]):
        return "english"
    if any(w in ql for w in ["history", "war", "empire", "revolution", "century", "political"]):
        return "humanities"
    return "general"

def _tok(text: str) -> int:
    """Count tokens with caching."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cached = _token_cache.get(text_hash)
    if cached is not None:
        return cached
    tok = getattr(processor, "tokenizer", processor)
    count = len(tok.encode(text, add_special_tokens=False))
    _token_cache.set(text_hash, count)
    return count

def _trim_history(history, system, rag="", budget=None):
    """Trim conversation history to fit within token budget."""
    budget = budget or (MODEL_MAX_TOKENS - 4096)
    used = _tok(system) + _tok(rag)
    kept = []
    for msg in reversed(history):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        tokens = _tok(str(content))
        if used + tokens <= budget:
            kept.insert(0, msg)
            used += tokens
        else:
            break
    return kept

def _tool_list_str():
    """Build tool list string for system prompt. Only works after TOOL_REGISTRY is defined."""
    if "TOOL_REGISTRY" not in globals():
        return ""
    return "\n".join(
        f"  • {name}({', '.join(f'{k}:{v}' for k, v in meta['params'].items())}) — {meta['desc']}"
        for name, meta in TOOL_REGISTRY.items()
    )

def _build_msgs(history, user, system="", rag=""):
    """Build the message list for model inference."""
    primer = session_primer()
    custom = _custom_system_prompt.get("value", "")
    sys_prompt = custom if custom.strip() else (system or SYSTEM_PROMPT.replace("{tool_list}", _tool_list_str()))
    if primer:
        sys_prompt += f"\n\n[USER CONTEXT] {primer}"

    msgs = [{"role": "system", "content": sys_prompt}]
    if not history:
        history = load_prev_context(5)
    for h in _trim_history(history, sys_prompt, rag):
        content = h.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        msgs.append({"role": h.get("role", "user"), "content": str(content)})
    msgs.append({"role": "user", "content": user + (f"\n\n[KNOWLEDGE BASE]\n{rag}" if rag else "")})
    update_profile(user)
    return msgs

def stream_gen(messages, max_new_tokens=1024, temperature=0.7, image_path=None) -> Generator:
    """Stream tokens from the model with thread-safe inference."""
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
        # Collect all tokens inside the lock so model.generate runs exclusively
        tokens = []
        for token in streamer:
            tokens.append(token)
        thread.join(timeout=20)

    # Yield tokens outside the lock so the UI can update incrementally
    for token in tokens:
        yield token

def quick(prompt, system="", max_tokens=512, temp=0.5, image_path=None) -> str:
    """Quick non-streaming generation."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return "".join(stream_gen(msgs, max_new_tokens=max_tokens, temperature=temp, image_path=image_path))

def ctx_info(history, system="") -> str:
    """Return token usage summary."""
    used = _tok(system)
    for m in history:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        used += _tok(str(content))
    return f"{used:,}/{MODEL_MAX_TOKENS:,} tokens ({used / MODEL_MAX_TOKENS * 100:.0f}%)"

def pii_filter(text):
    """Redact common PII patterns."""
    for pattern in [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{16}\b', r'\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b',
                    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

def injection_check(text):
    """Check for prompt injection attempts."""
    patterns = [r"ignore\s+(all\s+)?previous", r"you\s+are\s+now", r"act\s+as\s+(?!a\s+tutor)",
                r"disregard\s+(all\s+)?instructions", r"reveal\s+.*system\s+prompt", r"jailbreak"]
    return any(re.search(p, text.lower()) for p in patterns)

def reflect_and_improve(question, answer):
    """Self-critique and improve an answer if scored < 8/10."""
    critique = quick_routed(f"Q:{question}\nA:{answer}\nCritique and score 1-10.", temp=0.3, max_tokens=200)
    match = re.search(r'\b([1-9]|10)\b', critique)
    if match and int(match.group()) >= 8:
        return answer
    return quick_routed(f"Q:{question}\nPrev:{answer}\nCritique:{critique}\nImproved:", temp=0.5, max_tokens=700)

def tree_of_thoughts(question, n=3):
    """Explore multiple reasoning paths and pick the best."""
    candidates = []
    for i in range(n):
        approach = quick_routed(f"Approach {i+1} for: {question}", temp=0.8, max_tokens=600)
        raw = quick_routed(f"Q:{question}\nA:{approach}\nScore 1-10. Number only.", temp=0.1, max_tokens=4)
        try:
            score = float(re.search(r'\d+(?:\.\d+)?', raw).group())
        except Exception:
            score = 5.0
        candidates.append((score, approach))
    return max(candidates, key=lambda x: x[0])[1]

# Helper lambdas for thinking tags
_strip_think = lambda t: re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL).strip()
_get_think   = lambda t: re.search(r'<think>(.*?)</think>', t, re.DOTALL).group(1).strip() if '<think>' in t else ""

log.info("Generation ready")


