"""
minigrok.registry
Tool registry, plugin system
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
from .agent import agent_stream
from .background import system_status
from .config import Config, DIRS
from .models import VRAMJuggler
from .tools import browser_agent, web_search
from .voice import synthesise

# ────────────────────────────────────────────────────────────
# § 18  TOOL REGISTRY + PLUGIN SYSTEM
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 18  TOOL REGISTRY + PLUGIN SYSTEM
# ════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, dict] = {
    "web_search":       {"desc": "DDG search. news=true for headlines.", "params": {"query": "str", "max_results": "int", "news": "bool"}, "fn": _t_web_search},
    "crawl":            {"desc": "Fetch page (Crawl4AI→trafilatura fallback)", "params": {"url": "str"}, "fn": _t_crawl},
    "browser":          {"desc": "Playwright: navigate|click|type|screenshot|get_text|scroll|run_js|close", "params": {"action": "str", "url": "str", "selector": "str", "text_input": "str", "js_code": "str"}, "fn": browser_action_single},
    "browser_agent":    {"desc": "AUTONOMOUS browser agent — runs until goal done and verified", "params": {"goal": "str", "max_steps": "int", "start_url": "str"}, "fn": lambda goal, max_steps=20, start_url="": browser_agent(goal, int(max_steps), start_url)["result"]},
    "run_code":         {"desc": "Execute: python|bash|sql|javascript (python=persistent kernel)", "params": {"language": "str", "code": "str", "timeout": "int"}, "fn": run_code},
    "auto_fix_loop":    {"desc": "Run code, auto-fix errors until working (up to 10 iters)", "params": {"language": "str", "code": "str"}, "fn": auto_fix_loop},
    "code_review":      {"desc": "Detailed code review with suggestions", "params": {"code": "str", "language": "str"}, "fn": code_review},
    "generate_tests":   {"desc": "Write tests and run until green", "params": {"code": "str", "language": "str"}, "fn": generate_tests},
    "generate_docs":    {"desc": "Auto-generate docstrings and README section", "params": {"code": "str", "language": "str"}, "fn": generate_docs},
    "scan_project":     {"desc": "Understand a code project's structure", "params": {"folder_path": "str"}, "fn": scan_project},
    "solve_homework":   {"desc": "Solve homework with full step-by-step workings", "params": {"question": "str", "subject": "str"}, "fn": solve_homework},
    "study_notes":      {"desc": "Generate structured study notes on a topic", "params": {"topic": "str", "subject": "str"}, "fn": study_notes},
    "flashcards":       {"desc": "Generate study flashcards", "params": {"topic": "str", "n": "int"}, "fn": generate_flashcards},
    "essay_help":       {"desc": "Write or help with an essay", "params": {"topic": "str", "essay_type": "str", "word_count": "int"}, "fn": essay_help},
    "read_paper":       {"desc": "Deep read academic PDF paper", "params": {"pdf_path": "str"}, "fn": lambda pdf_path: json.dumps(read_paper(pdf_path), indent=2)},
    "analyse_video":    {"desc": "Frame-by-frame VL analysis of video + audio transcript", "params": {"video_path": "str", "question": "str", "n_frames": "int"}, "fn": analyse_video_frames},
    "generate_image":   {"desc": "Generate image via local Stable Diffusion (free)", "params": {"prompt": "str", "negative_prompt": "str", "steps": "int"}, "fn": generate_image},
    "wikipedia":        {"desc": "Search Wikipedia", "params": {"q": "str"}, "fn": _t_wikipedia},
    "arxiv":            {"desc": "Search arXiv papers", "params": {"q": "str", "n": "int"}, "fn": _t_arxiv},
    "calculate":        {"desc": "Safe maths evaluator", "params": {"expr": "str"}, "fn": _t_calculate},
    "deep_research":    {"desc": "Multi-source research report (parallel fetch)", "params": {"topic": "str", "depth": "int"}, "fn": _t_deep_research},
    "tool_chain":       {"desc": "Plan and execute a multi-step tool sequence", "params": {"goal": "str"}, "fn": execute_tool_chain},
    "remember":         {"desc": "Store in long-term prioritized memory", "params": {"key": "str", "content": "str"}, "fn": lambda key, content: (mem_set(key, content), "✅")[1]},
    "recall":           {"desc": "Retrieve from memory/RAG", "params": {"query": "str"}, "fn": mem_get},
    "memory_status":    {"desc": "Show memory health and importance scores", "params": {}, "fn": memory_status},
    "ws_read":          {"desc": "Read file from workspace", "params": {"rel_path": "str"}, "fn": ws_read},
    "ws_write":         {"desc": "Write file to workspace", "params": {"rel_path": "str", "content": "str"}, "fn": ws_write},
    "ws_list":          {"desc": "List workspace files", "params": {"subdir": "str"}, "fn": ws_list},
    "ws_note":          {"desc": "Add note to workspace memory.md", "params": {"content": "str"}, "fn": ws_note},
    "speak":            {"desc": "AI TTS with voice cloning and emotion", "params": {"text": "str", "emotion": "str", "voice": "str"}, "fn": lambda text, emotion="auto", voice="default": f"__AUDIO__{synthesise(text, emotion, _voice_profiles.get(voice, {}).get('clip')) or ''}"},
    "rag_search":       {"desc": "Search local knowledge base", "params": {"query": "str"}, "fn": lambda query: "\n---\n".join(f"[{c['source']}] {c['text'][:300]}" for c in rag_retrieve(query)) or "No results."},
    "tree_of_thoughts":  {"desc": "Explore 3 reasoning branches, pick best", "params": {"q": "str"}, "fn": tree_of_thoughts},
    "load_adapter":     {"desc": "Load MoE domain adapter (coder/scientist/writer/analyst)", "params": {"domain": "str"}, "fn": load_moe_adapter},
    "export_chat":      {"desc": "Export current conversation as Markdown file", "params": {}, "fn": lambda: "Export triggered from UI."},
    "process_file":     {"desc": "Process uploaded file into RAG knowledge base", "params": {"file_path": "str"}, "fn": process_upload},
    "youtube_analyse":  {"desc": "Download and visually analyse a YouTube/web video (yt-dlp + VL frame sampling)", "params": {"url": "str", "question": "str"}, "fn": lambda url, question="Summarise this video": _youtube_analyse(url, question)},
}

# __all__ — public API surface for when this is imported as a module
__all__ = [
    "agent_stream", "stream_train", "dpo_train", "TOOL_REGISTRY",
    "system_status", "rag_audit", "generate_image", "synthesise",
    "rag_add", "rag_retrieve", "mem_set", "mem_get",
    "run_code", "auto_fix_loop", "solve_homework", "study_notes",
    "browser_agent", "analyse_video_frames", "process_upload",
    "record_feedback", "start_learning", "stop_learning",
    "VRAMJuggler", "Config", "APP_NAME", "APP_VERSION",
]

# Plugin system: auto-load tools from workspace/plugins/
def _load_plugins():
    """Load custom tool plugins from workspace/plugins/ directory."""
    plugin_dir = Path(DIRS["ws_plugins"])
    loaded = 0
    for plugin_file in plugin_dir.glob("*.py"):
        try:
            spec = {"__file__": str(plugin_file), "__name__": plugin_file.stem}
            code = plugin_file.read_text(encoding="utf-8")
            exec(compile(code, str(plugin_file), "exec"), spec)  # noqa
            if "TOOLS" in spec and isinstance(spec["TOOLS"], dict):
                for name, tool_def in spec["TOOLS"].items():
                    if "fn" in tool_def and "desc" in tool_def:
                        TOOL_REGISTRY[name] = tool_def
                        loaded += 1
            log.info(f"  Plugin: {plugin_file.stem} ({loaded} tools)")
        except (SyntaxError, ImportError, AttributeError) as e:
            log.warning(f"  Plugin {plugin_file.stem} syntax/import error: {e}")
        except Exception as e:
            log.warning(f"  Plugin {plugin_file.stem} unexpected error: {type(e).__name__}: {e}")
    return loaded

_plugins_loaded = _load_plugins()
log.info(f"Tool Registry: {len(TOOL_REGISTRY)} tools ({_plugins_loaded} from plugins)")


