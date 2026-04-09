"""
minigrok.agent
Agent state machine, SQLite DPO, main loop
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
from .generation import quick, stream_gen, _build_msgs
from .models import model
from .registry import TOOL_REGISTRY

# ────────────────────────────────────────────────────────────
# § 5C2  AGENT STATE MACHINE (v13)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5c2  AGENT STATE MACHINE (v13)
#
# Formal states for the agent orchestration loop.
# Makes the execution path observable and testable.
# ════════════════════════════════════════════════════════════════════

class AgentPhase(Enum):
    """Enum representing which phase of the agent loop is active."""
    IDLE        = "idle"
    SECURITY    = "security"      # Rate limit + injection check + PII
    CACHE_CHECK = "cache_check"   # Check response cache
    RETRIEVAL   = "retrieval"     # RAG + memory lookup
    UNCERTAINTY = "uncertainty"   # Confidence gate
    REASONING   = "reasoning"     # LLM generation
    TOOL_EXEC   = "tool_exec"     # Tool call execution
    VERIFICATION= "verification"  # Hallucination check
    REFLECTION  = "reflection"    # Self-critique + improve
    DONE        = "done"          # Response ready


# ────────────────────────────────────────────────────────────
# § 5E  SQLITE FOR DPO + CORRECTIONS (v13)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 5e  SQLITE FOR DPO + CORRECTIONS (v13)
#
# _get_db() and _init_db() are defined early (§3 forward defs) so they
# can be called as soon as DR is known. The helpers below (db_add_*,
# db_get_*) are the public API used throughout the rest of the file.
# ════════════════════════════════════════════════════════════════════

_DB_PATH: Optional[str] = None   # Set by _init_db(); forward def already ran at § 3

def db_add_dpo(prompt: str, chosen: str, rejected: str, source: str = "user"):
    """Insert a DPO pair into SQLite."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO dpo_pairs (prompt,chosen,rejected,source,ts) VALUES (?,?,?,?,?)",
            (prompt, chosen, rejected, source, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"db_add_dpo failed: {e}")

def db_get_dpo(limit: int = 500) -> list:
    """Fetch recent DPO pairs as dicts."""
    try:
        conn = _get_db()
        rows = conn.execute(
            "SELECT prompt,chosen,rejected,source,ts FROM dpo_pairs ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"db_get_dpo failed: {e}")
        return []

def db_add_correction(question: str, bad_ans: str, good_ans: str, reason: str = ""):
    """Log a self-correction to SQLite."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO corrections (question,bad_ans,good_ans,reason,ts) VALUES (?,?,?,?,?)",
            (question[:500], bad_ans[:500], good_ans[:500], reason,
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"db_add_correction failed: {e}")

def db_add_bench(label: str, overall: float, by_type: dict):
    """Store a benchmark result in SQLite."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO bench_history (label,overall,by_type,ts) VALUES (?,?,?,?)",
            (label, overall, json.dumps(by_type), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"db_add_bench failed: {e}")

def db_get_bench_history(n: int = 20) -> list:
    """Fetch recent benchmark results for trend display."""
    try:
        conn = _get_db()
        rows = conn.execute(
            "SELECT label,overall,by_type,ts FROM bench_history ORDER BY ts DESC LIMIT ?",
            (n,)
        ).fetchall()
        conn.close()
        return [{"label":r["label"],"overall":r["overall"],
                  "by_type":json.loads(r["by_type"]),"ts":r["ts"]} for r in rows]
    except Exception as e:
        log.warning(f"db_get_bench_history failed: {e}")
        return []



# ────────────────────────────────────────────────────────────
# § 13  HALLUCINATION → AUTO-DPO CLOSED LOOP
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 13  HALLUCINATION → AUTO-DPO CLOSED LOOP
# ════════════════════════════════════════════════════════════════════

_dpo_pairs: list = []
DPO_PATH = Path(f"{DR}/memory/dpo.json")
if DPO_PATH.exists():
    try:
        _dpo_pairs = json.loads(DPO_PATH.read_text())
    except Exception:
        pass

_correction_q: queue.Queue = queue.Queue()
_correction_log: list = []
CORR_PATH = Path(f"{DR}/memory/corrections.json")
if CORR_PATH.exists():
    try:
        _correction_log = json.loads(CORR_PATH.read_text())
    except Exception:
        pass

_improve_q: queue.Queue = queue.Queue()

def record_feedback(question, chosen, rejected, source="user"):
    """Record a DPO training pair — stored in SQLite (crash-safe)."""
    # Keep in-memory list for backward compat with training code
    _dpo_pairs.append({
        "prompt": question, "chosen": chosen, "rejected": rejected,
        "ts": datetime.now().isoformat()
    })
    # Also persist to SQLite
    db_add_dpo(question, chosen, rejected, source=source)
    if len(_dpo_pairs) % Config.DPO_BATCH_MIN == 0:
        _improve_q.put("dpo")

def hallucination_check(response, context) -> Tuple[str, float]:
    """Check response faithfulness against provided context."""
    if not context.strip():
        return response, 0.75
    raw = quick(
        f"Context:\n{context[:1000]}\nResponse:\n{response[:400]}\n"
        f"Fraction of claims supported by context 0.0-1.0. Number only.",
        temp=0.1, max_tokens=6
    )
    try:
        score = float(re.search(r'0?\.\d+|1\.0|0|1', raw).group())
    except Exception:
        score = 0.7
    score = min(max(float(score), 0.0), 1.0)
    if score < 0.5:
        _correction_q.put({"question": context[:200], "bad_answer": response, "reason": f"hall_{score:.2f}"})
    return response, score

def auto_correct(question, bad_answer, reason="low_confidence") -> dict:
    """Auto-correct a bad answer using web research."""
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fw = ex.submit(lambda: _t_web_search(question, 3))
            fwiki = ex.submit(lambda: _t_wikipedia(question))
            research = f"{fw.result()[:800]}\n\n{fwiki.result()[:800]}"
    except Exception:
        research = _t_web_search(question, 3)
    good = quick(
        f"Q:{question}\nBad:{bad_answer[:300]}\nResearch:{research[:1500]}\nWrite CORRECT accurate answer.",
        system="Expert fact-checker.", max_tokens=600, temp=0.3
    )
    record_feedback(question, good, bad_answer)
    rag_add(f"Q:{question}\nA:{good}", "self_correction", question)
    result = {
        "question": question, "bad_answer": bad_answer[:200],
        "good_answer": good[:200], "reason": reason,
        "ts": datetime.now().isoformat()
    }
    _correction_log.append(result)
    # Persist to SQLite (in addition to in-memory list)
    db_add_correction(question, bad_answer, good, reason)
    try:
        _atomic_write(CORR_PATH, _correction_log[-200:])
    except Exception:
        pass
    # v13: Auto-add topic to background learner (Grok recommendation)
    try:
        topic_words = question.split()[:5]
        topic = " ".join(topic_words).strip()
        if topic and topic not in _learn_topics:
            _learn_topics.append(topic)
            log.debug(f"Auto-added correction topic: {topic!r}")
    except Exception:
        pass
    return result

def _correction_worker():
    while True:
        try:
            item = _correction_q.get(timeout=10)
            if item:
                auto_correct(item.get("question", ""), item.get("bad_answer", ""), item.get("reason", ""))
        except queue.Empty:
            continue
        except Exception as e:
            log.warning(f"Correction worker: {e}")

def _auto_improve_worker():
    while True:
        try:
            task = _improve_q.get(timeout=5)
            if task == "dpo" and len(_dpo_pairs) >= 10:
                dpo_train()
        except queue.Empty:
            continue
        except Exception as e:
            log.warning(f"Auto-improve worker: {e}")

threading.Thread(target=_correction_worker, daemon=True).start()
threading.Thread(target=_auto_improve_worker, daemon=True).start()
log.info("Hallucination → auto-correction → DPO loop")



# ────────────────────────────────────────────────────────────
# § 14  UNCERTAINTY GATE (streamlined — single LLM call)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 14  UNCERTAINTY GATE (streamlined — single LLM call)
# ════════════════════════════════════════════════════════════════════

def uncertainty_gate(question, context="") -> Tuple[str, float]:
    """Determine confidence level and action with a single LLM call (reduced from 2 in v9)."""
    raw = quick(
        f"Rate your confidence answering this question 0.0-1.0.\n"
        f"Question: {question}\n"
        f"{'Context: ' + context[:400] if context else 'No context available.'}\n"
        f"Return ONLY a number 0.0-1.0.",
        max_tokens=5, temp=0.1
    )
    try:
        conf = float(re.search(r'0?\.\d+|1\.0|0|1', raw).group())
    except Exception:
        conf = 0.6

    conf = round(min(max(conf, 0.0), 1.0), 3)
    threshold = Config.CONF_THRESHOLD
    if conf >= threshold:
        return "answer", conf
    elif conf >= threshold - 0.15:
        return "research", conf
    else:
        return "clarify", conf

log.info("Uncertainty gate (streamlined)")



# ────────────────────────────────────────────────────────────
# § 22  MAIN AGENT LOOP
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 22  MAIN AGENT LOOP
# ════════════════════════════════════════════════════════════════════

# v13: Agent state tracker with request tracing
_agent_state: dict = {
    "last_message":        "",
    "last_ts":             "",
    "last_correlation_id": "",   # v13: unique ID per turn for log correlation
    "current_phase":       AgentPhase.IDLE.value,
    "tool_calls_this_turn":0,
    "rag_hits":            0,
    "confidence":          0.0,
    "total_turns":         0,
    "total_tool_calls":    0,
    "corrections_made":    0,
}

# v13: Tool param injection validation (Kimi + Zai recommendation)
_PARAM_INJECTION_PATTERNS = re.compile(
    r"ignore.{0,5}previous|you.{0,5}are.{0,5}now|act.{0,5}as|system.{0,5}prompt|jailbreak|disregard|override.{0,5}instructions|new.{0,5}instructions",
    re.IGNORECASE
)

def validate_tool_params(tool_name: str, params: dict) -> Tuple[bool, str]:
    """
    Validate tool params against expected schema AND check for injection attacks.
    Returns (ok, error_message).
    Prevents prompt injection via tool parameters (e.g., query="ignore previous instructions...")
    """
    # Schema check
    expected = TOOL_REGISTRY.get(tool_name, {}).get("params", {}) if "TOOL_REGISTRY" in globals() else {}
    for key in params:
        if expected and key not in expected:
            return False, f"Unexpected parameter '{key}' for tool '{tool_name}'"
    # Injection check on string values
    for key, val in params.items():
        if isinstance(val, str) and _PARAM_INJECTION_PATTERNS.search(val):
            audit("tool_param_injection", {"tool": tool_name, "param": key, "val": val[:100]})
            return False, f"Suspicious content in parameter '{key}' — possible injection attempt"
    return True, ""


def _self_reflect(question: str, answer: str, subject: str) -> str:
    """
    v11: Self-reflection loop for complex subjects.
    Scores the answer, critiques it, and rewrites only if score < 8.
    Gated by Config.ENABLE_REFLECTION to save VRAM on low-memory setups.
    """
    if not Config.ENABLE_REFLECTION:
        return answer
    try:
        raw = quick(
            f"Rate this answer 1-10 for correctness and completeness.\n"
            f"Q: {question}\nA: {answer[:400]}\n"
            f"Return: score:<number> issues:<brief critique>",
            max_tokens=80, temp=0.1
        )
        score_match = re.search(r'score:\s*(\d+(?:\.\d+)?)', raw, re.I)
        score = float(score_match.group(1)) if score_match else 7.0
        if score >= 8.0:
            return answer
        # Rewrite with critique guidance
        issues = raw.split("issues:", 1)[-1].strip() if "issues:" in raw.lower() else ""
        improved = quick(
            f"Improve this answer based on the critique.\n"
            f"Q: {question}\nOriginal answer: {answer[:500]}\n"
            f"Issues: {issues}\nImproved answer:",
            system=SUBJECT_PROMPTS.get(subject, ""),
            max_tokens=700, temp=0.4
        )
        return improved if improved.strip() else answer
    except Exception:
        return answer


def _format_sources(chunks: list) -> str:
    if not chunks:
        return ""
    sources = []
    for c in chunks[:5]:
        src = c.get("source", "unknown")
        title = c.get("title", "")[:30]
        label = f"[{src}] {title}" if title else f"[{src}]"
        if label not in sources:
            sources.append(label)
    return "\n\n---\n📚 **Sources:** " + " | ".join(sources)

def agent_stream(message, history, image_path=None, log_fn=None):
    """
    v11 orchestration loop:
    security → cache → compress history → domain → RAG → uncertainty →
    tool chain OR direct generate → reflection → hallucination → citations → save
    """
    # ── v11: Agent state tracker (debug + observability) ──────────
    # v13: Per-turn correlation ID for log tracing
    _correlation_id = str(uuid.uuid4())[:8]
    _agent_state.update({
        "last_message":        message[:80],
        "last_ts":             datetime.now().isoformat(),
        "last_correlation_id": _correlation_id,
        "current_phase":       AgentPhase.SECURITY.value,
        "tool_calls_this_turn":0,
        "rag_hits":            0,
        "confidence":          0.0,
    })
    log.info(f"[{_correlation_id}] Turn start: {message[:60]!r}")

    if not _rate_limiter.allow():
        yield "⚠️ Rate limit reached. Please wait a moment."
        return
    if injection_check(message):
        yield "⚠️ Message flagged. Please rephrase your request."
        return
    message = pii_filter(message)
    audit("user_msg", {"len": len(message), "has_image": bool(image_path)})

    # v11: Compress history if it's grown too long
    if len(history) > Config.COMPRESSION_THRESHOLD:
        history = compress_history(history)

    # Check cache
    ctx_hash = hashlib.md5(str(history[-4:]).encode()).hexdigest() if history else ""
    cached = check_cache(message, ctx_hash)
    if cached:
        yield cached
        return

    # Domain routing + MoE adapter
    domain = classify_domain(message)
    subject = detect_subject(message)
    if Path(DIRS["moe"]).exists() and domain != (_active_adapter or ""):
        adapter_msg = load_moe_adapter(domain)
        if "✅" in adapter_msg:
            if log_fn:
                log_fn(f"  MoE: {adapter_msg}")

    _agent_state['current_phase'] = AgentPhase.RETRIEVAL.value
    # RAG retrieval
    rag_chunks = rag_retrieve(message, k=5)
    rag_ctx = "\n---\n".join(f"[{c.get('source', '')}] {c['text'][:400]}" for c in rag_chunks) if rag_chunks else ""
    source_footnotes = _format_sources(rag_chunks)
    _agent_state["rag_hits"] = len(rag_chunks)

    # Uncertainty gate (only for factual/knowledge queries)
    action, confidence = "answer", 0.8
    qt = _query_type(message)
    if qt in ("factual", "news") and not rag_ctx:
        action, confidence = uncertainty_gate(message, rag_ctx)
        if action == "clarify":
            yield f"🤔 I'm not confident enough to answer this well (confidence: {confidence:.0%}). Could you provide more context or rephrase?"
            return
        if action == "research":
            if log_fn:
                log_fn(f"  🔍 Auto-research (conf: {confidence:.0%})")
            research = _t_web_search(message, 3)
            rag_ctx = f"{rag_ctx}\n\n[AUTO-RESEARCH]\n{research}" if rag_ctx else research
            rag_chunks.extend(rag_retrieve(message, k=3))
            source_footnotes = _format_sources(rag_chunks)

    # Multi-step tool chain for complex requests
    if _is_complex(message):
        if log_fn:
            log_fn("  🔗 Complex query → tool chain")
        result = execute_tool_chain(message, log_fn)
        result = _strip_think(result)
        store_cache(message, result, ctx_hash)
        yield result + source_footnotes
        return

    _agent_state['current_phase'] = AgentPhase.REASONING.value
    # Build messages and stream
    sys_prompt = SYSTEM_PROMPT.replace("{tool_list}", _tool_list_str())
    sys_prompt += f"\n{SUBJECT_PROMPTS.get(subject, '')}"
    msgs = _build_msgs(history, message, sys_prompt, rag_ctx)
    full = ""
    for token in stream_gen(msgs, max_new_tokens=1024, temperature=0.7, image_path=image_path):
        full += token
        yield _strip_think(full)

    # Tool call handling (up to 5 rounds)
    for tool_round in range(5):
        tc = _parse_tool(full)
        if not tc:
            break
        tool_name = tc.get("tool", "")
        params = tc.get("params", {})
        if tool_name not in TOOL_REGISTRY:
            full += f"\n\n⚠️ Unknown tool: {tool_name}"
            yield _strip_think(full)
            break
        # v13: Validate tool params against schema + injection check
        _param_ok, _param_err = validate_tool_params(tool_name, params)
        if not _param_ok:
            full += f"\n\n⚠️ Tool call blocked: {_param_err}"
            yield _strip_think(full)
            audit("tool_blocked", {"tool": tool_name, "reason": _param_err, "cid": _correlation_id})
            break
        if log_fn:
            log_fn(f"  🔧 [{_correlation_id}] {tool_name}({', '.join(f'{k}={str(v)[:30]}' for k, v in params.items())})")
        _agent_state["current_phase"] = AgentPhase.TOOL_EXEC.value
        try:
            result = TOOL_REGISTRY[tool_name]["fn"](**params)
        except Exception as e:
            result = f"Error: {e}"
        result_str = str(result)[:3000]
        # Handle special return types
        if result_str.startswith("__IMAGE__"):
            img_path = result_str.replace("__IMAGE__", "")
            yield _strip_think(full) + f"\n\n![Generated Image]({img_path})"
            break
        if result_str.startswith("__AUDIO__"):
            yield _strip_think(full) + "\n\n🔊 Audio generated."
            break
        # Feed result back to model
        msgs.append({"role": "assistant", "content": full})
        msgs.append({"role": "user", "content": f"[TOOL RESULT: {tool_name}]\n{result_str}\n\nContinue your response using this result."})
        full = ""
        for token in stream_gen(msgs, max_new_tokens=1024, temperature=0.7):
            full += token
            yield _strip_think(full)
        _agent_state["tool_calls_this_turn"] += 1
        _agent_state["total_tool_calls"] += 1

    final = _strip_think(full)

    _agent_state['current_phase'] = AgentPhase.VERIFICATION.value
    # Hallucination check (only when RAG context is present and query is factual)
    if rag_ctx and qt in ("factual", "math"):
        final, hall_score = hallucination_check(final, rag_ctx)
        if hall_score < 0.5:
            _agent_state["corrections_made"] += 1
            if log_fn:
                log_fn(f"  ⚠️ Hallucination score: {hall_score:.2f} — auto-correcting")

    _agent_state['current_phase'] = AgentPhase.REFLECTION.value
    # v11: Self-reflection loop (properly wired, gated by toggle)
    if len(final.split()) > 150 and subject in ("maths", "coding", "science", "humanities"):
        final = _self_reflect(message, final, subject)

    final += source_footnotes
    _agent_state["total_turns"] += 1
    yield final
    store_cache(message, final, ctx_hash)
    save_conversation(history + [{"role": "user", "content": message}, {"role": "assistant", "content": final}])
    rag_add(f"Q:{message}\nA:{final[:800]}", "conversation", message[:50])

log.info("Agent loop ready")


