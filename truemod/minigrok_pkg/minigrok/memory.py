"""
minigrok.memory
Conversation memory, user profile, prioritized memory
"""


# Cross-module imports
import os, json, re, logging, threading, math
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from filelock import FileLock
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

from .base import log, APP_NAME, APP_VERSION
from .config import Config, DIRS, DR
from .models import quick


# ════════════════════════════════════════════════════════════════════
# § 6  PERSISTENT CONVERSATION MEMORY + USER PROFILE
# ════════════════════════════════════════════════════════════════════

CONV_PATH    = Path(f"{DR}/memory/conversations.json")
PROFILE_PATH = Path(f"{DR}/memory/user_profile.json")

_profile: dict = {
    "name": None, "coding_languages": [], "topics": [],
    "session_count": 0, "first_seen": None, "last_seen": None
}
if PROFILE_PATH.exists():
    try:
        _profile.update(json.loads(PROFILE_PATH.read_text()))
    except Exception:
        pass

_profile["session_count"] = _profile.get("session_count", 0) + 1
_profile["last_seen"]     = datetime.now().isoformat()
if not _profile.get("first_seen"):
    _profile["first_seen"] = datetime.now().isoformat()

_saved_convs: list = []
if CONV_PATH.exists():
    try:
        _saved_convs = json.loads(CONV_PATH.read_text())
    except Exception:
        pass

def _save_profile():
    PROFILE_PATH.write_text(json.dumps(_profile, indent=2))

def update_profile(msg: str):
    """Extract user info from messages to build profile."""
    ml = msg.lower()
    nm = re.search(r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)", msg)
    if nm:
        _profile["name"] = nm.group(1)
    for lang in {"python", "javascript", "typescript", "java", "rust", "go", "sql", "c++", "c#", "ruby", "php", "swift", "kotlin"}:
        if lang in ml and lang not in _profile.get("coding_languages", []):
            _profile.setdefault("coding_languages", []).append(lang)
    if len(msg.split()) > 5:
        topics = _profile.setdefault("topics", [])
        if msg[:80] not in topics[-10:]:
            topics.append(msg[:80])
        _profile["topics"] = topics[-50:]
    _save_profile()

def save_conversation(history: list):
    """Persist conversation history to disk with atomic write."""
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        entry = {"role": msg.get("role", "user"), "content": str(content)[:2000], "ts": datetime.now().isoformat()}
        if _saved_convs and _saved_convs[-1].get("content") == entry["content"]:
            continue
        _saved_convs.append(entry)
    while len(_saved_convs) > Config.MAX_SAVED_CONVS * 2:
        _saved_convs.pop(0)
    _atomic_write(CONV_PATH, _saved_convs)

def load_prev_context(n=8) -> list:
    if not _saved_convs:
        return []
    return [{"role": m["role"], "content": m["content"]} for m in _saved_convs[-(n * 2):]]

def compress_history(history: list) -> list:
    """
    v11: Compress old conversation turns into a summary to extend effective context.
    When history exceeds COMPRESSION_THRESHOLD, older turns are summarised into one
    message and only the most recent COMPRESSION_KEEP turns are kept verbatim.
    Improves effective context by ~5x.
    """
    if len(history) <= Config.COMPRESSION_THRESHOLD:
        return history
    # Split: turns to compress vs turns to keep verbatim
    old = history[:len(history) - Config.COMPRESSION_KEEP]
    recent = history[-Config.COMPRESSION_KEEP:]
    # Format old turns for summarisation
    old_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else APP_NAME}: {str(m.get('content', ''))[:400]}"
        for m in old
    )
    summary = quick(
        f"Summarise this conversation so far into 3-5 key points. "
        f"Focus on facts, decisions, and topics discussed.\n\n{old_text}",
        max_tokens=300, temp=0.3
    )
    summary_msg = {
        "role": "user",
        "content": f"[Earlier conversation summary]\n{summary}\n[End of summary — continuing below]"
    }
    return [summary_msg] + recent

def session_primer() -> str:
    """Build a context string about the user for the system prompt."""
    parts = []
    if _profile.get("name"):
        parts.append(f"User: {_profile['name']}.")
    if _profile.get("coding_languages"):
        parts.append(f"Uses: {', '.join(_profile['coding_languages'][:5])}.")
    session_num = _profile.get("session_count", 1)
    if session_num > 1:
        parts.append(f"Session #{session_num}.")
    topics = _profile.get("topics", [])[-3:]
    if topics:
        parts.append(f"Recent topics: {'; '.join(t[:40] for t in topics)}")
    return " ".join(parts)

def export_conversation(history: list) -> str:
    """Export conversation as Markdown file, return path."""
    lines = [f"# {APP_NAME} Conversation Export", f"Date: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for msg in history:
        role = msg.get("role", "user").title()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        lines.append(f"## {role}")
        lines.append(str(content))
        lines.append("")
    fname = f"{DIRS['ws_exports']}/chat_{datetime.now():%Y%m%d_%H%M%S}.md"
    Path(fname).write_text("\n".join(lines))
    return fname

log.info(f"Session memory (#{_profile['session_count']}, {len(_saved_convs)} msgs)")



# ════════════════════════════════════════════════════════════════════
# § 10  PRIORITIZED MEMORY (forgetting curve)
# ════════════════════════════════════════════════════════════════════

PMEM_PATH = Path(f"{DR}/memory/pmem.json")
MEM_PATH  = Path(f"{DR}/memory/kv.json")
_pmem: dict = {}
_memory: dict = {}

for path, store in [(PMEM_PATH, _pmem), (MEM_PATH, _memory)]:
    if path.exists():
        try:
            store.update(json.loads(path.read_text()))
        except Exception:
            pass

def _atomic_write(path: Path, data: dict):
    """
    Atomically write JSON to disk using a lock file + rename.
    Prevents corruption on concurrent writes or mid-write crashes.
    """
    lock_path = str(path) + ".lock"
    tmp_path  = str(path) + f".tmp.{os.getpid()}"
    with FileLock(lock_path, timeout=5):
        try:
            Path(tmp_path).write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, str(path))   # Atomic rename
        except Exception as e:
            log.warning(f"Atomic write failed for {path}: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

def _decay(created: str) -> float:
    """Calculate memory decay based on forgetting curve."""
    try:
        age_days = (datetime.now() - datetime.fromisoformat(created)).total_seconds() / 86400
        return math.exp(-0.693 * age_days / Config.MEMORY_HALF_LIFE)
    except Exception:
        return 0.5

def _effective_score(entry: dict) -> float:
    """Calculate effective importance including decay and access frequency."""
    importance = entry.get("importance", 0.5)
    decay = _decay(entry.get("created", datetime.now().isoformat()))
    access_bonus = min(entry.get("access_count", 0) * 0.05, 0.3)
    return importance * decay + access_bonus

def _score_importance_heuristic(text: str) -> float:
    """Fast heuristic importance scoring (avoids LLM call)."""
    score = 0.5
    # Longer content tends to be more important
    words = len(text.split())
    if words > 100:
        score += 0.1
    if words > 300:
        score += 0.1
    # Technical content
    if any(w in text.lower() for w in ["api", "password", "key", "config", "important", "critical", "remember"]):
        score += 0.2
    # Code
    if re.search(r'def\s+\w+|class\s+\w+|function\s+\w+', text):
        score += 0.1
    return min(score, 1.0)

def mem_set(key: str, val: str, tags=None):
    """Store a key-value pair in prioritized memory."""
    importance = _score_importance_heuristic(val)
    now = datetime.now().isoformat()
    _pmem[key] = {
        "value": val, "importance": round(importance, 3), "tags": tags or [],
        "created": now, "last_accessed": now, "access_count": 0
    }
    _memory[key] = {"value": val, "ts": now}
    _atomic_write(MEM_PATH, _memory)
    _atomic_write(PMEM_PATH, _pmem)
    from .rag import rag_add as _rag_add
    _rag_add(f"{key}: {val}", "memory", key)
    # Garbage collect faded memories
    dead = [k for k, v in _pmem.items() if _effective_score(v) < Config.MEMORY_MIN_IMP]
    for k in dead:
        del _pmem[k]

def mem_get(query: str) -> str:
    """Retrieve from prioritized memory with relevance scoring."""
    scored = []
    ql = query.lower()
    for key, entry in _pmem.items():
        if ql in key.lower() or ql in entry["value"].lower():
            rel = 0.8
        else:
            overlap = len(set(ql.split()) & set(entry["value"].lower().split()[:20]))
            rel = overlap / max(len(ql.split()), 1)
        if rel > 0.1:
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = datetime.now().isoformat()
            scored.append((_effective_score(entry) * 0.4 + rel * 0.6, key, entry))
    if scored:
        scored.sort(reverse=True)
        _atomic_write(PMEM_PATH, _pmem)
        return "\n---\n".join(f"[{s:.0%}] {k}: {e['value']}" for s, k, e in scored[:5])
    from .rag import rag_retrieve as _rag_retrieve
    chunks = _rag_retrieve(query, k=3)
    return "\n---\n".join(c["text"][:400] for c in chunks) if chunks else "Nothing found."

def memory_status() -> str:
    """Return human-readable memory health summary."""
    if not _pmem:
        return "No prioritized memories."
    scores = [_effective_score(e) for e in _pmem.values()]
    return (f"Total: {len(_pmem)}\n  🟢 High: {sum(1 for s in scores if s >= 0.7)}\n"
            f"  🟡 Med:  {sum(1 for s in scores if 0.3 <= s < 0.7)}\n"
            f"  🔴 Fading: {sum(1 for s in scores if s < 0.3)}")

log.info(f"Prioritized memory ({len(_pmem)} entries)")


