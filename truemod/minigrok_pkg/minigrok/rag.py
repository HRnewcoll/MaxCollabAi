"""
minigrok.rag
RAG chunker, ChromaDB, graph RAG, MoE, response cache
"""


# Cross-module imports
import os, json, re, logging, threading, hashlib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from peft import PeftModel

from .base import log
from .config import Config, DIRS
from .utils import pii_filter, injection_check
from .models import quick_routed as quick, _model_lock


# ════════════════════════════════════════════════════════════════════
# § 8  CODE-AWARE RAG CHUNKER
# ════════════════════════════════════════════════════════════════════

def _chunk_python(source: str, max_lines: int = 80) -> List[str]:
    """Chunk Python source by AST nodes (functions, classes)."""
    chunks = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        lines = source.split("\n")
        return ["\n".join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)
                if "\n".join(lines[i:i+max_lines]).strip()]
    lines = source.split("\n")
    imports = ["\n".join(lines[n.lineno-1:(n.end_lineno if hasattr(n, "end_lineno") else n.lineno)])
               for n in _ast.iter_child_nodes(tree)
               if isinstance(n, (_ast.Import, _ast.ImportFrom))]
    header = "\n".join(imports)
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            start = node.lineno - 1
            if hasattr(node, "decorator_list") and node.decorator_list:
                start = min(d.lineno - 1 for d in node.decorator_list)
            end = node.end_lineno if hasattr(node, "end_lineno") else start + 1
            chunks.append(f"{header}\n\n{chr(10).join(lines[start:end])}".strip())
    return [c for c in chunks if c.strip()] or [source]

def _chunk_js(source: str) -> List[str]:
    """Chunk JavaScript/TypeScript source by function/class boundaries."""
    lines = source.split("\n")
    boundaries = [0]
    pat = r'^\s{0,2}((?:export\s+)?(?:async\s+)?function|(?:export\s+)?class|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()'
    for i, line in enumerate(lines):
        if re.match(pat, line) and i > 0:
            boundaries.append(i)
    boundaries.append(len(lines))
    return ["\n".join(lines[boundaries[i]:boundaries[i+1]]).strip()
            for i in range(len(boundaries) - 1)
            if "\n".join(lines[boundaries[i]:boundaries[i+1]]).strip()] or [source]

def _looks_like_code(text: str) -> bool:
    sigs = [r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'function\s+\w+',
            r'const\s+\w+\s*=', r'console\.log', r'print\(']
    return sum(1 for p in sigs if re.search(p, text[:2000])) >= 3

def smart_chunk(text: str, filename: str = "") -> List[str]:
    """Intelligently chunk text based on file type (Python AST, JS regex, or sentence-aware)."""
    ext = Path(filename).suffix.lower()
    if ext == ".py" or (not ext and _looks_like_code(text) and "def " in text):
        return _chunk_python(text)
    if ext in (".js", ".ts", ".jsx", ".tsx") or (not ext and _looks_like_code(text)):
        return _chunk_js(text)
    # Prose: sentence-aware chunking
    try:
        import nltk
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, word_count = [], [], 0
    for sent in sents:
        words = sent.split()
        if word_count + len(words) > 400 and current:
            chunks.append(" ".join(current))
            current, word_count = [], 0
        current.extend(words)
        word_count += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]

log.info("Code-aware chunker ready")



# ════════════════════════════════════════════════════════════════════
# § 9  CHROMADB RAG + GRAPH RAG + MoE ROUTING
# ════════════════════════════════════════════════════════════════════

_chroma = chromadb.PersistentClient(path=DIRS["chroma"])
_col    = _chroma.get_or_create_collection("minigrok_rag", metadata={"hnsw:space": "cosine"})

# v11: toggleable embedding model — bge-m3 is higher quality, MiniLM is faster
_EMBEDDER_NAME = "BAAI/bge-m3" if Config.USE_LARGE_EMBEDDER else "all-MiniLM-L6-v2"
embedder = SentenceTransformer(_EMBEDDER_NAME)
log.info(f"Embedder: {_EMBEDDER_NAME}")

try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_OK = True
except Exception:
    reranker = None
    RERANK_OK = False

# Knowledge Graph
KG = nx.DiGraph()
KG_PATH = Path(f"{DR}/memory/kg.json")
if KG_PATH.exists():
    try:
        KG = nx.node_link_graph(json.loads(KG_PATH.read_text()))
    except Exception:
        pass

def _save_kg():
    """Save knowledge graph to disk."""
    try:
        KG_PATH.write_text(json.dumps(nx.node_link_data(KG)))
    except Exception as e:
        log.warning(f"KG save failed: {e}")

# ── Hybrid BM25 index (v11) ──────────────────────────────────────
# Maintains an in-memory BM25 index alongside ChromaDB for hybrid search.
# Hybrid = 60% semantic + 40% BM25 via reciprocal rank fusion.
_bm25_docs: List[str] = []
_bm25_ids:  List[str] = []
_bm25_index: Optional[BM25Okapi] = None

def _rebuild_bm25():
    """Rebuild BM25 index from current docs. Call after bulk inserts."""
    global _bm25_index
    if _bm25_docs:
        tokenized = [d.lower().split() for d in _bm25_docs]
        _bm25_index = BM25Okapi(tokenized)

def _hybrid_fuse(sem_results: list, bm25_results: list, k: int, alpha: float = 0.6) -> list:
    """
    Reciprocal Rank Fusion: combine semantic and BM25 rankings.
    alpha=1.0 = pure semantic, alpha=0.0 = pure BM25.
    """
    def rrf_score(rank: int, c: int = 60) -> float:
        return 1.0 / (c + rank)

    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}

    for rank, doc in enumerate(sem_results):
        h = doc.get("id", "")
        scores[h] = scores.get(h, 0) + alpha * rrf_score(rank)
        doc_map[h] = doc

    for rank, doc in enumerate(bm25_results):
        h = doc.get("id", "")
        scores[h] = scores.get(h, 0) + (1 - alpha) * rrf_score(rank)
        if h not in doc_map:
            doc_map[h] = doc

    fused = sorted(scores.keys(), key=lambda h: -scores[h])
    return [doc_map[h] for h in fused[:k]]

# MoE domain embeddings
_MOE_DESCS = {
    "coder": "programming code debugging python javascript algorithms functions software engineering",
    "scientist": "research science physics chemistry mathematics equations theorems proofs",
    "writer": "writing essay story article summarise draft creative narrative blog",
    "analyst": "analyse data chart statistics predict business strategy report finance"
}
_domain_names = list(_MOE_DESCS.keys())
_domain_embs  = embedder.encode(list(_MOE_DESCS.values()), normalize_embeddings=True).astype("float32")
_active_adapter = None

def classify_domain(query: str) -> str:
    """Classify query into MoE domain using embedding similarity."""
    qe = embedder.encode([query], normalize_embeddings=True).astype("float32")
    sims = (_domain_embs @ qe.T).squeeze()
    best = int(np.argmax(sims))
    return _domain_names[best] if sims[best] > 0.3 else "writer"

def load_moe_adapter(domain: str) -> str:
    """Load a domain-specific LoRA adapter if available."""
    global _active_adapter
    adapter_path = Path(DIRS["moe"]) / domain
    if not adapter_path.exists() or not any(adapter_path.rglob("*.safetensors")):
        return f"No adapter for '{domain}'"
    if _active_adapter == domain:
        return f"Adapter '{domain}' already active"
    try:
        with _model_lock:
            global model
            if hasattr(model, "disable_adapter_layers"):
                model.disable_adapter_layers()
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
        _active_adapter = domain
        return f"✅ Loaded '{domain}' adapter"
    except Exception as e:
        return f"⚠️ Adapter load failed: {e}"

_seen: set = set()

def rag_add(text, source="manual", title="", url="", filename="") -> int:
    """Add text to RAG knowledge base with dedup and KG extraction."""
    if injection_check(text):
        return 0
    text = pii_filter(text)
    added = 0
    now = datetime.now().isoformat()
    chunks = smart_chunk(text, filename=filename)
    for chunk in chunks:
        chunk_hash = hashlib.md5(chunk[:300].encode()).hexdigest()
        if chunk_hash in _seen:
            continue
        emb = embedder.encode([chunk]).astype("float32").tolist()[0]
        try:
            res = _col.query(query_embeddings=[emb], n_results=1, include=["distances"])
            if res["distances"] and res["distances"][0] and 1 - res["distances"][0][0] >= Config.DEDUP_THRESHOLD:
                continue
        except Exception:
            pass
        _seen.add(chunk_hash)
        _col.add(
            documents=[chunk], embeddings=[emb], ids=[chunk_hash],
            metadatas=[{"source": source, "title": title, "url": url, "ts": now, "filename": filename}]
        )
        # v11: also add to BM25 index
        _bm25_docs.append(chunk)
        _bm25_ids.append(chunk_hash)
        added += 1
        # Extract entities for KG
        ents = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', chunk)))[:10]
        for e in ents:
            KG.add_node(e, source=source)
        for i, a in enumerate(ents):
            for b in ents[i+1:i+3]:
                if a != b:
                    w = KG[a][b].get("weight", 0) + 1 if KG.has_edge(a, b) else 1
                    KG.add_edge(a, b, weight=w)
    if added:
        _save_kg()
        _rebuild_bm25()
    return added

def rag_retrieve(query, k=None) -> list:
    """Retrieve relevant chunks from RAG with optional reranking and KG enrichment."""
    count = _col.count()
    if count == 0:
        return []
    if k is None:
        rich_keywords = {"compare", "analyse", "analyze", "explain", "research", "summarise", "overview", "report"}
        k = 8 if len(query.split()) > 12 or set(query.lower().split()) & rich_keywords else 3

    queries = [query]
    # Only generate expansion query for complex queries (saves LLM call for simple ones)
    if len(query.split()) > 8:
        try:
            queries.append(quick(f"Short passage answering: {query}", max_tokens=80, temp=0.4))
        except Exception:
            pass

    all_docs = {}
    for q in queries[:3]:
        emb = embedder.encode([q]).astype("float32").tolist()[0]
        try:
            res = _col.query(
                query_embeddings=[emb], n_results=min(k * 3, count),
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                doc_hash = hashlib.md5(doc[:200].encode()).hexdigest()
                if doc_hash not in all_docs:
                    all_docs[doc_hash] = {
                        "text": doc, "source": meta.get("source", ""),
                        "title": meta.get("title", ""), "score": 1 - dist, "id": doc_hash
                    }
        except Exception:
            continue

    sem_candidates = list(all_docs.values())

    # v11: Hybrid BM25 + semantic via Reciprocal Rank Fusion
    bm25_candidates = []
    if _bm25_index is not None and _bm25_docs:
        try:
            bm25_scores = _bm25_index.get_scores(query.lower().split())
            top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:k * 2]
            for idx in top_bm25_idx:
                if idx < len(_bm25_docs) and bm25_scores[idx] > 0:
                    doc_hash = _bm25_ids[idx]
                    bm25_candidates.append({
                        "text": _bm25_docs[idx], "source": "bm25", "title": "",
                        "score": float(bm25_scores[idx]), "id": doc_hash
                    })
        except Exception:
            pass

    if bm25_candidates:
        candidates = _hybrid_fuse(sem_candidates, bm25_candidates, k=k * 2, alpha=0.6)
    else:
        candidates = sem_candidates

    if RERANK_OK and reranker and len(candidates) > k:
        scores = reranker.predict([(query, c["text"][:512]) for c in candidates])
        candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    result = candidates[:k]

    # Graph enrichment
    if KG.number_of_nodes() > 0:
        ents = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        related = set()
        for e in ents[:3]:
            if KG.has_node(e):
                related.update(list(KG.neighbors(e))[:4])
        if related:
            result.append({"text": f"Related: {', '.join(related)}", "source": "KG", "title": "KG", "score": 0.5, "id": "kg"})
    return result

log.info("RAG + Graph RAG + MoE routing ready")



# ════════════════════════════════════════════════════════════════════
# § 15  RESPONSE CACHE (query-type TTL)
# ════════════════════════════════════════════════════════════════════

def _query_type(query: str) -> str:
    """Classify query type for cache TTL decisions."""
    ql = query.lower()
    if any(w in ql for w in ["news", "latest", "today", "current"]):
        return "news"
    if any(w in ql for w in ["write", "create", "poem", "story", "imagine"]):
        return "creative"
    if any(w in ql for w in ["my", "i am", "i'm", "personal"]):
        return "personal"
    if any(w in ql for w in ["code", "function", "program", "debug"]):
        return "code"
    if any(w in ql for w in ["calculate", "solve", "equation", "integral"]):
        return "math"
    return "factual"

def check_cache(prompt: str, context_hash: str = "") -> Optional[str]:
    """Check response cache with context-aware keying."""
    ttl = Config.CACHE_TTLS.get(_query_type(prompt), 600)
    if ttl == 0:
        return None
    key = hashlib.md5(f"{prompt}:{context_hash}".encode()).hexdigest()
    return _response_cache.get(key)

def store_cache(prompt: str, response: str, context_hash: str = ""):
    """Store response in cache with appropriate TTL."""
    ttl = Config.CACHE_TTLS.get(_query_type(prompt), 600)
    if ttl > 0:
        key = hashlib.md5(f"{prompt}:{context_hash}".encode()).hexdigest()
        _response_cache.set(key, response)

log.info("Response cache ready")


"""
minigrok.rag
RAG chunker, ChromaDB, graph RAG, MoE, response cache
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
from .generation import quick
from .models import model, _model_lock

# ────────────────────────────────────────────────────────────
# § 8  CODE-AWARE RAG CHUNKER
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 8  CODE-AWARE RAG CHUNKER
# ════════════════════════════════════════════════════════════════════

def _chunk_python(source: str, max_lines: int = 80) -> List[str]:
    """Chunk Python source by AST nodes (functions, classes)."""
    chunks = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        lines = source.split("\n")
        return ["\n".join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)
                if "\n".join(lines[i:i+max_lines]).strip()]
    lines = source.split("\n")
    imports = ["\n".join(lines[n.lineno-1:(n.end_lineno if hasattr(n, "end_lineno") else n.lineno)])
               for n in _ast.iter_child_nodes(tree)
               if isinstance(n, (_ast.Import, _ast.ImportFrom))]
    header = "\n".join(imports)
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            start = node.lineno - 1
            if hasattr(node, "decorator_list") and node.decorator_list:
                start = min(d.lineno - 1 for d in node.decorator_list)
            end = node.end_lineno if hasattr(node, "end_lineno") else start + 1
            chunks.append(f"{header}\n\n{chr(10).join(lines[start:end])}".strip())
    return [c for c in chunks if c.strip()] or [source]

def _chunk_js(source: str) -> List[str]:
    """Chunk JavaScript/TypeScript source by function/class boundaries."""
    lines = source.split("\n")
    boundaries = [0]
    pat = r'^\s{0,2}((?:export\s+)?(?:async\s+)?function|(?:export\s+)?class|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()'
    for i, line in enumerate(lines):
        if re.match(pat, line) and i > 0:
            boundaries.append(i)
    boundaries.append(len(lines))
    return ["\n".join(lines[boundaries[i]:boundaries[i+1]]).strip()
            for i in range(len(boundaries) - 1)
            if "\n".join(lines[boundaries[i]:boundaries[i+1]]).strip()] or [source]

def _looks_like_code(text: str) -> bool:
    sigs = [r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'function\s+\w+',
            r'const\s+\w+\s*=', r'console\.log', r'print\(']
    return sum(1 for p in sigs if re.search(p, text[:2000])) >= 3

def smart_chunk(text: str, filename: str = "") -> List[str]:
    """Intelligently chunk text based on file type (Python AST, JS regex, or sentence-aware)."""
    ext = Path(filename).suffix.lower()
    if ext == ".py" or (not ext and _looks_like_code(text) and "def " in text):
        return _chunk_python(text)
    if ext in (".js", ".ts", ".jsx", ".tsx") or (not ext and _looks_like_code(text)):
        return _chunk_js(text)
    # Prose: sentence-aware chunking
    try:
        import nltk
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, word_count = [], [], 0
    for sent in sents:
        words = sent.split()
        if word_count + len(words) > 400 and current:
            chunks.append(" ".join(current))
            current, word_count = [], 0
        current.extend(words)
        word_count += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]

log.info("Code-aware chunker ready")



# ────────────────────────────────────────────────────────────
# § 9  CHROMADB RAG + GRAPH RAG + MoE ROUTING
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 9  CHROMADB RAG + GRAPH RAG + MoE ROUTING
# ════════════════════════════════════════════════════════════════════

_chroma = chromadb.PersistentClient(path=DIRS["chroma"])
_col    = _chroma.get_or_create_collection("minigrok_rag", metadata={"hnsw:space": "cosine"})

# v11: toggleable embedding model — bge-m3 is higher quality, MiniLM is faster
_EMBEDDER_NAME = "BAAI/bge-m3" if Config.USE_LARGE_EMBEDDER else "all-MiniLM-L6-v2"
embedder = SentenceTransformer(_EMBEDDER_NAME)
log.info(f"Embedder: {_EMBEDDER_NAME}")

try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_OK = True
except Exception:
    reranker = None
    RERANK_OK = False

# Knowledge Graph
KG = nx.DiGraph()
KG_PATH = Path(f"{DR}/memory/kg.json")
if KG_PATH.exists():
    try:
        KG = nx.node_link_graph(json.loads(KG_PATH.read_text()))
    except Exception:
        pass

def _save_kg():
    """Save knowledge graph to disk."""
    try:
        KG_PATH.write_text(json.dumps(nx.node_link_data(KG)))
    except Exception as e:
        log.warning(f"KG save failed: {e}")

# ── Hybrid BM25 index (v11) ──────────────────────────────────────
# Maintains an in-memory BM25 index alongside ChromaDB for hybrid search.
# Hybrid = 60% semantic + 40% BM25 via reciprocal rank fusion.
_bm25_docs: List[str] = []
_bm25_ids:  List[str] = []
_bm25_index: Optional[BM25Okapi] = None

def _rebuild_bm25():
    """Rebuild BM25 index from current docs. Call after bulk inserts."""
    global _bm25_index
    if _bm25_docs:
        tokenized = [d.lower().split() for d in _bm25_docs]
        _bm25_index = BM25Okapi(tokenized)

def _hybrid_fuse(sem_results: list, bm25_results: list, k: int, alpha: float = 0.6) -> list:
    """
    Reciprocal Rank Fusion: combine semantic and BM25 rankings.
    alpha=1.0 = pure semantic, alpha=0.0 = pure BM25.
    """
    def rrf_score(rank: int, c: int = 60) -> float:
        return 1.0 / (c + rank)

    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}

    for rank, doc in enumerate(sem_results):
        h = doc.get("id", "")
        scores[h] = scores.get(h, 0) + alpha * rrf_score(rank)
        doc_map[h] = doc

    for rank, doc in enumerate(bm25_results):
        h = doc.get("id", "")
        scores[h] = scores.get(h, 0) + (1 - alpha) * rrf_score(rank)
        if h not in doc_map:
            doc_map[h] = doc

    fused = sorted(scores.keys(), key=lambda h: -scores[h])
    return [doc_map[h] for h in fused[:k]]

# MoE domain embeddings
_MOE_DESCS = {
    "coder": "programming code debugging python javascript algorithms functions software engineering",
    "scientist": "research science physics chemistry mathematics equations theorems proofs",
    "writer": "writing essay story article summarise draft creative narrative blog",
    "analyst": "analyse data chart statistics predict business strategy report finance"
}
_domain_names = list(_MOE_DESCS.keys())
_domain_embs  = embedder.encode(list(_MOE_DESCS.values()), normalize_embeddings=True).astype("float32")
_active_adapter = None

def classify_domain(query: str) -> str:
    """Classify query into MoE domain using embedding similarity."""
    qe = embedder.encode([query], normalize_embeddings=True).astype("float32")
    sims = (_domain_embs @ qe.T).squeeze()
    best = int(np.argmax(sims))
    return _domain_names[best] if sims[best] > 0.3 else "writer"

def load_moe_adapter(domain: str) -> str:
    """Load a domain-specific LoRA adapter if available."""
    global _active_adapter
    adapter_path = Path(DIRS["moe"]) / domain
    if not adapter_path.exists() or not any(adapter_path.rglob("*.safetensors")):
        return f"No adapter for '{domain}'"
    if _active_adapter == domain:
        return f"Adapter '{domain}' already active"
    try:
        with _model_lock:
            global model
            if hasattr(model, "disable_adapter_layers"):
                model.disable_adapter_layers()
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
        _active_adapter = domain
        return f"✅ Loaded '{domain}' adapter"
    except Exception as e:
        return f"⚠️ Adapter load failed: {e}"

_seen: set = set()

def rag_add(text, source="manual", title="", url="", filename="") -> int:
    """Add text to RAG knowledge base with dedup and KG extraction."""
    if injection_check(text):
        return 0
    text = pii_filter(text)
    added = 0
    now = datetime.now().isoformat()
    chunks = smart_chunk(text, filename=filename)
    for chunk in chunks:
        chunk_hash = hashlib.md5(chunk[:300].encode()).hexdigest()
        if chunk_hash in _seen:
            continue
        emb = embedder.encode([chunk]).astype("float32").tolist()[0]
        try:
            res = _col.query(query_embeddings=[emb], n_results=1, include=["distances"])
            if res["distances"] and res["distances"][0] and 1 - res["distances"][0][0] >= Config.DEDUP_THRESHOLD:
                continue
        except Exception:
            pass
        _seen.add(chunk_hash)
        _col.add(
            documents=[chunk], embeddings=[emb], ids=[chunk_hash],
            metadatas=[{"source": source, "title": title, "url": url, "ts": now, "filename": filename}]
        )
        # v11: also add to BM25 index
        _bm25_docs.append(chunk)
        _bm25_ids.append(chunk_hash)
        added += 1
        # Extract entities for KG
        ents = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', chunk)))[:10]
        for e in ents:
            KG.add_node(e, source=source)
        for i, a in enumerate(ents):
            for b in ents[i+1:i+3]:
                if a != b:
                    w = KG[a][b].get("weight", 0) + 1 if KG.has_edge(a, b) else 1
                    KG.add_edge(a, b, weight=w)
    if added:
        _save_kg()
        _rebuild_bm25()
    return added

def rag_retrieve(query, k=None) -> list:
    """Retrieve relevant chunks from RAG with optional reranking and KG enrichment."""
    count = _col.count()
    if count == 0:
        return []
    if k is None:
        rich_keywords = {"compare", "analyse", "analyze", "explain", "research", "summarise", "overview", "report"}
        k = 8 if len(query.split()) > 12 or set(query.lower().split()) & rich_keywords else 3

    queries = [query]
    # Only generate expansion query for complex queries (saves LLM call for simple ones)
    if len(query.split()) > 8:
        try:
            queries.append(quick(f"Short passage answering: {query}", max_tokens=80, temp=0.4))
        except Exception:
            pass

    all_docs = {}
    for q in queries[:3]:
        emb = embedder.encode([q]).astype("float32").tolist()[0]
        try:
            res = _col.query(
                query_embeddings=[emb], n_results=min(k * 3, count),
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                doc_hash = hashlib.md5(doc[:200].encode()).hexdigest()
                if doc_hash not in all_docs:
                    all_docs[doc_hash] = {
                        "text": doc, "source": meta.get("source", ""),
                        "title": meta.get("title", ""), "score": 1 - dist, "id": doc_hash
                    }
        except Exception:
            continue

    sem_candidates = list(all_docs.values())

    # v11: Hybrid BM25 + semantic via Reciprocal Rank Fusion
    bm25_candidates = []
    if _bm25_index is not None and _bm25_docs:
        try:
            bm25_scores = _bm25_index.get_scores(query.lower().split())
            top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:k * 2]
            for idx in top_bm25_idx:
                if idx < len(_bm25_docs) and bm25_scores[idx] > 0:
                    doc_hash = _bm25_ids[idx]
                    bm25_candidates.append({
                        "text": _bm25_docs[idx], "source": "bm25", "title": "",
                        "score": float(bm25_scores[idx]), "id": doc_hash
                    })
        except Exception:
            pass

    if bm25_candidates:
        candidates = _hybrid_fuse(sem_candidates, bm25_candidates, k=k * 2, alpha=0.6)
    else:
        candidates = sem_candidates

    if RERANK_OK and reranker and len(candidates) > k:
        scores = reranker.predict([(query, c["text"][:512]) for c in candidates])
        candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    result = candidates[:k]

    # Graph enrichment
    if KG.number_of_nodes() > 0:
        ents = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        related = set()
        for e in ents[:3]:
            if KG.has_node(e):
                related.update(list(KG.neighbors(e))[:4])
        if related:
            result.append({"text": f"Related: {', '.join(related)}", "source": "KG", "title": "KG", "score": 0.5, "id": "kg"})
    return result

log.info("RAG + Graph RAG + MoE routing ready")



# ────────────────────────────────────────────────────────────
# § 15  RESPONSE CACHE (query-type TTL)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 15  RESPONSE CACHE (query-type TTL)
# ════════════════════════════════════════════════════════════════════

def _query_type(query: str) -> str:
    """Classify query type for cache TTL decisions."""
    ql = query.lower()
    if any(w in ql for w in ["news", "latest", "today", "current"]):
        return "news"
    if any(w in ql for w in ["write", "create", "poem", "story", "imagine"]):
        return "creative"
    if any(w in ql for w in ["my", "i am", "i'm", "personal"]):
        return "personal"
    if any(w in ql for w in ["code", "function", "program", "debug"]):
        return "code"
    if any(w in ql for w in ["calculate", "solve", "equation", "integral"]):
        return "math"
    return "factual"

def check_cache(prompt: str, context_hash: str = "") -> Optional[str]:
    """Check response cache with context-aware keying."""
    ttl = Config.CACHE_TTLS.get(_query_type(prompt), 600)
    if ttl == 0:
        return None
    key = hashlib.md5(f"{prompt}:{context_hash}".encode()).hexdigest()
    return _response_cache.get(key)

def store_cache(prompt: str, response: str, context_hash: str = ""):
    """Store response in cache with appropriate TTL."""
    ttl = Config.CACHE_TTLS.get(_query_type(prompt), 600)
    if ttl > 0:
        key = hashlib.md5(f"{prompt}:{context_hash}".encode()).hexdigest()
        _response_cache.set(key, response)

log.info("Response cache ready")


