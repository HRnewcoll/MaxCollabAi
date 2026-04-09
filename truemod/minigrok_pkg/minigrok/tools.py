"""
minigrok.tools
Tool validator, tool implementations
"""


# Cross-module imports
import os, json, re, logging, threading, hashlib, tempfile, time
import shutil, queue, math, io, base64, uuid, traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
from PIL import Image
import PyPDF2, pandas as pd
import wikipedia as wikipedia_lib, arxiv as arxiv_lib
from duckduckgo_search import DDGS
import trafilatura, feedparser
from jsonschema import validate, ValidationError
from RestrictedPython import compile_restricted, safe_globals, safe_builtins

from .base import log
from .config import Config, DIRS
from .utils import with_retry, _breakers, _tool_cache, pii_filter, injection_check
from .models import quick, quick_routed, VRAMJuggler
from .generation import SUBJECT_PROMPTS, detect_subject
from .rag import rag_add, rag_retrieve
from .memory import mem_set, mem_get, memory_status, ws_read, ws_write, ws_list, ws_note
from .voice import synthesise, _voice_profiles


# ════════════════════════════════════════════════════════════════════
# § 12  STRUCTURED TOOL VALIDATOR (JSON schema + auto-retry)
# ════════════════════════════════════════════════════════════════════

TOOL_CALL_SCHEMA = {
    "type": "object", "required": ["tool", "params"],
    "properties": {"tool": {"type": "string"}, "params": {"type": "object"}},
    "additionalProperties": False
}

def _validate_tool_call(raw: dict) -> Tuple[bool, str]:
    try:
        validate(instance=raw, schema=TOOL_CALL_SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, str(e.message)

def _parse_tool(text: str, max_retries: int = 3) -> Optional[dict]:
    """Parse and validate a tool call from model output with auto-retry on malformed JSON."""
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not match:
        return None
    raw_str = match.group(1)
    for attempt in range(max_retries):
        try:
            data = json.loads(raw_str)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                fix = quick(f"Fix JSON syntax error:\n{raw_str}\nError:{e}\nReturn ONLY valid JSON with 'tool' and 'params'.",
                           max_tokens=200, temp=0.1)
                fix = re.sub(r'```json?\s*|\s*```', '', fix).strip()
                jm = re.search(r'\{.*\}', fix, re.DOTALL)
                if jm:
                    raw_str = jm.group()
            continue
        ok, err = _validate_tool_call(data)
        if ok:
            return data
        if attempt < max_retries - 1:
            fix = quick(f'Fix tool call JSON:\n{json.dumps(data)}\nError:{err}\nReturn ONLY: {{"tool":"name","params":{{}}}}',
                       max_tokens=200, temp=0.1)
            fix = re.sub(r'```json?\s*|\s*```', '', fix).strip()
            jm = re.search(r'\{.*\}', fix, re.DOTALL)
            if jm:
                raw_str = jm.group()
        else:
            audit("tool_validation_failed", {"raw": raw_str[:200], "error": err})
            return None
    return None

log.info("Structured tool validator ready")



# ════════════════════════════════════════════════════════════════════
# § 16  TOOL IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════

# ── Browser ─────────────────────────────────────────────────────
_browser = None
_page = None

def _get_page():
    global _browser, _page
    if not Config.ENABLE_BROWSER:
        raise RuntimeError("Browser is disabled. Set Config.ENABLE_BROWSER=True.")
    from playwright.sync_api import sync_playwright
    if _browser is None:
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=True, args=["--no-sandbox"])
        _page = _browser.new_page(viewport={"width": 1280, "height": 800})
    return _page

def _screenshot(label="") -> str:
    path = os.path.join(DIRS["outputs"], f"ss_{int(time.time()*1000)}.png")
    _get_page().screenshot(path=path, full_page=False)
    return path

def _vl_decide(goal, ss_path, step, hist) -> dict:
    raw = quick(
        f"Goal:{goal}\nStep {step}. Prev:{hist}\nDecide next action.\n"
        f'JSON only: {{"reasoning":"...","action":"navigate|click|type|scroll|extract|search|done",'
        f'"params":{{"url":"","selector":"","text":""}},"done":false}}',
        image_path=ss_path, max_tokens=250, temp=0.2
    )
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {"action": "scroll", "params": {}, "done": False}
    except Exception:
        return {"action": "scroll", "params": {}, "done": False}

def _verify_goal(goal, ss_path, page_content, steps_text) -> Tuple[bool, str]:
    raw = quick(
        f"GOAL:{goal}\nSTEPS:{steps_text}\nPAGE:{page_content[:600]}\n"
        f'Was goal FULLY achieved? JSON: {{"complete":true/false,"confidence":0.0-1.0,"explanation":"","missing":""}}',
        image_path=ss_path, max_tokens=150, temp=0.1
    )
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        d = json.loads(m.group()) if m else {}
        if d.get("complete") and float(d.get("confidence", 0)) >= 0.7:
            return True, d.get("explanation", "")
        return False, d.get("missing", "Incomplete")
    except Exception:
        return False, "Could not verify"

def browser_agent(goal, max_steps=20, start_url="", save_to_workspace=True, log_fn=None) -> dict:
    """Autonomous browser agent: navigates, acts, and verifies goal completion."""
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n🤖 Browser Agent: {goal}")
    page = _get_page()
    steps_done = []
    screenshots = []
    if start_url:
        try:
            page.goto(start_url, wait_until="networkidle", timeout=20000)
            _log(f"  ▶ {start_url}")
        except Exception:
            pass
    for step in range(1, max_steps + 1):
        ss_path = _screenshot(f"step_{step}")
        screenshots.append(ss_path)
        hist = "\n".join(f"  {s['step']}: {s['action']} → {s['result'][:50]}" for s in steps_done[-5:])
        d = _vl_decide(goal, ss_path, step, hist)
        action = d.get("action", "")
        params = d.get("params", {})
        done = d.get("done", False)
        _log(f"  Step {step}: {action} — {d.get('reasoning', '')[:70]}")
        sr = {"step": step, "action": action, "target": "", "result": ""}
        try:
            if action == "navigate" and params.get("url"):
                page.goto(params["url"], wait_until="networkidle", timeout=20000)
                ct = trafilatura.extract(page.content()) or ""
                rag_add(ct[:4000], "browser", params["url"])
                sr["result"] = f"Navigated to {params['url']}"
            elif action == "click":
                sel = params.get("selector", "")
                txt = params.get("text", "")
                if sel:
                    page.click(sel, timeout=5000)
                elif txt:
                    page.get_by_text(txt).first.click(timeout=5000)
                sr["result"] = "Clicked"
            elif action == "type":
                if params.get("selector"):
                    page.fill(params["selector"], params.get("text", ""))
                else:
                    page.keyboard.type(params.get("text", ""))
                sr["result"] = "Typed"
            elif action == "scroll":
                page.evaluate("window.scrollBy(0,600)")
                sr["result"] = "Scrolled"
            elif action == "extract":
                ct = trafilatura.extract(page.content()) or page.inner_text("body")
                rag_add(ct[:5000], "browser_extract", page.url)
                if save_to_workspace:
                    Path(f"{DIRS['ws_downloads']}/page_{int(time.time())}.txt").write_text(ct[:10000])
                sr["result"] = f"Extracted {len(ct.split())} words"
            elif action == "search" and params.get("query"):
                page.goto(f"https://www.google.com/search?q={params['query'].replace(' ', '+')}")
                sr["result"] = "Searched"
            elif action == "done" or done:
                sr["result"] = "Goal complete"
                steps_done.append(sr)
                break
            time.sleep(0.8)
        except Exception as e:
            sr["result"] = f"Error: {e}"
            _log(f"    ⚠️  {e}")
        steps_done.append(sr)
        if done:
            break

    try:
        final = trafilatura.extract(page.content()) or ""
    except Exception:
        final = ""

    # Verify
    verified = False
    if steps_done:
        ss = _screenshot("verify")
        st = "\n".join(f"{s['step']}: {s['action']} → {s['result']}" for s in steps_done)
        verified, exp = _verify_goal(goal, ss, final, st)
        _log(f"  {'✅' if verified else '❌'} {exp[:80]}")
    _steps_text = "\n".join(
        f"{s['step']}: {s['action']} → {s['result']}" for s in steps_done[:10]
    )
    summary = quick(
        f"Goal:{goal}\nSteps:\n{_steps_text}\nPage:{final[:800]}\nSummarise result.",
        max_tokens=500, temp=0.4
    )
    if save_to_workspace:
        with open(WS_NOTES, "a") as f:
            f.write(f"\n## Browser: {goal}\n{datetime.now():%Y-%m-%d %H:%M}\nSteps:{len(steps_done)}\n{summary[:300]}\n")
    return {"result": summary, "steps": steps_done, "steps_taken": len(steps_done),
            "screenshots": screenshots, "verified": verified}

def browser_action_single(action, url="", selector="", text_input="", js_code="") -> str:
    """Execute a single browser action (manual mode)."""
    page = _get_page()
    try:
        if action == "navigate":
            page.goto(url, wait_until="networkidle", timeout=25000)
            ct = trafilatura.extract(page.content()) or ""
            rag_add(ct[:4000], "browser", url)
            return f"Navigated.\n\n{ct[:2000]}"
        elif action == "click":
            page.click(selector, timeout=5000)
            return f"Clicked {selector}"
        elif action == "type":
            page.fill(selector, text_input)
            return "Typed."
        elif action == "screenshot":
            path = _screenshot("manual")
            return f"__IMAGE__{path}"
        elif action == "get_text":
            el = page.query_selector(selector or "body")
            return el.inner_text()[:3000] if el else ""
        elif action == "scroll":
            page.evaluate("window.scrollBy(0,600)")
            return "Scrolled."
        elif action == "run_js":
            return str(page.evaluate(js_code))
        elif action == "close":
            global _browser, _page
            if _browser:
                _browser.close()
            _browser = _page = None
            return "Browser closed."
        return f"Unknown: {action}"
    except Exception as e:
        return f"Browser error: {e}"

# ── Web tools ───────────────────────────────────────────────────
def _t_web_search(query, max_results=5, news=False) -> str:
    """Search the web via DuckDuckGo with caching and circuit breaker."""
    cached = _tool_cache.get(f"s:{query}:{news}")
    if cached:
        return cached
    def _do():
        out = []
        with DDGS() as ddgs:
            for r in (ddgs.news if news else ddgs.text)(query, max_results=max_results * 2, safesearch="off"):
                url = r.get("href", r.get("url", ""))
                body = trafilatura.extract(trafilatura.fetch_url(url) or "") if url else ""
                body = body or r.get("body", "")
                if body and len(body.split()) > 20:
                    rag_add(body[:3000], "web", r.get("title", ""), url=url)
                    out.append(f"**{r.get('title', '')}**\n{body[:400]}")
                    time.sleep(0.2)
                if len(out) >= max_results:
                    break
        return "\n\n---\n".join(out) if out else "No results."
    result = with_retry(_do, "ddg")
    if "Error" not in str(result):
        _tool_cache.set(f"s:{query}:{news}", result)
    return result

def _t_crawl(url) -> str:
    """Fetch and extract content from a URL (crawl4ai → trafilatura fallback)."""
    try:
        from crawl4ai import AsyncWebCrawler
        async def _crawl():
            async with AsyncWebCrawler() as c:
                r = await c.arun(url=url)
                return r.markdown or ""
        text = asyncio.get_event_loop().run_until_complete(_crawl())
        if text and len(text.split()) > 50:
            rag_add(text[:5000], "crawl", url)
            return text[:3000]
    except Exception:
        pass
    try:
        text = trafilatura.extract(trafilatura.fetch_url(url) or "") or ""
        if text:
            rag_add(text[:5000], "scrape", url)
            return text[:3000]
    except Exception:
        pass
    return _t_web_search(url, 1)

def _t_wikipedia(query) -> str:
    cached = _tool_cache.get(f"w:{query}")
    if cached:
        return cached
    def _do():
        hits = wikipedia_lib.search(query, results=3)
        pg = wikipedia_lib.page(hits[0], auto_suggest=False)
        rag_add(pg.content, "wikipedia", pg.title)
        return f"**{pg.title}**\n\n{pg.content[:2500]}"
    result = with_retry(_do, "wikipedia")
    if "Error" not in str(result):
        _tool_cache.set(f"w:{query}", result)
    return result

def _t_arxiv(query, n=3) -> str:
    cached = _tool_cache.get(f"ax:{query}:{n}")
    if cached:
        return cached
    def _do():
        papers = []
        for p in arxiv_lib.Client().results(
            arxiv_lib.Search(query=query, max_results=int(n), sort_by=arxiv_lib.SortCriterion.Relevance)
        ):
            rag_add(f"Title:{p.title}\nAbstract:{p.summary}", "arxiv", p.title)
            papers.append(f"**{p.title}** ({p.published.year})\n{p.summary[:400]}")
        return "\n\n---\n".join(papers) if papers else "No papers."
    result = with_retry(_do, "arxiv")
    if "Error" not in str(result):
        _tool_cache.set(f"ax:{query}:{n}", result)
    return result

def _t_calculate(expr) -> str:
    """Safe mathematical expression evaluator."""
    try:
        import ast as _a, operator as op
        ops = {_a.Add: op.add, _a.Sub: op.sub, _a.Mult: op.mul, _a.Div: op.truediv,
               _a.Pow: op.pow, _a.USub: op.neg}
        def ev(n):
            if isinstance(n, _a.Constant):
                return n.value
            if isinstance(n, _a.BinOp):
                return ops[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, _a.UnaryOp):
                return ops[type(n.op)](ev(n.operand))
            raise ValueError
        return str(ev(_a.parse(expr, mode="eval").body))
    except Exception:
        try:
            import math as _m
            safe = {k: getattr(_m, k) for k in dir(_m) if not k.startswith("_")}
            safe.update({"abs": abs, "round": round, "sum": sum, "min": min, "max": max})
            return str(eval(expr, {"__builtins__": {}}, safe))  # noqa
        except Exception as e:
            return f"Calc error: {e}"

def _t_deep_research(topic, depth=3) -> str:
    """Multi-source research report with parallel fetching."""
    depth = min(int(depth), 5)
    with ThreadPoolExecutor(max_workers=3) as ex:
        fw = ex.submit(_t_wikipedia, topic)
        fa = ex.submit(_t_arxiv, topic, min(depth, 5))
        fn = ex.submit(_t_web_search, topic, 3, True)
        parts = [
            f"## Wikipedia\n{fw.result()[:800]}",
            f"## Academic\n{fa.result()[:1000]}",
            f"## News\n{fn.result()[:600]}"
        ]
    try:
        raw = quick_routed(f"List {depth} subtopics for '{topic}' as JSON array.", max_tokens=60, temp=0.4)
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        subs = json.loads(m.group()) if m else [topic]
    except Exception:
        subs = [topic]
    with ThreadPoolExecutor(max_workers=min(depth, 4)) as ex:
        futs = {ex.submit(_t_web_search, s, 2): s for s in subs[:depth]}
        for f, s in futs.items():
            parts.append(f"## {s}\n{f.result()[:500]}")
    return quick_routed(
        f"Write a comprehensive research report on '{topic}'.\nData:\n{chr(10).join(parts)[:5000]}",
        max_tokens=2000, temp=0.45
    )

# ── Workspace (with path validation) ───────────────────────────
def _validate_path(rel_path: str) -> Optional[Path]:
    """Validate workspace path to prevent directory traversal."""
    resolved = (Path(WORKSPACE) / rel_path).resolve()
    ws_resolved = Path(WORKSPACE).resolve()
    if not str(resolved).startswith(str(ws_resolved)):
        return None
    return resolved

def ws_read(rel_path):
    p = _validate_path(rel_path)
    if p is None:
        return "❌ Invalid path (directory traversal blocked)."
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else f"Not found: {rel_path}"

def ws_write(rel_path, content):
    p = _validate_path(rel_path)
    if p is None:
        return "❌ Invalid path (directory traversal blocked)."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"✅ Written: {rel_path}"

def ws_list(subdir=""):
    base = _validate_path(subdir) if subdir else Path(WORKSPACE)
    if base is None:
        return "❌ Invalid path."
    files = [f for f in base.rglob("*") if f.is_file()]
    return f"Workspace ({len(files)} files):\n" + "\n".join(
        f"  {f.relative_to(WORKSPACE)}" for f in files[:50]
    )

def ws_note(content):
    with open(WS_NOTES, "a") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M}\n{content}\n")
    return "✅ Note saved."

# ── Code tools (improved sandbox) ──────────────────────────────
_kernel_globals = dict(safe_globals)
_kernel_globals["__builtins__"] = dict(safe_builtins)
_kernel_globals["_getiter_"] = iter
_kernel_globals["_getattr_"] = getattr
_kernel_globals["_write_"] = lambda x: x
import math as _m2, random as _r2, statistics as _s2, datetime as _dt2
_kernel_globals.update({
    "math": _m2, "random": _r2, "statistics": _s2, "json": json, "re": re, "np": np, "pd": pd, "datetime": _dt2,
    "range": range, "len": len, "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "bool": bool,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "type": type, "isinstance": isinstance, "any": any, "all": all, "print": None, "tuple": tuple, "map": map, "filter": filter,
})

_BASH_ALLOWLIST = {
    "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "python", "python3", "pip", "git",
    "mkdir", "cp", "mv", "zip", "unzip", "df", "du", "which", "date", "diff", "awk", "sed", "sort", "uniq", "curl", "wget", "tar"
}

def run_code(language, code, timeout=30, persistent=True) -> str:
    """Execute code in a sandboxed environment. Supports python, bash, sql, javascript."""
    lang = language.lower().strip()
    if lang in ("python", "py"):
        try:
            bc = compile_restricted(code, "<kernel>", "exec")
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        buf = io.StringIO()
        err = [None]
        done = threading.Event()
        ctx = _kernel_globals if persistent else dict(_kernel_globals)
        ctx["print"] = lambda *a, **kw: print(*a, file=buf, **kw)
        # Limit kernel size to prevent memory leak
        if persistent and len(_kernel_globals) > Config.KERNEL_VAR_LIMIT:
            keys_to_remove = list(_kernel_globals.keys())[len(safe_globals) + 20:][:50]
            for k in keys_to_remove:
                _kernel_globals.pop(k, None)
        def _run():
            try:
                exec(bc, ctx)  # noqa
            except Exception as ex:
                err[0] = str(ex)
            finally:
                done.set()
        threading.Thread(target=_run, daemon=True).start()
        if not done.wait(timeout=timeout):
            return f"⏱️ Timed out ({timeout}s)"
        out = buf.getvalue()
        return (f"Error: {err[0]}\n{out}" if err[0] else out) or "(no output)"
    elif lang in ("bash", "sh"):
        first = code.strip().split()[0].split("/")[-1] if code.strip() else ""
        if first and first not in _BASH_ALLOWLIST:
            return f"❌ '{first}' not in allowlist."
        for bad in [r"rm\s+-rf\s+[/~]", r"curl.+\|\s*(bash|sh)", r":(){ :\|:& };:", r">\s*/dev/sd"]:
            if re.search(bad, code, re.I):
                return "❌ Blocked (dangerous command)."
        try:
            r = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=timeout)
            return (r.stdout + r.stderr)[:3000] or "(no output)"
        except subprocess.TimeoutExpired:
            return "⏱️ Timed out."
    elif lang in ("sql", "sqlite"):
        import sqlite3
        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            for stmt in code.split(";"):
                if stmt.strip():
                    cur.execute(stmt.strip())
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            conn.close()
            if rows:
                return " | ".join(cols) + "\n" + "\n".join(" | ".join(str(c) for c in r) for r in rows[:100])
            return "OK — no rows."
        except Exception as e:
            return f"SQL error: {e}"
    elif lang in ("js", "javascript", "node"):
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            subprocess.run(["apt", "install", "-y", "nodejs"], capture_output=True)
            node = shutil.which("node") or "node"
        try:
            with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
                f.write(code)
                fname = f.name
            r = subprocess.run([node, fname], capture_output=True, text=True, timeout=timeout)
            os.unlink(fname)
            return (r.stdout + r.stderr)[:2000] or "(no output)"
        except subprocess.TimeoutExpired:
            return "⏱️ Timed out."
    return f"❌ Unsupported: {language}"

def auto_fix_loop(language, code, max_iters=10) -> str:
    """Iteratively run and fix code until it works."""
    for i in range(max_iters):
        out = run_code(language, code)
        if "Error" not in out and "error" not in out.lower():
            return f"✅ Fixed in {i+1} iter(s):\n```{language}\n{code}\n```\nOutput:\n{out}"
        fixed = quick(
            f"Fix this {language} code:\n```{language}\n{code}\n```\nError:{out}\nReturn ONLY corrected code.",
            system=f"Expert {language} programmer.", max_tokens=600, temp=0.2
        )
        code = re.sub(r"```\w*\n?|```", "", fixed).strip()
    return f"Could not fix after {max_iters} attempts."

def code_review(code, language="python") -> str:
    return quick(
        f"Review this {language} code:\n```{language}\n{code}\n```\nCover: correctness, edge cases, performance, security, style.",
        system="Senior software engineer.", max_tokens=1000, temp=0.4
    )

def generate_tests(code, language="python") -> str:
    tests = quick(
        f"Write comprehensive pytest tests for:\n```{language}\n{code}\n```\nReturn ONLY test code.",
        system="Expert test engineer.", max_tokens=800, temp=0.2
    )
    return auto_fix_loop(language, code + "\n\n" + re.sub(r"```\w*\n?|```", "", tests).strip(), max_iters=5)

def generate_docs(code, language="python") -> str:
    return quick(
        f"Add docstrings and type hints to:\n```{language}\n{code}\n```\nReturn documented code + README section.",
        system="Technical documentation expert.", max_tokens=800, temp=0.3
    )

def scan_project(folder_path) -> str:
    if not Path(folder_path).exists():
        return f"Not found: {folder_path}"
    code_ext = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".go", ".rs", ".sql", ".sh"}
    previews = []
    for f in list(Path(folder_path).rglob("*"))[:30]:
        if f.is_file() and f.suffix in code_ext:
            try:
                ct = f.read_text(encoding="utf-8", errors="ignore")[:2000]
                rag_add(ct[:3000], "project", str(f.relative_to(folder_path)), filename=f.name)
                previews.append(f"=== {f.relative_to(folder_path)} ===\n{ct[:400]}")
            except Exception:
                pass
    return quick_routed(
        f"Project at {folder_path}.\nPreviews:\n{chr(10).join(previews[:8])}\n"
        f"Describe purpose, architecture, main components.",
        max_tokens=500, temp=0.4
    )

# ── Homework tools ──────────────────────────────────────────────
def solve_homework(question, subject="auto", show_working=True) -> str:
    if subject == "auto":
        subject = detect_subject(question)
    sys_p = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["general"])
    sympy_result = None
    if subject == "maths":
        try:
            code = quick_routed(f"Write SymPy code to solve: {question}\nReturn ONLY code.", max_tokens=200, temp=0.1)
            code = re.sub(r"```python\n?|```", "", code).strip()
            r = run_code("python", code, timeout=10)
            if "Error" not in r:
                sympy_result = r
        except Exception:
            pass
    rag_ctx = "\n---\n".join(c["text"][:300] for c in rag_retrieve(question, k=3))
    prompt = (
        f"{'SHOW FULL STEP-BY-STEP WORKING. ' if show_working else ''}Solve this {subject} problem:\n\n{question}\n\n"
        + (f"SymPy verification: {sympy_result}\n" if sympy_result else "")
        + (f"Context:\n{rag_ctx[:400]}\n" if rag_ctx else "")
        + "Structure: 1.Understand 2.Method 3.Steps 4.Final answer 5.Check"
    )
    solution = quick_routed(prompt, system=sys_p, max_tokens=1200, temp=0.3)
    sd = f"{DIRS['ws_homework']}/{subject}"
    os.makedirs(sd, exist_ok=True)
    Path(f"{sd}/q_{int(time.time())}.md").write_text(f"## Q\n{question}\n\n## Solution\n{solution}")
    return solution

def study_notes(topic, subject="general") -> str:
    with ThreadPoolExecutor(max_workers=3) as ex:
        fw = ex.submit(_t_wikipedia, topic)
        fa = ex.submit(_t_arxiv, topic, 3)
        fweb = ex.submit(_t_web_search, topic, 3)
        notes = quick_routed(
            f"Comprehensive study notes on: {topic}\nSources:\n{fw.result()[:800]}\n{fa.result()[:600]}\n{fweb.result()[:400]}\n"
            f"Format: # Topic\n## Key Concepts\n## Formulas/Definitions\n## Examples\n## Summary",
            system=SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["general"]), max_tokens=1500, temp=0.4
        )
    fname = f"{DIRS['ws_research']}/{topic.replace(' ', '_')}_{datetime.now():%Y%m%d}.md"
    Path(fname).write_text(notes)
    return notes

def generate_flashcards(topic, n=10) -> str:
    rag_ctx = "\n---\n".join(c["text"][:300] for c in rag_retrieve(topic, k=5))
    cards = quick_routed(
        f"Create {n} flashcards for: {topic}\nContext:{rag_ctx[:800]}\nFormat each:\nQ: [question]\nA: [answer]\n---",
        max_tokens=800, temp=0.5
    )
    Path(f"{DIRS['ws_research']}/flashcards_{topic.replace(' ', '_')}.md").write_text(f"# Flashcards: {topic}\n\n{cards}")
    return cards

def essay_help(topic, essay_type="analytical", word_count=500) -> str:
    search = _t_web_search(topic, 3)
    rag_ctx = "\n---\n".join(c["text"][:300] for c in rag_retrieve(topic, k=4))
    return quick_routed(
        f"Write a {word_count}-word {essay_type} essay on: {topic}\n"
        f"Research:\n{rag_ctx[:800]}\n{search[:600]}\n"
        f"Structure: Intro+thesis → Body+evidence → Conclusion",
        system=SUBJECT_PROMPTS["english"], max_tokens=max(word_count * 2, 800), temp=0.6
    )

def read_paper(pdf_path) -> dict:
    try:
        try:
            from docling.document_converter import DocumentConverter
            text = DocumentConverter().convert(pdf_path).document.export_to_markdown()
        except Exception:
            text = "\n\n".join(pg.extract_text() or "" for pg in PyPDF2.PdfReader(open(pdf_path, "rb")).pages)
        rag_add(text[:8000], "paper", Path(pdf_path).stem, filename=Path(pdf_path).name)
        return {
            "title": quick(f"Extract paper title:\n{text[:500]}", max_tokens=50, temp=0.1).strip(),
            "abstract": quick(f"Extract abstract:\n{text[:3000]}", max_tokens=300, temp=0.1).strip(),
            "methodology": quick(f"Summarise methodology:\n{text[1000:5000]}", max_tokens=400, temp=0.2).strip(),
            "results": quick(f"Summarise results:\n{text[3000:8000]}", max_tokens=400, temp=0.2).strip()
        }
    except Exception as e:
        return {"error": str(e)}

def analyse_video_frames(video_path, n_frames=8, question="Describe what's happening") -> str:
    if not Path(video_path).exists():
        return f"Not found: {video_path}"
    _cv2_local = _get_cv2()
    cap = _cv2_local.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video."
    total = int(cap.get(_cv2_local.CAP_PROP_FRAME_COUNT))
    fps = cap.get(_cv2_local.CAP_PROP_FPS) or 30
    duration = total / fps
    indices = [int(i * total / n_frames) for i in range(min(n_frames, total))]
    descs = []
    for idx in indices:
        cap.set(_cv2_local.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = idx / fps
        fp = os.path.join(DIRS["outputs"], f"vf_{idx}.jpg")
        _cv2_local.imwrite(fp, frame)
        desc = quick(f"Describe this video frame at {ts:.1f}s of {duration:.0f}s video.", max_tokens=150, temp=0.3, image_path=fp)
        descs.append(f"[{ts:.1f}s] {desc}")
        os.unlink(fp)
    cap.release()
    audio_tr = ""
    try:
        ap = os.path.join(DIRS["outputs"], "vaudio.wav")
        subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", ap, "-y"],
                       capture_output=True, timeout=30)
        if Path(ap).exists():
            audio_tr = _get_whisper("base").transcribe(ap, language="en")["text"][:1000]
    except Exception:
        pass
    result = quick(
        f"Video ({duration:.0f}s, {len(descs)} frames):\n{''.join(descs)}\n"
        + (f"Audio:\n{audio_tr}\n" if audio_tr else "")
        + f"Answer: {question}",
        max_tokens=600, temp=0.4
    )
    rag_add(f"Video:{Path(video_path).stem}\n{''.join(descs)}\n{audio_tr}", "video", Path(video_path).stem)
    return result

def _youtube_analyse(url: str, question: str = "Summarise this video") -> str:
    """Download a YouTube or web video via yt-dlp, then analyse frames with VL model."""
    try:
        output_path = os.path.join(DIRS["outputs"], "yt_video.mp4")
        subprocess.run(
            ["yt-dlp", "-f", "worst[ext=mp4]/worst", "--max-filesize", "150M",
             "-o", output_path, url],
            capture_output=True, timeout=180,
        )
        if Path(output_path).exists():
            result = analyse_video_frames(output_path, n_frames=8, question=question)
            try:
                os.unlink(output_path)
            except Exception:
                pass
            return result
        return "❌ Video download failed. Check the URL and try again."
    except Exception as e:
        return f"❌ YouTube analysis error: {e}"

# ── Image Generation (Stable Diffusion — local, free) ──────────
# v11: SD now goes through the VRAM Juggler — Qwen is offloaded to CPU
# while SD generates, then reloaded. Prevents OOM on T4.
_sd_pipe = None

def _load_sd():
    """Load Stable Diffusion pipeline (lazy, first use only). Uses VRAM Juggler."""
    global _sd_pipe
    if not Config.ENABLE_IMAGE_GEN:
        return None
    if _sd_pipe is not None:
        return _sd_pipe
    try:
        from diffusers import StableDiffusionPipeline
        log.info("Loading Stable Diffusion...")
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            cache_dir=DIRS["sd_cache"]
        ).to("cuda")
        _sd_pipe.set_progress_bar_config(disable=True)
        log.info("Stable Diffusion ready")
        return _sd_pipe
    except Exception as e:
        log.warning(f"SD load failed: {e}, trying smaller model...")
        try:
            from diffusers import AutoPipelineForText2Image
            _sd_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=DIRS["sd_cache"]
            ).to("cuda")
            return _sd_pipe
        except Exception as e2:
            log.error(f"SD completely failed: {e2}")
            return None

def generate_image(prompt, negative_prompt="blurry, low quality, distorted", steps=4, width=512, height=512) -> str:
    """Generate an image using local Stable Diffusion. Offloads Qwen to CPU first."""
    if not Config.ENABLE_IMAGE_GEN:
        return "❌ Image generation disabled. Set Config.ENABLE_IMAGE_GEN=True."
    # Offload Qwen to CPU RAM to free VRAM for SD
    VRAMJuggler.offload("qwen")
    try:
        pipe = _load_sd()
        if pipe is None:
            return "❌ Stable Diffusion not available. Try restarting runtime."
        # Register SD with juggler if first time
        if "sd" not in VRAMJuggler.MODELS and pipe is not None:
            VRAMJuggler.register("sd", pipe, priority=1)
        with torch.no_grad():
            result = pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=int(steps), width=int(width), height=int(height),
                guidance_scale=0.0  # sd-turbo uses guidance_scale=0
            )
        img = result.images[0]
        fname = os.path.join(DIRS["outputs"], f"img_{int(time.time()*1000)}.png")
        img.save(fname)
        return f"__IMAGE__{fname}"
    except Exception as e:
        return f"❌ Image generation failed: {e}"
    finally:
        # Always reload Qwen back to VRAM after SD is done
        torch.cuda.empty_cache()
        VRAMJuggler.load_to_gpu("qwen")

# ── File Upload → RAG Pipeline ──────────────────────────────────
def process_upload(file_path: str) -> str:
    """Auto-process uploaded files into RAG knowledge base."""
    if not file_path or not Path(file_path).exists():
        return "No file."
    path = Path(file_path)
    ext = path.suffix.lower()
    text = ""
    try:
        if ext == ".pdf":
            try:
                from docling.document_converter import DocumentConverter
                text = DocumentConverter().convert(str(path)).document.export_to_markdown()
            except Exception:
                text = "\n\n".join(pg.extract_text() or "" for pg in PyPDF2.PdfReader(open(str(path), "rb")).pages)
        elif ext == ".docx":
            from docx import Document
            text = "\n\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
        elif ext in (".xlsx", ".csv"):
            df = pd.read_excel(str(path)) if ext == ".xlsx" else pd.read_csv(str(path))
            text = f"Columns: {', '.join(df.columns)}\n\n{df.head(50).to_string()}"
        elif ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".txt", ".md", ".json", ".sh", ".sql"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif ext in (".mp4", ".avi", ".mov", ".webm"):
            return analyse_video_frames(str(path))
        elif ext in (".mp3", ".wav", ".ogg"):
            text = _get_whisper("base").transcribe(str(path), language="en")["text"]
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            desc = quick("Describe this image in detail.", image_path=str(path), max_tokens=300, temp=0.3)
            text = f"Image: {path.name}\n{desc}"
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return f"❌ Unsupported: {ext}"
    except Exception as e:
        return f"❌ Error processing {path.name}: {e}"

    if text:
        added = rag_add(text[:10000], "upload", path.stem, filename=path.name)
        return f"✅ Processed {path.name}: {added} chunks added to knowledge base."
    return f"⚠️ No content extracted from {path.name}"

log.info("All tool functions ready")


