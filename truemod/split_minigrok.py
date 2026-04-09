#!/usr/bin/env python3
"""
split_minigrok.py — HARDENED MiniGrok v13.2 Monolith → Package Splitter
======================================================================
This version resolves missing imports and cross-module dependencies.

Usage:
    python split_minigrok.py --src minigrok_v13.2.py --out minigrok_pkg
"""

import re, sys, os, textwrap, shutil
from pathlib import Path

# ── Module Mapping ────────────────────────────────────────────────
MODULE_MAP = {
    "base": {
        "tags": ["0"], # Special tag for items extracted from §1
        "desc": "Logging, metadata, and core flags",
    },
    "config": {
        "tags": ["1", "2", "3", "5c"],
        "desc": "Configuration, system paths, exceptions",
    },
    "utils": {
        "tags": ["4"],
        "desc": "Bounded caches, circuit breakers, rate limiter",
    },
    "models": {
        "tags": ["5", "5b", "5d"],
        "desc": "Model loading, VRAM juggler, model router",
    },
    "generation": {
        "tags": ["7"],
        "desc": "Generation / streaming / context management",
    },
    "memory": {
        "tags": ["6", "10"],
        "desc": "Conversation memory, user profile, prioritized memory",
    },
    "rag": {
        "tags": ["8", "9", "15"],
        "desc": "RAG chunker, ChromaDB, graph RAG, MoE, response cache",
    },
    "voice": {
        "tags": ["11"],
        "desc": "AI voice: TTS (F5/Kokoro/gTTS), Whisper STT",
    },
    "tools": {
        "tags": ["12", "16"],
        "desc": "Tool validator, tool implementations",
    },
    "chain": {
        "tags": ["17"],
        "desc": "Multi-step tool chaining",
    },
    "registry": {
        "tags": ["18"],
        "desc": "Tool registry, plugin system",
    },
    "agent": {
        "tags": ["5c2", "5e", "13", "14", "22"],
        "desc": "Agent state machine, SQLite DPO, main loop",
    },
    "background": {
        "tags": ["19", "20", "21"],
        "desc": "Background learner, training system, status",
    },
    "app": {
        "tags": ["23"],
        "desc": "Gradio UI (all tabs), launch logic",
    },
    "tests/test_core": {
        "tags": ["25"],
        "desc": "Pytest test skeleton",
    },
}

# Cross-module symbol mapping (Heuristic based on v13.2 structure)
# This tells the splitter where to find functions used in other modules.
SYMBOL_LOCATIONS = {
    "Config": "config", "DIRS": "config", "WORKSPACE": "config", "STORAGE_LOCK": "config",
    "log": "base", "APP_NAME": "base", "APP_VERSION": "base", "IN_COLAB": "base",
    "LRUCache": "utils", "CircuitBreaker": "utils", "RateLimiter": "utils",
    "processor": "models", "model": "models", "VRAMJuggler": "models", "_model_lock": "models",
    "quick": "generation", "stream_gen": "generation", "_build_msgs": "generation",
    "db_add_dpo": "agent", "db_get_dpo": "agent", "agent_stream": "agent", "AgentPhase": "agent",
    "synthesise": "voice", "_get_whisper": "voice",
    "TOOL_REGISTRY": "registry", "register_tool": "registry",
    "browser_agent": "tools", "code_executor": "tools", "web_search": "tools",
    "system_status": "background", "health_check": "background",
    "create_ui": "app", "_launch": "app",
}

def discover_sections(lines):
    sec_re = re.compile(r"^\x23\s*\u00a7\s*([0-9a-zA-Z]+)\s+(.*)", re.I)
    sections = []
    for i, line in enumerate(lines):
        m = sec_re.match(line.strip())
        if m:
            tag = m.group(1).lower()
            sections.append((i, tag, m.group(2).strip()))
    return sections

def extract_section_blocks(lines, sections):
    blocks = {}
    sep_re = re.compile(r"^#\s*[═]{10,}")
    for idx, (sec_line, tag, title) in enumerate(sections):
        start = sec_line
        for j in range(sec_line - 1, max(sec_line - 5, 0), -1):
            if sep_re.match(lines[j].strip()): start = j; break
            if lines[j].strip() == "": start = j
        if idx + 1 < len(sections):
            next_sec_line = sections[idx + 1][0]
            end = next_sec_line
            for j in range(next_sec_line - 1, max(next_sec_line - 5, 0), -1):
                if sep_re.match(lines[j].strip()): end = j; break
                if lines[j].strip() == "": end = j
        else: end = len(lines)
        blocks[tag] = {"lines": lines[start:end], "title": title}
    return blocks

def get_standard_header(monolith_lines):
    """Extract §1 imports and basic setup as a common header."""
    header = []
    in_s1 = False
    for line in monolith_lines:
        if "# § 1  IMPORTS" in line: in_s1 = True
        if "# § 2  CONFIGURATION" in line: break
        if in_s1: header.append(line)
    
    # Filter out local log creation as it's moved to base.py
    clean_header = []
    skip = False
    for line in header:
        if 'log = logging.getLogger' in line: skip = True
        if skip and line.strip() == "": skip = False
        if not skip and "# §" not in line and "════" not in line:
            clean_header.append(line)
    return "".join(clean_header)

def build_base_module(monolith_lines):
    out = ['"""\nminigrok.base\nLogging, metadata, and core flags.\n"""\n']
    out.append("import json, logging, os\n\n")
    
    # Extract metadata
    for line in monolith_lines:
        if line.startswith("APP_NAME") or line.startswith("APP_VERSION"):
            out.append(line)
    
    # Extract IN_COLAB block
    recording = False
    for line in monolith_lines:
        if "try:" in line and "google.colab" in line: recording = True
        if recording:
            out.append(line)
            if "IN_COLAB = False" in line: recording = False
    
    # Extract Logging block
    out.append("\n# ── Logging ─────────────────────────────────────────────────────\n")
    recording = False
    for line in monolith_lines:
        if 'log = logging.getLogger("MiniGrok")' in line: recording = True
        if recording:
            out.append(line)
            if "log.addHandler(_handler)" in line: recording = False
            
    out.append("\nlog.info(\"Imports OK\")\n")
    return "".join(out)

def build_module(mod_name, tags, blocks, std_header):
    out = [f'"""\nminigrok.{mod_name}\n{MODULE_MAP[mod_name]["desc"]}\n"""\n\n']
    out.append(std_header)
    out.append("\n# Cross-module imports\n")
    out.append("from .base import log, APP_NAME, APP_VERSION, IN_COLAB\n")
    
    # Find used symbols to inject specific imports
    all_code = "".join(["".join(blocks[tag]["lines"]) for tag in tags if tag in blocks])
    referenced_mods = set()
    for symbol, loc in SYMBOL_LOCATIONS.items():
        if loc != mod_name and loc != "base" and re.search(r'\b' + symbol + r'\b', all_code):
            referenced_mods.add(loc)
    
    for ref in sorted(referenced_mods):
        # Specific symbols for the ref module
        symbols = [s for s, l in SYMBOL_LOCATIONS.items() if l == ref and re.search(r'\b' + s + r'\b', all_code)]
        if symbols:
            out.append(f"from .{ref} import {', '.join(symbols)}\n")

    for tag in tags:
        if tag not in blocks: continue
        block = blocks[tag]
        out.append(f"\n# {'─'*60}\n# § {tag.upper()}  {block['title']}\n# {'─'*60}\n")
        out.extend(block["lines"])
    
    return "".join(out)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="minigrok_v13.2.py")
    parser.add_argument("--out", default="minigrok_pkg")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists(): print(f"❌ Source not found"); sys.exit(1)

    with open(src, "r", encoding="utf-8") as f: all_lines = f.readlines()
    sections = discover_sections(all_lines)
    blocks = extract_section_blocks(all_lines, sections)
    std_header = get_standard_header(all_lines)

    if out.exists(): shutil.rmtree(out)
    pkg_dir = out / "minigrok"
    pkg_dir.mkdir(parents=True)

    print(f"🏗️ Generating modules...")
    
    # 1. Base module
    base_content = build_base_module(all_lines)
    (pkg_dir / "base.py").write_text(base_content, encoding="utf-8")
    
    # 2. Other modules
    for mod_name, info in MODULE_MAP.items():
        if mod_name == "base": continue
        content = build_module(mod_name, info["tags"], blocks, std_header)
        (pkg_dir / (mod_name.replace("/", os.sep) + ".py")).parent.mkdir(parents=True, exist_ok=True)
        (pkg_dir / (mod_name.replace("/", os.sep) + ".py")).write_text(content, encoding="utf-8")

    # 3. Support files (unchanged logic but using new main.py with nest_asyncio)
    (pkg_dir / "__init__.py").write_text("from .base import APP_NAME, APP_VERSION\n", encoding="utf-8")
    (out / "main.py").write_text(textwrap.dedent("""\
        import nest_asyncio
        try: nest_asyncio.apply()
        except: pass
        from minigrok.app import _launch
        if __name__ == '__main__': _launch()
    """), encoding="utf-8")

    # Syntax check
    print(f"\n🔍 Syntax checking...")
    for py_file in sorted(out.rglob("*.py")):
        try:
            import py_compile
            py_compile.compile(str(py_file), doraise=True)
            print(f" ✅ {py_file.name}")
        except Exception as e:
            print(f" ❌ {py_file.name}: {e}")

    print(f"\n✅ Ready at {out.resolve()}")

if __name__ == "__main__": main()
