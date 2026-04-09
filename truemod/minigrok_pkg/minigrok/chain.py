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


