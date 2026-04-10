"""
minigrok.background
Background learner, training system, status
"""


# Cross-module imports
import os, json, re, logging, threading, time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

import torch

from .base import log, APP_NAME, APP_VERSION
from .config import Config, DIRS, DR
from .models import (processor, model, VRAMJuggler, _model_lock,
                     MODEL_MAX_TOKENS, quick)
from .rag import rag_add, rag_retrieve, _col, KG, _bm25_docs
from .memory import _pmem, _saved_convs


# ════════════════════════════════════════════════════════════════════
# § 19  BACKGROUND CONTINUOUS LEARNER (thread-safe)
# ════════════════════════════════════════════════════════════════════

LEARN_TOPICS_PATH = Path(f"{DR}/memory/learn_topics.json")
LEARN_LOG_PATH    = Path(f"{DR}/memory/learn_log.json")
_learn_topics     = ["large language models", "retrieval augmented generation", "reinforcement learning"]
if LEARN_TOPICS_PATH.exists():
    try:
        _learn_topics = json.loads(LEARN_TOPICS_PATH.read_text())
    except Exception:
        pass
_learn_log = []
if LEARN_LOG_PATH.exists():
    try:
        _learn_log = json.loads(LEARN_LOG_PATH.read_text())
    except Exception:
        pass
_learn_active = threading.Event()

def add_learning_topic(topic: str) -> str:
    if topic not in _learn_topics:
        _learn_topics.append(topic)
        LEARN_TOPICS_PATH.write_text(json.dumps(_learn_topics, indent=2))
    return f"✅ Tracking: {', '.join(_learn_topics)}"

def _bg_learn_cycle():
    global processor, model
    log.info(f"Background learning: {', '.join(_learn_topics)}")
    new_examples = []
    papers_found = 0
    cutoff = datetime.now() - timedelta(days=7)
    for topic in _learn_topics:
        try:
            for paper in arxiv_lib.Client().results(
                arxiv_lib.Search(query=topic, max_results=3, sort_by=arxiv_lib.SortCriterion.SubmittedDate)
            ):
                if paper.published.replace(tzinfo=None) < cutoff:
                    continue
                ph = hashlib.md5(paper.title.encode()).hexdigest()
                if any(l.get("hash") == ph for l in _learn_log[-100:]):
                    continue
                rag_add(f"Title:{paper.title}\nAbstract:{paper.summary}", "arxiv_bg", paper.title)
                try:
                    raw = quick_routed(f"Generate Q&A about this paper:\n{paper.title}\n{paper.summary[:400]}\n"
                               f'JSON: {{"question":"...","answer":"..."}}', max_tokens=300, temp=0.6)
                    m = re.search(r'\{.*?\}', raw, re.DOTALL)
                    qa = json.loads(m.group()) if m else {}
                    if qa.get("question") and qa.get("answer"):
                        new_examples.append({"text": processor.apply_chat_template(
                            [{"role": "user", "content": qa["question"]},
                             {"role": "assistant", "content": qa["answer"]}], tokenize=False)})
                except Exception:
                    pass
                _learn_log.append({"hash": ph, "title": paper.title[:80], "topic": topic, "ts": datetime.now().isoformat()})
                papers_found += 1
        except Exception as e:
            log.warning(f"BG learn arXiv: {e}")
    # RSS feeds
    try:
        for feed_url in ["https://arxiv.org/rss/cs.AI", "https://arxiv.org/rss/cs.CL", "https://arxiv.org/rss/cs.LG"]:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                ct = entry.get("summary", "")
                url = entry.get("link", "")
                if ct and len(ct.split()) > 30:
                    rag_add(ct[:3000], "rss_bg", entry.get("title", ""), url=url)
    except Exception as e:
        log.warning(f"BG RSS: {e}")
    LEARN_LOG_PATH.write_text(json.dumps(_learn_log[-500:], indent=2))
    log.info(f"  {papers_found} papers, {len(new_examples)} examples")
    if len(new_examples) >= 10:
        log.info(f"  Training on {len(new_examples)} new examples…")
        try:
            train_ds = Dataset.from_list(new_examples)
            # Ensure Qwen is in VRAM before training (SD may have offloaded it)
            VRAMJuggler.load_to_gpu("qwen")
            with _model_lock:
                model.train()
                mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                               lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
                mk = get_peft_model(mk, lc)
                SFTTrainer(
                    model=mk, train_dataset=train_ds,
                    args=SFTConfig(
                        output_dir=DIRS["tmp_train"], num_train_epochs=1, per_device_train_batch_size=1,
                        gradient_accumulation_steps=4, warmup_steps=5, learning_rate=1e-4, fp16=True,
                        logging_steps=10, save_strategy="no", optim="paged_adamw_8bit",
                        max_seq_length=512, report_to="none", dataset_text_field="text"
                    ),
                    tokenizer=getattr(processor, "tokenizer", processor)
                ).train()
                merged = "/content/bg_merged"
                mk = mk.merge_and_unload()
                mk.save_pretrained(merged, safe_serialization=True)
                getattr(processor, "tokenizer", processor).save_pretrained(merged)
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                for p in [DIRS["tmp_train"], merged]:
                    shutil.rmtree(p, ignore_errors=True)
                del mk
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
            log.info(f"  BG training complete ({len(new_examples)} examples)")
        except Exception as e:
            log.warning(f"BG training failed: {e}")
            torch.cuda.empty_cache()
            model.eval()

def _bg_learn_worker():
    while _learn_active.is_set():
        try:
            _bg_learn_cycle()
        except Exception as e:
            log.warning(f"BG learn error: {e}")
        _learn_active.wait(timeout=Config.BG_LEARN_INTERVAL)

def start_learning() -> str:
    if not Config.ENABLE_BG_LEARNING:
        return "Background learning is disabled. Set Config.ENABLE_BG_LEARNING=True."
    if _learn_active.is_set():
        return "Already running."
    _learn_active.set()
    threading.Thread(target=_bg_learn_worker, daemon=True).start()
    return f"✅ Background learning started. Topics: {', '.join(_learn_topics)}"

def stop_learning() -> str:
    _learn_active.clear()
    return "✅ Background learning stopped."

def learning_status() -> str:
    active = "🟢 Active" if _learn_active.is_set() else "🔴 Stopped"
    recent = "\n".join(f"  • {l['title'][:60]} ({l['topic']}) — {l['ts'][:10]}" for l in _learn_log[-5:])
    return f"Status: {active}\nTopics: {', '.join(_learn_topics)}\nPapers: {len(_learn_log)}\nRecent:\n{recent or '  (none)'}"

TOOL_REGISTRY["add_learning_topic"] = {"desc": "Add topic for background arXiv learning", "params": {"topic": "str"}, "fn": add_learning_topic}
TOOL_REGISTRY["learning_status"] = {"desc": "Check background learning status", "params": {}, "fn": learning_status}
log.info("Background continual learner ready")



# ════════════════════════════════════════════════════════════════════
# § 20  TRAINING SYSTEM (Unsloth + DPO + Eval Gate + Auto-Rollback)
# ════════════════════════════════════════════════════════════════════

DATASET_CATALOGUE = {
    "LIMA (quality 1k)":        ("GAIR/lima", "train"),
    "Dolly 15k":                ("databricks/dolly-15k", "train"),
    "Code-Feedback (debug)":    ("m-a-p/Code-Feedback", "train[:2000]"),
    "OpenMathInstruct (math)":  ("nvidia/OpenMathInstruct-1", "train[:2000]"),
    "GSM8K":                    ("gsm8k", "train"),
    "Stack-Smol (200+ langs)":  ("bigcode/the-stack-smol", "train[:1000]"),
    "Open-Orca (reasoning)":    ("Open-Orca/OpenOrca", "train[:2000]"),
    "Wikipedia EN":             ("wikipedia", "train[:3000]"),
    "FineWeb (quality web)":    ("HuggingFaceFW/fineweb", "train[:1000]"),
    "Alpaca 52k":               ("tatsu-lab/alpaca", "train"),
}

EVAL_QS = [
    {"q": "What is the derivative of x^3 + 2x^2 - 5x + 3?", "type": "maths", "expected": ["3x^2", "4x", "-5"]},
    {"q": "Solve: 2x + 5 = 13", "type": "maths", "expected": ["4", "x = 4"]},
    {"q": "Write a Python function to check if a number is prime", "type": "coding", "expected": ["def", "prime", "return"]},
    {"q": "What is Newton's second law of motion?", "type": "science", "expected": ["F", "ma", "force", "mass", "acceleration"]},
    {"q": "What is the capital of Australia?", "type": "knowledge", "expected": ["Canberra"]},
    {"q": "Explain photosynthesis", "type": "science", "expected": ["light", "carbon dioxide", "glucose", "oxygen"]},
    {"q": "What year did World War II end?", "type": "knowledge", "expected": ["1945"]},
    {"q": "What is time complexity of binary search?", "type": "coding", "expected": ["log", "O(log n)"]},
    {"q": "If all roses are flowers and some flowers fade, can we conclude all roses fade?", "type": "reasoning", "expected": ["no", "cannot", "not necessarily"]},
    {"q": "Write a SQL query to select the top 5 highest salaries", "type": "coding", "expected": ["SELECT", "ORDER BY", "DESC", "LIMIT", "5"]},
]

def _score_answer(q_obj, answer) -> float:
    al = answer.lower()
    kws = q_obj.get("expected", [])
    kw_score = sum(1 for k in kws if k.lower() in al) / max(len(kws), 1) if kws else 0.5
    raw = quick_routed(f"Q:{q_obj['q']}\nA:{answer[:400]}\nRate correctness 0.0-1.0. Number only.", max_tokens=5, temp=0.1)
    try:
        model_score = float(re.search(r'0?\.\d+|1\.0|0|1', raw).group())
    except Exception:
        model_score = 0.5
    return round(0.5 * kw_score + 0.5 * model_score, 3)

def run_eval(label="eval", log_fn=None) -> dict:
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n📊 Eval: {label} ({len(EVAL_QS)} questions)")
    scores = []
    type_scores = {}
    for q in EVAL_QS:
        ans = quick(q["q"], max_tokens=300, temp=0.3)
        ans = _strip_think(ans)
        s = _score_answer(q, ans)
        scores.append(s)
        type_scores.setdefault(q["type"], []).append(s)
        _log(f"  {'✅' if s >= 0.7 else '⚠️' if s >= 0.4 else '❌'} {s:.2f} | {q['q'][:50]}")
    overall = sum(scores) / len(scores) if scores else 0
    by_type = {t: round(sum(v) / len(v), 3) for t, v in type_scores.items()}
    _log(f"\n  Overall: {overall:.1%}")
    for t, v in by_type.items():
        _log(f"  {t:12s}: {v:.1%}")
    result = {"label": label, "overall": round(overall, 3), "by_type": by_type, "timestamp": datetime.now().isoformat()}
    db_add_bench(label, result["overall"], by_type)
    return result

def eval_comparison(before, after) -> Tuple[bool, str]:
    b = before.get("overall", 0)
    a = after.get("overall", 0)
    delta = a - b
    regressions = [f"{t}: {before['by_type'].get(t, 0):.1%}→{v:.1%}"
                   for t, v in after.get("by_type", {}).items()
                   if v < before.get("by_type", {}).get(t, 0) - 0.1]
    if a > b:
        return True, f"✅ {b:.1%}→{a:.1%} (+{delta:.1%})" + (f" (minor regressions: {', '.join(regressions)})" if regressions else "")
    return False, f"❌ No improvement: {b:.1%}→{a:.1%} ({delta:+.1%}). Rolling back."

def _fmt(row):
    for qk, ak in [("instruction", "output"), ("prompt", "completion"), ("question", "answer")]:
        if qk in row and ak in row and str(row.get(ak, "")).strip():
            return processor.apply_chat_template(
                [{"role": "user", "content": str(row[qk])[:600]},
                 {"role": "assistant", "content": str(row[ak])[:1200]}], tokenize=False)
    for key in ["text", "content", "passage"]:
        if key in row and len(str(row.get(key, ""))) > 80:
            t = str(row[key])[:1500]
            return processor.apply_chat_template(
                [{"role": "user", "content": f"Explain: {t.split('.')[0].strip()}"},
                 {"role": "assistant", "content": t}], tokenize=False)
    return None

def score_ex(text):
    """Score training example quality (higher = better)."""
    penalties = sum(1 for s in ["as an ai", "as a large language", "in today's fast-paced", "leveraging the power"]
                    if s in text.lower()) * 0.2
    length_bonus = min(len(text.split()) / 500, 0.2)
    return max(0.0, min(1.0, 1.0 - penalties + length_bonus))

def synthetic_data(topic, n=50):
    examples = []
    for _ in range(n):
        raw = quick_routed(f"Generate Q&A about '{topic}'.\nJSON: {{\"question\":\"...\",\"answer\":\"...\"}}", max_tokens=300, temp=0.85)
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            qa = json.loads(m.group()) if m else {}
            if qa.get("question") and qa.get("answer"):
                examples.append({"text": processor.apply_chat_template(
                    [{"role": "user", "content": qa["question"]},
                     {"role": "assistant", "content": qa["answer"]}], tokenize=False)})
        except Exception:
            continue
    return examples

def stream_train(dataset_ids, max_per_ds=500, save_drive=True, use_unsloth=True,
                  use_curriculum=True, quality_threshold=0.4, synthetic_topics=None,
                  run_bench=True, log_fn=None) -> str:
    global processor, model
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n🚀 Streaming Train — {len(dataset_ids)} datasets\n")
    before_eval = None
    if run_bench:
        _log("\n📊 Running BEFORE eval…")
        before_eval = run_eval("before", log_fn)
        rb = f"{DIRS['checkpts']}/rollback_{int(time.time())}"
        os.makedirs(rb, exist_ok=True)
        for f in Path(MODEL_DIR).glob("*"):
            if f.is_file():
                shutil.copy(f, rb)
        _log(f"  💾 Rollback saved: {rb}")
    examples = []
    for ds_id in dataset_ids:
        hf, split = DATASET_CATALOGUE.get(ds_id, (ds_id, "train[:500]"))
        _log(f"  📡 {hf}")
        try:
            ds = hf_load(hf, split=split, streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if count >= max_per_ds:
                    break
                fmt = _fmt(row)
                if fmt and score_ex(fmt) >= quality_threshold:
                    examples.append({"text": fmt, "quality": score_ex(fmt)})
                    count += 1
            _log(f"     ✓ {count}")
        except Exception as e:
            _log(f"     ⚠️  {e}")
    if synthetic_topics:
        for t in synthetic_topics:
            _log(f"  🧬 Synthetic: {t}")
            s = synthetic_data(t, n=50)
            examples.extend([{**x, "quality": 0.8} for x in s])
    if not examples:
        return "❌ No examples."
    if use_curriculum:
        examples.sort(key=lambda x: (-x["quality"], len(x["text"].split())))
    train_ds = Dataset.from_list([{"text": e["text"]} for e in examples])
    _log(f"  Total: {len(train_ds)} examples")
    with _model_lock:
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel
                um, ut = FastLanguageModel.from_pretrained(model_name=MODEL_DIR, max_seq_length=1024, load_in_4bit=True)
                um = FastLanguageModel.get_peft_model(um, r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA,
                    target_modules=Config.LORA_TARGETS, lora_dropout=Config.LORA_DROPOUT,
                    bias="none", use_gradient_checkpointing=True)
                SFTTrainer(model=um, train_dataset=train_ds,
                    args=SFTConfig(output_dir=DIRS["checkpts"], num_train_epochs=1, per_device_train_batch_size=2,
                        gradient_accumulation_steps=4, warmup_steps=10, learning_rate=2e-4, fp16=True,
                        logging_steps=20, save_strategy="steps", save_steps=100, optim="adamw_8bit",
                        max_seq_length=1024, report_to="none", dataset_text_field="text"),
                    tokenizer=ut).train()
                merged = "/content/merged_tmp"
                um.save_pretrained_merged(merged, ut, save_method="merged_16bit")
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                if save_drive:
                    shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                shutil.rmtree(merged, ignore_errors=True)
                del um
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
                # v13: Update active model info after fine-tune
                _active_model_info.update({"id": Config.MODEL_ID, "vision": True, "dir": MODEL_DIR, "fine_tuned": True})
                VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}
                _log("✅ Unsloth done — fine-tuned model now active")
            except Exception as e:
                _log(f"  ⚠️  Unsloth failed ({e}), TRL fallback…")
                model.train()
                mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                if hasattr(mk, "visual"):
                    for p in mk.visual.parameters():
                        p.requires_grad = False
                lc = LoraConfig(r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA, target_modules=Config.LORA_TARGETS,
                    lora_dropout=Config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
                mk = get_peft_model(mk, lc)
                SFTTrainer(model=mk, train_dataset=train_ds,
                    args=SFTConfig(output_dir=DIRS["checkpts"], num_train_epochs=1, per_device_train_batch_size=1,
                        gradient_accumulation_steps=8, warmup_steps=10, learning_rate=2e-4, fp16=True,
                        logging_steps=20, save_strategy="steps", save_steps=100, optim="paged_adamw_8bit",
                        max_seq_length=1024, report_to="none", dataset_text_field="text"),
                    tokenizer=getattr(processor, "tokenizer", processor)).train()
                merged = "/content/merged_trl"
                mk = mk.merge_and_unload()
                mk.save_pretrained(merged, safe_serialization=True)
                getattr(processor, "tokenizer", processor).save_pretrained(merged)
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                if save_drive:
                    shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                shutil.rmtree(merged, ignore_errors=True)
                del mk
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
                _active_model_info.update({"id": Config.MODEL_ID, "vision": True, "dir": MODEL_DIR, "fine_tuned": True})
                VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}
    if run_bench and before_eval:
        _log("\n📊 Running AFTER eval…")
        after_eval = run_eval("after", log_fn)
        should_keep, explanation = eval_comparison(before_eval, after_eval)
        _log(f"\n{explanation}")
        if not should_keep:
            _log("  🔄 Rolling back…")
            rbs = sorted(Path(DIRS["checkpts"]).glob("rollback_*"))
            if rbs:
                with _model_lock:
                    shutil.copytree(str(rbs[-1]), MODEL_DIR, dirs_exist_ok=True)
                    processor, model = _load_model(MODEL_DIR)
                    model.eval()
                _log("  ✅ Rolled back")
            return f"⚠️ ROLLED BACK: {explanation}"
        for old in sorted(Path(DIRS["checkpts"]).glob("rollback_*"))[:-3]:
            shutil.rmtree(old, ignore_errors=True)
    return f"✅ Training complete. {len(train_ds)} examples."

def dpo_train(log_fn=None) -> str:
    global processor, model
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    if len(_dpo_pairs) < 10:
        return f"❌ Need ≥10 pairs. Have {len(_dpo_pairs)}."
    _log(f"\n🎯 DPO on {len(_dpo_pairs)} pairs…")
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in _dpo_pairs])
    with _model_lock:
        model.train()
        mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                       lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        mk = get_peft_model(mk, lc)
        DPOTrainer(model=mk, ref_model=None,
            args=DPOConfig(output_dir=DIRS["tmp_train"], num_train_epochs=1, per_device_train_batch_size=1,
                gradient_accumulation_steps=4, learning_rate=5e-5, fp16=True, save_strategy="no",
                report_to="none", beta=0.1),
            train_dataset=ds, tokenizer=getattr(processor, "tokenizer", processor)).train()
        merged = "/content/dpo_m"
        mk = mk.merge_and_unload()
        mk.save_pretrained(merged, safe_serialization=True)
        getattr(processor, "tokenizer", processor).save_pretrained(merged)
        shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
        shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
        for p in [DIRS["tmp_train"], merged]:
            shutil.rmtree(p, ignore_errors=True)
        del mk
        torch.cuda.empty_cache()
        processor, model = _load_model(MODEL_DIR)
        model.eval()
    # v13: Auto-benchmark after DPO (Grok recommendation)
    log.info("Auto-benchmark after DPO…")
    try:
        after = run_eval("after_dpo", log_fn=log_fn)
        db_add_bench("after_dpo", after["overall"], after["by_type"])
        if log_fn:
            log_fn(f"  📊 Post-DPO eval: {after['overall']:.1%}")
    except Exception as e:
        log.warning(f"Post-DPO eval failed: {e}")
    return f"✅ DPO done. {len(_dpo_pairs)} pairs."

log.info("Training system ready")



# ════════════════════════════════════════════════════════════════════
# § 21  RAG QUALITY AUDIT + SYSTEM STATUS
# ════════════════════════════════════════════════════════════════════

def rag_audit(test_queries=None, k=5) -> dict:
    if test_queries is None:
        test_queries = [q["q"] for q in EVAL_QS[:5]]
    results = []
    for q in test_queries:
        t0 = time.time()
        chunks = rag_retrieve(q, k=k)
        latency = time.time() - t0
        results.append({"query": q[:50], "latency_ms": round(latency * 1000),
                        "chunks": len(chunks), "top_score": max((c.get("score", 0) for c in chunks), default=0)})
    avg_lat = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    avg_k = sum(r["chunks"] for r in results) / len(results) if results else 0
    return {"tests": results, "avg_latency_ms": round(avg_lat), "avg_chunks": round(avg_k, 1),
            "total_docs": _col.count(), "kg_nodes": KG.number_of_nodes(), "kg_edges": KG.number_of_edges()}

def system_status() -> str:
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_free = _vram_free()
    juggler_models = ", ".join(
        f"{n}({'VRAM' if e['in_vram'] else 'CPU'})"
        for n, e in VRAMJuggler.MODELS.items()
    ) if VRAMJuggler.MODELS else "none registered"
    return (
        f"## {APP_NAME} v{APP_VERSION} Status\n\n"
        f"**GPU:** {GPU_NAME}\n"
        f"**VRAM:** {vram_used:.1f}GB used / {vram_free:.1f}GB free / {VRAM_GB:.1f}GB total\n"
        f"**VRAM Juggler:** {juggler_models}\n"
        f"**Model:** {Config.MODEL_ID} (4-bit)\n"
        f"**Adapter:** {_active_adapter or 'base'}\n"
        f"**Embedder:** {_EMBEDDER_NAME}\n"
        f"**Context:** {MODEL_MAX_TOKENS:,} tokens\n"
        f"**TTS:** {_tts_engine if Config.ENABLE_VOICE else '🔴 Disabled'}\n\n"
        f"**Features:** "
        f"{'🟢 ImgGen' if Config.ENABLE_IMAGE_GEN else '🔴 ImgGen'} | "
        f"{'🟢 Browser' if Config.ENABLE_BROWSER else '🔴 Browser'} | "
        f"{'🟢 BgLearn' if Config.ENABLE_BG_LEARNING else '🔴 BgLearn'} | "
        f"{'🟢 Reflect' if Config.ENABLE_REFLECTION else '🔴 Reflect'}\n\n"
        f"**RAG:** {_col.count()} chunks (BM25: {len(_bm25_docs)}) | KG: {KG.number_of_nodes()} nodes\n"
        f"**Memory:** {len(_pmem)} prioritized | {len(_saved_convs)} conversation msgs\n"
        f"**DPO:** {len(_dpo_pairs)} pairs | Corrections: {len(_correction_log)}\n"
        f"**Cache:** {len(_response_cache)} responses | {len(_tool_cache)} tool results\n"
        f"**Learning:** {'🟢 Active' if _learn_active.is_set() else '🔴 Stopped'} ({len(_learn_log)} papers)\n"
        f"**Plugins:** {_plugins_loaded} loaded from {DIRS['ws_plugins']}\n"
        f"**Tools:** {len(TOOL_REGISTRY)}\n\n"
        f"**Agent Stats:** {_agent_state.get('total_turns', 0)} turns | "
        f"{_agent_state.get('total_tool_calls', 0)} tool calls | "
        f"{_agent_state.get('corrections_made', 0)} auto-corrections\n"
        f"**Last query:** {_agent_state.get('last_message', '(none)')} "
        f"[{_agent_state.get('rag_hits', 0)} RAG hits]\n"
        f"**Models:** primary={_active_model_info.get('id','?')} "
        f"({'vision' if _active_model_info.get('vision') else 'text-only'}) | "
        f"fast={'ready' if _fast_model is not None else 'not loaded'}\n"
        + _bench_trend_str()
    )

def _bench_trend_str() -> str:
    """Return last 3 benchmark scores as a mini ASCII trend (for status text)."""
    history = db_get_bench_history(n=5)
    if not history:
        return ""
    history.reverse()
    bars = []
    for h in history[-3:]:
        pct = int(h["overall"] * 10)
        bar = "█" * pct + "░" * (10 - pct)
        bars.append(f"{bar} {h['overall']:.0%} ({h['ts'][:10]})")
    return "**Eval history:**\n" + "\n".join(f"  {b}" for b in bars) + "\n"

def _bench_plot() -> Optional[object]:
    """
    v13: Real matplotlib chart of benchmark history (Grok recommendation).
    Returns a Figure object for gr.Plot, or None if no data.
    """
    try:
        history = db_get_bench_history(n=20)
    except Exception:
        return None
    if len(history) < 2:
        return None
    history.reverse()   # Oldest first
    try:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#242424")
        ax.tick_params(colors="#888888")
        ax.spines[:].set_color("#3a3a3a")

        dates = [h["ts"][:10] for h in history]
        overall = [h["overall"] for h in history]

        ax.plot(dates, overall, "o-", color="#10a37f", linewidth=2, markersize=5, label="Overall")

        # Per-type lines if consistent
        all_types = set()
        for h in history:
            all_types.update(h["by_type"].keys())

        colors = {"maths":"#60a5fa","coding":"#f59e0b","science":"#a78bfa",
                  "knowledge":"#34d399","reasoning":"#f87171"}
        for t in sorted(all_types):
            vals = [h["by_type"].get(t) for h in history]
            if any(v is not None for v in vals):
                ys = [v if v is not None else float("nan") for v in vals]
                ax.plot(dates, ys, "--", color=colors.get(t,"#888"), linewidth=1,
                        alpha=0.7, label=t)

        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(f"{APP_NAME} Benchmark Trend", color="#f0f0f0", fontsize=10)
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="#f0f0f0", loc="lower right")
        plt.xticks(rotation=25, ha="right", fontsize=7, color="#888")
        plt.tight_layout()
        return fig
    except Exception as e:
        log.debug(f"bench_plot failed: {e}")
        return None

def health_check() -> dict:
    """/health endpoint — returns system health as JSON."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": _active_model_info.get("id", Config.MODEL_ID),
        "model_vision": _active_model_info.get("vision", True),
        "fast_model": Config.FAST_MODEL_ID if _fast_model is not None else None,
        "fallback_chain": [e["id"] for e in Config.MODEL_FALLBACK_CHAIN],
        "circuit_breakers": {
            name: {"failures": cb.fails, "is_open": cb.is_open(), "cooldown_s": cb.cooldown}
            for name, cb in _breakers.items()
        },
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_free_gb": round(_vram_free(), 2),
        "rag_chunks": _col.count(),
        "bm25_docs": len(_bm25_docs),
        "kg_nodes": KG.number_of_nodes(),
        "memory_entries": len(_pmem),
        "dpo_pairs": len(_dpo_pairs),
        "tools": len(TOOL_REGISTRY),
        "bg_learning": _learn_active.is_set(),
        "tts_engine": _tts_engine,
    }

log.info("RAG audit + system status + health ready")


"""
minigrok.background
Background learner, training system, status
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
from .models import processor, model, VRAMJuggler, _model_lock
from .registry import TOOL_REGISTRY

# ────────────────────────────────────────────────────────────
# § 19  BACKGROUND CONTINUOUS LEARNER (thread-safe)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 19  BACKGROUND CONTINUOUS LEARNER (thread-safe)
# ════════════════════════════════════════════════════════════════════

LEARN_TOPICS_PATH = Path(f"{DR}/memory/learn_topics.json")
LEARN_LOG_PATH    = Path(f"{DR}/memory/learn_log.json")
_learn_topics     = ["large language models", "retrieval augmented generation", "reinforcement learning"]
if LEARN_TOPICS_PATH.exists():
    try:
        _learn_topics = json.loads(LEARN_TOPICS_PATH.read_text())
    except Exception:
        pass
_learn_log = []
if LEARN_LOG_PATH.exists():
    try:
        _learn_log = json.loads(LEARN_LOG_PATH.read_text())
    except Exception:
        pass
_learn_active = threading.Event()

def add_learning_topic(topic: str) -> str:
    if topic not in _learn_topics:
        _learn_topics.append(topic)
        LEARN_TOPICS_PATH.write_text(json.dumps(_learn_topics, indent=2))
    return f"✅ Tracking: {', '.join(_learn_topics)}"

def _bg_learn_cycle():
    global processor, model
    log.info(f"Background learning: {', '.join(_learn_topics)}")
    new_examples = []
    papers_found = 0
    cutoff = datetime.now() - timedelta(days=7)
    for topic in _learn_topics:
        try:
            for paper in arxiv_lib.Client().results(
                arxiv_lib.Search(query=topic, max_results=3, sort_by=arxiv_lib.SortCriterion.SubmittedDate)
            ):
                if paper.published.replace(tzinfo=None) < cutoff:
                    continue
                ph = hashlib.md5(paper.title.encode()).hexdigest()
                if any(l.get("hash") == ph for l in _learn_log[-100:]):
                    continue
                rag_add(f"Title:{paper.title}\nAbstract:{paper.summary}", "arxiv_bg", paper.title)
                try:
                    raw = quick_routed(f"Generate Q&A about this paper:\n{paper.title}\n{paper.summary[:400]}\n"
                               f'JSON: {{"question":"...","answer":"..."}}', max_tokens=300, temp=0.6)
                    m = re.search(r'\{.*?\}', raw, re.DOTALL)
                    qa = json.loads(m.group()) if m else {}
                    if qa.get("question") and qa.get("answer"):
                        new_examples.append({"text": processor.apply_chat_template(
                            [{"role": "user", "content": qa["question"]},
                             {"role": "assistant", "content": qa["answer"]}], tokenize=False)})
                except Exception:
                    pass
                _learn_log.append({"hash": ph, "title": paper.title[:80], "topic": topic, "ts": datetime.now().isoformat()})
                papers_found += 1
        except Exception as e:
            log.warning(f"BG learn arXiv: {e}")
    # RSS feeds
    try:
        for feed_url in ["https://arxiv.org/rss/cs.AI", "https://arxiv.org/rss/cs.CL", "https://arxiv.org/rss/cs.LG"]:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                ct = entry.get("summary", "")
                url = entry.get("link", "")
                if ct and len(ct.split()) > 30:
                    rag_add(ct[:3000], "rss_bg", entry.get("title", ""), url=url)
    except Exception as e:
        log.warning(f"BG RSS: {e}")
    LEARN_LOG_PATH.write_text(json.dumps(_learn_log[-500:], indent=2))
    log.info(f"  {papers_found} papers, {len(new_examples)} examples")
    if len(new_examples) >= 10:
        log.info(f"  Training on {len(new_examples)} new examples…")
        try:
            train_ds = Dataset.from_list(new_examples)
            # Ensure Qwen is in VRAM before training (SD may have offloaded it)
            VRAMJuggler.load_to_gpu("qwen")
            with _model_lock:
                model.train()
                mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                               lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
                mk = get_peft_model(mk, lc)
                SFTTrainer(
                    model=mk, train_dataset=train_ds,
                    args=SFTConfig(
                        output_dir=DIRS["tmp_train"], num_train_epochs=1, per_device_train_batch_size=1,
                        gradient_accumulation_steps=4, warmup_steps=5, learning_rate=1e-4, fp16=True,
                        logging_steps=10, save_strategy="no", optim="paged_adamw_8bit",
                        max_seq_length=512, report_to="none", dataset_text_field="text"
                    ),
                    tokenizer=getattr(processor, "tokenizer", processor)
                ).train()
                merged = "/content/bg_merged"
                mk = mk.merge_and_unload()
                mk.save_pretrained(merged, safe_serialization=True)
                getattr(processor, "tokenizer", processor).save_pretrained(merged)
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                for p in [DIRS["tmp_train"], merged]:
                    shutil.rmtree(p, ignore_errors=True)
                del mk
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
            log.info(f"  BG training complete ({len(new_examples)} examples)")
        except Exception as e:
            log.warning(f"BG training failed: {e}")
            torch.cuda.empty_cache()
            model.eval()

def _bg_learn_worker():
    while _learn_active.is_set():
        try:
            _bg_learn_cycle()
        except Exception as e:
            log.warning(f"BG learn error: {e}")
        _learn_active.wait(timeout=Config.BG_LEARN_INTERVAL)

def start_learning() -> str:
    if not Config.ENABLE_BG_LEARNING:
        return "Background learning is disabled. Set Config.ENABLE_BG_LEARNING=True."
    if _learn_active.is_set():
        return "Already running."
    _learn_active.set()
    threading.Thread(target=_bg_learn_worker, daemon=True).start()
    return f"✅ Background learning started. Topics: {', '.join(_learn_topics)}"

def stop_learning() -> str:
    _learn_active.clear()
    return "✅ Background learning stopped."

def learning_status() -> str:
    active = "🟢 Active" if _learn_active.is_set() else "🔴 Stopped"
    recent = "\n".join(f"  • {l['title'][:60]} ({l['topic']}) — {l['ts'][:10]}" for l in _learn_log[-5:])
    return f"Status: {active}\nTopics: {', '.join(_learn_topics)}\nPapers: {len(_learn_log)}\nRecent:\n{recent or '  (none)'}"

TOOL_REGISTRY["add_learning_topic"] = {"desc": "Add topic for background arXiv learning", "params": {"topic": "str"}, "fn": add_learning_topic}
TOOL_REGISTRY["learning_status"] = {"desc": "Check background learning status", "params": {}, "fn": learning_status}
log.info("Background continual learner ready")



# ────────────────────────────────────────────────────────────
# § 20  TRAINING SYSTEM (Unsloth + DPO + Eval Gate + Auto-Rollback)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 20  TRAINING SYSTEM (Unsloth + DPO + Eval Gate + Auto-Rollback)
# ════════════════════════════════════════════════════════════════════

DATASET_CATALOGUE = {
    "LIMA (quality 1k)":        ("GAIR/lima", "train"),
    "Dolly 15k":                ("databricks/dolly-15k", "train"),
    "Code-Feedback (debug)":    ("m-a-p/Code-Feedback", "train[:2000]"),
    "OpenMathInstruct (math)":  ("nvidia/OpenMathInstruct-1", "train[:2000]"),
    "GSM8K":                    ("gsm8k", "train"),
    "Stack-Smol (200+ langs)":  ("bigcode/the-stack-smol", "train[:1000]"),
    "Open-Orca (reasoning)":    ("Open-Orca/OpenOrca", "train[:2000]"),
    "Wikipedia EN":             ("wikipedia", "train[:3000]"),
    "FineWeb (quality web)":    ("HuggingFaceFW/fineweb", "train[:1000]"),
    "Alpaca 52k":               ("tatsu-lab/alpaca", "train"),
}

EVAL_QS = [
    {"q": "What is the derivative of x^3 + 2x^2 - 5x + 3?", "type": "maths", "expected": ["3x^2", "4x", "-5"]},
    {"q": "Solve: 2x + 5 = 13", "type": "maths", "expected": ["4", "x = 4"]},
    {"q": "Write a Python function to check if a number is prime", "type": "coding", "expected": ["def", "prime", "return"]},
    {"q": "What is Newton's second law of motion?", "type": "science", "expected": ["F", "ma", "force", "mass", "acceleration"]},
    {"q": "What is the capital of Australia?", "type": "knowledge", "expected": ["Canberra"]},
    {"q": "Explain photosynthesis", "type": "science", "expected": ["light", "carbon dioxide", "glucose", "oxygen"]},
    {"q": "What year did World War II end?", "type": "knowledge", "expected": ["1945"]},
    {"q": "What is time complexity of binary search?", "type": "coding", "expected": ["log", "O(log n)"]},
    {"q": "If all roses are flowers and some flowers fade, can we conclude all roses fade?", "type": "reasoning", "expected": ["no", "cannot", "not necessarily"]},
    {"q": "Write a SQL query to select the top 5 highest salaries", "type": "coding", "expected": ["SELECT", "ORDER BY", "DESC", "LIMIT", "5"]},
]

def _score_answer(q_obj, answer) -> float:
    al = answer.lower()
    kws = q_obj.get("expected", [])
    kw_score = sum(1 for k in kws if k.lower() in al) / max(len(kws), 1) if kws else 0.5
    raw = quick_routed(f"Q:{q_obj['q']}\nA:{answer[:400]}\nRate correctness 0.0-1.0. Number only.", max_tokens=5, temp=0.1)
    try:
        model_score = float(re.search(r'0?\.\d+|1\.0|0|1', raw).group())
    except Exception:
        model_score = 0.5
    return round(0.5 * kw_score + 0.5 * model_score, 3)

def run_eval(label="eval", log_fn=None) -> dict:
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n📊 Eval: {label} ({len(EVAL_QS)} questions)")
    scores = []
    type_scores = {}
    for q in EVAL_QS:
        ans = quick(q["q"], max_tokens=300, temp=0.3)
        ans = _strip_think(ans)
        s = _score_answer(q, ans)
        scores.append(s)
        type_scores.setdefault(q["type"], []).append(s)
        _log(f"  {'✅' if s >= 0.7 else '⚠️' if s >= 0.4 else '❌'} {s:.2f} | {q['q'][:50]}")
    overall = sum(scores) / len(scores) if scores else 0
    by_type = {t: round(sum(v) / len(v), 3) for t, v in type_scores.items()}
    _log(f"\n  Overall: {overall:.1%}")
    for t, v in by_type.items():
        _log(f"  {t:12s}: {v:.1%}")
    result = {"label": label, "overall": round(overall, 3), "by_type": by_type, "timestamp": datetime.now().isoformat()}
    db_add_bench(label, result["overall"], by_type)
    return result

def eval_comparison(before, after) -> Tuple[bool, str]:
    b = before.get("overall", 0)
    a = after.get("overall", 0)
    delta = a - b
    regressions = [f"{t}: {before['by_type'].get(t, 0):.1%}→{v:.1%}"
                   for t, v in after.get("by_type", {}).items()
                   if v < before.get("by_type", {}).get(t, 0) - 0.1]
    if a > b:
        return True, f"✅ {b:.1%}→{a:.1%} (+{delta:.1%})" + (f" (minor regressions: {', '.join(regressions)})" if regressions else "")
    return False, f"❌ No improvement: {b:.1%}→{a:.1%} ({delta:+.1%}). Rolling back."

def _fmt(row):
    for qk, ak in [("instruction", "output"), ("prompt", "completion"), ("question", "answer")]:
        if qk in row and ak in row and str(row.get(ak, "")).strip():
            return processor.apply_chat_template(
                [{"role": "user", "content": str(row[qk])[:600]},
                 {"role": "assistant", "content": str(row[ak])[:1200]}], tokenize=False)
    for key in ["text", "content", "passage"]:
        if key in row and len(str(row.get(key, ""))) > 80:
            t = str(row[key])[:1500]
            return processor.apply_chat_template(
                [{"role": "user", "content": f"Explain: {t.split('.')[0].strip()}"},
                 {"role": "assistant", "content": t}], tokenize=False)
    return None

def score_ex(text):
    """Score training example quality (higher = better)."""
    penalties = sum(1 for s in ["as an ai", "as a large language", "in today's fast-paced", "leveraging the power"]
                    if s in text.lower()) * 0.2
    length_bonus = min(len(text.split()) / 500, 0.2)
    return max(0.0, min(1.0, 1.0 - penalties + length_bonus))

def synthetic_data(topic, n=50):
    examples = []
    for _ in range(n):
        raw = quick_routed(f"Generate Q&A about '{topic}'.\nJSON: {{\"question\":\"...\",\"answer\":\"...\"}}", max_tokens=300, temp=0.85)
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            qa = json.loads(m.group()) if m else {}
            if qa.get("question") and qa.get("answer"):
                examples.append({"text": processor.apply_chat_template(
                    [{"role": "user", "content": qa["question"]},
                     {"role": "assistant", "content": qa["answer"]}], tokenize=False)})
        except Exception:
            continue
    return examples

def stream_train(dataset_ids, max_per_ds=500, save_drive=True, use_unsloth=True,
                  use_curriculum=True, quality_threshold=0.4, synthetic_topics=None,
                  run_bench=True, log_fn=None) -> str:
    global processor, model
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    _log(f"\n🚀 Streaming Train — {len(dataset_ids)} datasets\n")
    before_eval = None
    if run_bench:
        _log("\n📊 Running BEFORE eval…")
        before_eval = run_eval("before", log_fn)
        rb = f"{DIRS['checkpts']}/rollback_{int(time.time())}"
        os.makedirs(rb, exist_ok=True)
        for f in Path(MODEL_DIR).glob("*"):
            if f.is_file():
                shutil.copy(f, rb)
        _log(f"  💾 Rollback saved: {rb}")
    examples = []
    for ds_id in dataset_ids:
        hf, split = DATASET_CATALOGUE.get(ds_id, (ds_id, "train[:500]"))
        _log(f"  📡 {hf}")
        try:
            ds = hf_load(hf, split=split, streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if count >= max_per_ds:
                    break
                fmt = _fmt(row)
                if fmt and score_ex(fmt) >= quality_threshold:
                    examples.append({"text": fmt, "quality": score_ex(fmt)})
                    count += 1
            _log(f"     ✓ {count}")
        except Exception as e:
            _log(f"     ⚠️  {e}")
    if synthetic_topics:
        for t in synthetic_topics:
            _log(f"  🧬 Synthetic: {t}")
            s = synthetic_data(t, n=50)
            examples.extend([{**x, "quality": 0.8} for x in s])
    if not examples:
        return "❌ No examples."
    if use_curriculum:
        examples.sort(key=lambda x: (-x["quality"], len(x["text"].split())))
    train_ds = Dataset.from_list([{"text": e["text"]} for e in examples])
    _log(f"  Total: {len(train_ds)} examples")
    with _model_lock:
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel
                um, ut = FastLanguageModel.from_pretrained(model_name=MODEL_DIR, max_seq_length=1024, load_in_4bit=True)
                um = FastLanguageModel.get_peft_model(um, r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA,
                    target_modules=Config.LORA_TARGETS, lora_dropout=Config.LORA_DROPOUT,
                    bias="none", use_gradient_checkpointing=True)
                SFTTrainer(model=um, train_dataset=train_ds,
                    args=SFTConfig(output_dir=DIRS["checkpts"], num_train_epochs=1, per_device_train_batch_size=2,
                        gradient_accumulation_steps=4, warmup_steps=10, learning_rate=2e-4, fp16=True,
                        logging_steps=20, save_strategy="steps", save_steps=100, optim="adamw_8bit",
                        max_seq_length=1024, report_to="none", dataset_text_field="text"),
                    tokenizer=ut).train()
                merged = "/content/merged_tmp"
                um.save_pretrained_merged(merged, ut, save_method="merged_16bit")
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                if save_drive:
                    shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                shutil.rmtree(merged, ignore_errors=True)
                del um
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
                # v13: Update active model info after fine-tune
                _active_model_info.update({"id": Config.MODEL_ID, "vision": True, "dir": MODEL_DIR, "fine_tuned": True})
                VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}
                _log("✅ Unsloth done — fine-tuned model now active")
            except Exception as e:
                _log(f"  ⚠️  Unsloth failed ({e}), TRL fallback…")
                model.train()
                mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                if hasattr(mk, "visual"):
                    for p in mk.visual.parameters():
                        p.requires_grad = False
                lc = LoraConfig(r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA, target_modules=Config.LORA_TARGETS,
                    lora_dropout=Config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
                mk = get_peft_model(mk, lc)
                SFTTrainer(model=mk, train_dataset=train_ds,
                    args=SFTConfig(output_dir=DIRS["checkpts"], num_train_epochs=1, per_device_train_batch_size=1,
                        gradient_accumulation_steps=8, warmup_steps=10, learning_rate=2e-4, fp16=True,
                        logging_steps=20, save_strategy="steps", save_steps=100, optim="paged_adamw_8bit",
                        max_seq_length=1024, report_to="none", dataset_text_field="text"),
                    tokenizer=getattr(processor, "tokenizer", processor)).train()
                merged = "/content/merged_trl"
                mk = mk.merge_and_unload()
                mk.save_pretrained(merged, safe_serialization=True)
                getattr(processor, "tokenizer", processor).save_pretrained(merged)
                shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
                if save_drive:
                    shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
                shutil.rmtree(merged, ignore_errors=True)
                del mk
                torch.cuda.empty_cache()
                processor, model = _load_model(MODEL_DIR)
                model.eval()
                _active_model_info.update({"id": Config.MODEL_ID, "vision": True, "dir": MODEL_DIR, "fine_tuned": True})
                VRAMJuggler.MODELS["qwen"] = {"obj": model, "in_vram": True, "priority": 10}
    if run_bench and before_eval:
        _log("\n📊 Running AFTER eval…")
        after_eval = run_eval("after", log_fn)
        should_keep, explanation = eval_comparison(before_eval, after_eval)
        _log(f"\n{explanation}")
        if not should_keep:
            _log("  🔄 Rolling back…")
            rbs = sorted(Path(DIRS["checkpts"]).glob("rollback_*"))
            if rbs:
                with _model_lock:
                    shutil.copytree(str(rbs[-1]), MODEL_DIR, dirs_exist_ok=True)
                    processor, model = _load_model(MODEL_DIR)
                    model.eval()
                _log("  ✅ Rolled back")
            return f"⚠️ ROLLED BACK: {explanation}"
        for old in sorted(Path(DIRS["checkpts"]).glob("rollback_*"))[:-3]:
            shutil.rmtree(old, ignore_errors=True)
    return f"✅ Training complete. {len(train_ds)} examples."

def dpo_train(log_fn=None) -> str:
    global processor, model
    def _log(m):
        log.info(m)
        if log_fn:
            log_fn(m)
    if len(_dpo_pairs) < 10:
        return f"❌ Need ≥10 pairs. Have {len(_dpo_pairs)}."
    _log(f"\n🎯 DPO on {len(_dpo_pairs)} pairs…")
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in _dpo_pairs])
    with _model_lock:
        model.train()
        mk = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                       lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        mk = get_peft_model(mk, lc)
        DPOTrainer(model=mk, ref_model=None,
            args=DPOConfig(output_dir=DIRS["tmp_train"], num_train_epochs=1, per_device_train_batch_size=1,
                gradient_accumulation_steps=4, learning_rate=5e-5, fp16=True, save_strategy="no",
                report_to="none", beta=0.1),
            train_dataset=ds, tokenizer=getattr(processor, "tokenizer", processor)).train()
        merged = "/content/dpo_m"
        mk = mk.merge_and_unload()
        mk.save_pretrained(merged, safe_serialization=True)
        getattr(processor, "tokenizer", processor).save_pretrained(merged)
        shutil.copytree(merged, MODEL_DIR, dirs_exist_ok=True)
        shutil.copytree(MODEL_DIR, f"{DR}/model", dirs_exist_ok=True)
        for p in [DIRS["tmp_train"], merged]:
            shutil.rmtree(p, ignore_errors=True)
        del mk
        torch.cuda.empty_cache()
        processor, model = _load_model(MODEL_DIR)
        model.eval()
    # v13: Auto-benchmark after DPO (Grok recommendation)
    log.info("Auto-benchmark after DPO…")
    try:
        after = run_eval("after_dpo", log_fn=log_fn)
        db_add_bench("after_dpo", after["overall"], after["by_type"])
        if log_fn:
            log_fn(f"  📊 Post-DPO eval: {after['overall']:.1%}")
    except Exception as e:
        log.warning(f"Post-DPO eval failed: {e}")
    return f"✅ DPO done. {len(_dpo_pairs)} pairs."

log.info("Training system ready")



# ────────────────────────────────────────────────────────────
# § 21  RAG QUALITY AUDIT + SYSTEM STATUS
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 21  RAG QUALITY AUDIT + SYSTEM STATUS
# ════════════════════════════════════════════════════════════════════

def rag_audit(test_queries=None, k=5) -> dict:
    if test_queries is None:
        test_queries = [q["q"] for q in EVAL_QS[:5]]
    results = []
    for q in test_queries:
        t0 = time.time()
        chunks = rag_retrieve(q, k=k)
        latency = time.time() - t0
        results.append({"query": q[:50], "latency_ms": round(latency * 1000),
                        "chunks": len(chunks), "top_score": max((c.get("score", 0) for c in chunks), default=0)})
    avg_lat = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    avg_k = sum(r["chunks"] for r in results) / len(results) if results else 0
    return {"tests": results, "avg_latency_ms": round(avg_lat), "avg_chunks": round(avg_k, 1),
            "total_docs": _col.count(), "kg_nodes": KG.number_of_nodes(), "kg_edges": KG.number_of_edges()}

def system_status() -> str:
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_free = _vram_free()
    juggler_models = ", ".join(
        f"{n}({'VRAM' if e['in_vram'] else 'CPU'})"
        for n, e in VRAMJuggler.MODELS.items()
    ) if VRAMJuggler.MODELS else "none registered"
    return (
        f"## {APP_NAME} v{APP_VERSION} Status\n\n"
        f"**GPU:** {GPU_NAME}\n"
        f"**VRAM:** {vram_used:.1f}GB used / {vram_free:.1f}GB free / {VRAM_GB:.1f}GB total\n"
        f"**VRAM Juggler:** {juggler_models}\n"
        f"**Model:** {Config.MODEL_ID} (4-bit)\n"
        f"**Adapter:** {_active_adapter or 'base'}\n"
        f"**Embedder:** {_EMBEDDER_NAME}\n"
        f"**Context:** {MODEL_MAX_TOKENS:,} tokens\n"
        f"**TTS:** {_tts_engine if Config.ENABLE_VOICE else '🔴 Disabled'}\n\n"
        f"**Features:** "
        f"{'🟢 ImgGen' if Config.ENABLE_IMAGE_GEN else '🔴 ImgGen'} | "
        f"{'🟢 Browser' if Config.ENABLE_BROWSER else '🔴 Browser'} | "
        f"{'🟢 BgLearn' if Config.ENABLE_BG_LEARNING else '🔴 BgLearn'} | "
        f"{'🟢 Reflect' if Config.ENABLE_REFLECTION else '🔴 Reflect'}\n\n"
        f"**RAG:** {_col.count()} chunks (BM25: {len(_bm25_docs)}) | KG: {KG.number_of_nodes()} nodes\n"
        f"**Memory:** {len(_pmem)} prioritized | {len(_saved_convs)} conversation msgs\n"
        f"**DPO:** {len(_dpo_pairs)} pairs | Corrections: {len(_correction_log)}\n"
        f"**Cache:** {len(_response_cache)} responses | {len(_tool_cache)} tool results\n"
        f"**Learning:** {'🟢 Active' if _learn_active.is_set() else '🔴 Stopped'} ({len(_learn_log)} papers)\n"
        f"**Plugins:** {_plugins_loaded} loaded from {DIRS['ws_plugins']}\n"
        f"**Tools:** {len(TOOL_REGISTRY)}\n\n"
        f"**Agent Stats:** {_agent_state.get('total_turns', 0)} turns | "
        f"{_agent_state.get('total_tool_calls', 0)} tool calls | "
        f"{_agent_state.get('corrections_made', 0)} auto-corrections\n"
        f"**Last query:** {_agent_state.get('last_message', '(none)')} "
        f"[{_agent_state.get('rag_hits', 0)} RAG hits]\n"
        f"**Models:** primary={_active_model_info.get('id','?')} "
        f"({'vision' if _active_model_info.get('vision') else 'text-only'}) | "
        f"fast={'ready' if _fast_model is not None else 'not loaded'}\n"
        + _bench_trend_str()
    )

def _bench_trend_str() -> str:
    """Return last 3 benchmark scores as a mini ASCII trend (for status text)."""
    history = db_get_bench_history(n=5)
    if not history:
        return ""
    history.reverse()
    bars = []
    for h in history[-3:]:
        pct = int(h["overall"] * 10)
        bar = "█" * pct + "░" * (10 - pct)
        bars.append(f"{bar} {h['overall']:.0%} ({h['ts'][:10]})")
    return "**Eval history:**\n" + "\n".join(f"  {b}" for b in bars) + "\n"

def _bench_plot() -> Optional[object]:
    """
    v13: Real matplotlib chart of benchmark history (Grok recommendation).
    Returns a Figure object for gr.Plot, or None if no data.
    """
    try:
        history = db_get_bench_history(n=20)
    except Exception:
        return None
    if len(history) < 2:
        return None
    history.reverse()   # Oldest first
    try:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#242424")
        ax.tick_params(colors="#888888")
        ax.spines[:].set_color("#3a3a3a")

        dates = [h["ts"][:10] for h in history]
        overall = [h["overall"] for h in history]

        ax.plot(dates, overall, "o-", color="#10a37f", linewidth=2, markersize=5, label="Overall")

        # Per-type lines if consistent
        all_types = set()
        for h in history:
            all_types.update(h["by_type"].keys())

        colors = {"maths":"#60a5fa","coding":"#f59e0b","science":"#a78bfa",
                  "knowledge":"#34d399","reasoning":"#f87171"}
        for t in sorted(all_types):
            vals = [h["by_type"].get(t) for h in history]
            if any(v is not None for v in vals):
                ys = [v if v is not None else float("nan") for v in vals]
                ax.plot(dates, ys, "--", color=colors.get(t,"#888"), linewidth=1,
                        alpha=0.7, label=t)

        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_title(f"{APP_NAME} Benchmark Trend", color="#f0f0f0", fontsize=10)
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="#f0f0f0", loc="lower right")
        plt.xticks(rotation=25, ha="right", fontsize=7, color="#888")
        plt.tight_layout()
        return fig
    except Exception as e:
        log.debug(f"bench_plot failed: {e}")
        return None

def health_check() -> dict:
    """/health endpoint — returns system health as JSON."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": _active_model_info.get("id", Config.MODEL_ID),
        "model_vision": _active_model_info.get("vision", True),
        "fast_model": Config.FAST_MODEL_ID if _fast_model is not None else None,
        "fallback_chain": [e["id"] for e in Config.MODEL_FALLBACK_CHAIN],
        "circuit_breakers": {
            name: {"failures": cb.fails, "is_open": cb.is_open(), "cooldown_s": cb.cooldown}
            for name, cb in _breakers.items()
        },
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_free_gb": round(_vram_free(), 2),
        "rag_chunks": _col.count(),
        "bm25_docs": len(_bm25_docs),
        "kg_nodes": KG.number_of_nodes(),
        "memory_entries": len(_pmem),
        "dpo_pairs": len(_dpo_pairs),
        "tools": len(TOOL_REGISTRY),
        "bg_learning": _learn_active.is_set(),
        "tts_engine": _tts_engine,
    }

log.info("RAG audit + system status + health ready")


