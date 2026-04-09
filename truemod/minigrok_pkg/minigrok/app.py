"""
minigrok.app
Gradio UI (all tabs), launch logic
"""


# Cross-module imports
import os, json, re, logging, threading, secrets
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

import gradio as gr

from .base import log, APP_NAME, APP_VERSION
from .config import Config, DIRS, WORKSPACE
from .models import (model, MODEL_MAX_TOKENS, VRAMJuggler, quick)
from .generation import _custom_system_prompt, ctx_info
from .memory import (session_primer, save_conversation, export_conversation,
                     _saved_convs, CONV_PATH)
from .agent import agent_stream, _agent_state
from .background import (system_status, rag_audit, start_learning,
                          stop_learning, learning_status, stream_train, dpo_train)
from .rag import rag_add, _col, _bm25_docs
from .voice import synthesise, _voice_profiles, create_voice_profile
from .tools import generate_image, run_code
from .registry import TOOL_REGISTRY, _plugins_loaded


# ────────────────────────────────────────────────────────────
# § 23  GRADIO UI — Complete 8-tab interface (v12)
# ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════
# § 23  GRADIO UI — Complete 8-tab interface (v12)
#        Restores everything from v9 + v10 + v11 improvements
# ════════════════════════════════════════════════════════════════════

import gradio as gr

MOBILE_CSS = """
:root{--bg:#1a1a1a;--panel:#242424;--input:#2e2e2e;
--border:#3a3a3a;--txt:#f0f0f0;--txt2:#888;--acc:#10a37f;
--acc2:#0d8a6c;--red:#ef4444;--r:12px;}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
body,.gradio-container{background:var(--bg)!important;color:var(--txt)!important;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif!important;}
.gradio-container{max-width:800px!important;margin:0 auto!important;padding:0!important;}
#chat-input textarea{background:var(--input)!important;color:var(--txt)!important;border:1px solid var(--border)!important;border-radius:24px!important;padding:12px 16px!important;font-size:16px!important;line-height:1.5!important;resize:none!important;box-shadow:0 2px 6px rgba(0,0,0,0.2)!important;}
#chat-input textarea:focus{border-color:var(--acc)!important;outline:none!important;}
.think-box{background:rgba(255,255,255,0.03)!important;border-left:2px solid var(--acc)!important;color:var(--txt2)!important;font-style:italic!important;font-size:14px!important;margin:8px 0!important;}
.logbox{background:#000!important;color:#0f0!important;font-family:monospace!important;font-size:12px!important;border:1px solid #333!important;border-radius:8px!important;}
.chips{display:flex;flex-wrap:wrap;gap:8px;padding:12px;overflow-x:auto;scrollbar-width:none;}
.chips::-webkit-scrollbar{display:none;}
.chip{background:var(--panel);color:var(--txt);border:1px solid var(--border);border-radius:18px;padding:6px 14px;font-size:14px;white-space:nowrap;cursor:pointer;transition:all 0.2s;}
.chip:hover{background:var(--acc);border-color:var(--acc);transform:translateY(-1px);}
.message.user{background:var(--input)!important;border-radius:18px 18px 4px 18px!important;padding:12px 16px!important;margin-bottom:12px!important;border:1px solid var(--border)!important;}
.message.bot{background:transparent!important;padding:12px 0!important;margin-bottom:12px!important;}
"""

def create_ui():
    with gr.Blocks(css=MOBILE_CSS, theme=gr.themes.Default(), title=f"{APP_NAME} v{APP_VERSION}") as app:
        file_st = gr.State(None)
        img_st  = gr.State(None)
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="", height=440, type="messages", autofocus=True)
            ctx_bar = gr.Textbox(label="", lines=1, interactive=False, value=f"0/{MODEL_MAX_TOKENS:,} tokens (0%)")
            subject_dd = gr.Dropdown(choices=["auto","maths","coding","science","english","humanities"], value="auto", label="📚 Subject", container=False)

            gr.HTML("""<div class="chips">
              <button class="chip" onclick="cht('💻 Write & run: ')">💻 Code</button>
              <button class="chip" onclick="cht('📝 Solve step by step: ')">📝 Homework</button>
              <button class="chip" onclick="cht('🔍 Research: ')">🔍 Research</button>
              <button class="chip" onclick="cht('🌐 Browse and find: ')">🌐 Browse</button>
              <button class="chip" onclick="cht('📊 Deep research: ')">📊 Deep Dive</button>
              <button class="chip" onclick="cht('📚 Study notes on: ')">📚 Notes</button>
              <button class="chip" onclick="cht('🔗 Plan and execute: ')">🔗 Chain</button>
              <button class="chip" onclick="cht('🎬 Analyse video: ')">🎬 Video</button>
              <button class="chip" onclick="cht('🃏 Flashcards on: ')">🃏 Cards</button>
              <button class="chip" onclick="cht('✍️ Essay on: ')">✍️ Essay</button>
              <button class="chip" onclick="cht('🧮 Calculate: ')">🧮 Math</button>
              <button class="chip" onclick="cht('🌳 Think hard (ToT): ')">🌳 ToT</button>
            </div>
            <script>function cht(t){const tb=document.querySelector("textarea");if(tb){tb.value=t;tb.dispatchEvent(new Event("input",{bubbles:true}));tb.focus();}}</script>""")

            with gr.Row():
                agent_chk = gr.Checkbox(label="🔬 Agent", value=True, scale=1)
                speak_chk = gr.Checkbox(label="🔊 Speak", value=False, scale=1)
                reason_dd = gr.Dropdown(["standard","tot","reflect"], value="standard", label="Mode", scale=2)
                v_prof_dd = gr.Dropdown(["default"]+list(_voice_profiles.keys()), value="default", label="Voice", scale=2)
                clr_btn   = gr.Button("🗑", variant="stop", scale=1, min_width=50)

            audio_out   = gr.Audio(label="🔊", autoplay=True)
            emotion_box = gr.Textbox(label="", lines=1, interactive=False, placeholder="emotion…")

            with gr.Row():
                thu_btn = gr.Button("👍", size="sm")
                thd_btn = gr.Button("👎", size="sm")
                dpo_ct  = gr.Textbox(label="DPO pairs", lines=1, interactive=False, value=str(len(_dpo_pairs)), scale=2)
                export_btn = gr.Button("💾 Export", size="sm", scale=1)

            with gr.Accordion("🧠 Thinking", open=False):
                think_box = gr.Textbox(label="", lines=5, interactive=False, elem_classes=["think-box"])
            with gr.Accordion("🔧 Tools", open=True):
                tool_box  = gr.Textbox(label="", lines=4, interactive=False, elem_classes=["logbox"])

            with gr.Group():
                with gr.Row():
                    attach_btn = gr.Button("➕", min_width=45, scale=0)
                    chat_in  = gr.Textbox(placeholder=f"Message {APP_NAME}…", label="", lines=1, max_lines=6, scale=10, container=False)
                    send_btn = gr.Button("↑", variant="primary", scale=0, min_width=50)
                
                with gr.Row(visible=False) as attach_menu:
                    file_in  = gr.File(label="📎 File→RAG", scale=1, file_count="single")
                    img_in   = gr.Image(label="🖼 VL", type="filepath", scale=1, height=64)
                    audio_in = gr.Audio(label="🎤 Voice", type="filepath", scale=1, sources=["microphone","upload"])
            
            menu_state = gr.State(False)
            def _toggle_menu(v): return not v, gr.update(visible=not v)
            attach_btn.click(_toggle_menu, menu_state, [menu_state, attach_menu])

            with gr.Accordion("⚙️ Settings", open=False):
                with gr.Row():
                    temp_sl = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    maxt_sl = gr.Slider(128, 2048, value=1024, step=128, label="Max tokens")
                sys_p   = gr.Textbox(label="System Prompt", value=SYSTEM_PROMPT, lines=3, max_lines=5)
                status_box = gr.Textbox(label="System Status", lines=10, interactive=False, elem_classes=["logbox"], value=system_status())
                with gr.Row():
                    ref_btn  = gr.Button("🔄 Refresh", size="sm")
                    free_btn = gr.Button("🧹 Free VRAM", size="sm")
                    hlth_btn = gr.Button("🏥 Health", size="sm")

            # ── Chat handlers ───────────────────────────────────
            def _send(msg, hist, fp, ip, agent, spk, rmode, vp, sub, temp, maxt, sysp):
                if not msg.strip() and not fp and not ip:
                    yield hist, hist, ctx_info(hist), None, "", "", "", system_status()
                    return
                gen = agent_stream(msg, hist, image_path=ip, log_fn=None)
                lh = hist; th = tl = em = ""; au = None
                for response in gen:
                    lh = list(hist) + [{"role":"user","content":msg},{"role":"assistant","content":response}]
                    em = detect_emotion(response[:200])
                    yield lh, lh, ctx_info(lh), au, th, tl, em, system_status()
                    if spk and Config.ENABLE_VOICE:
                        clip = _voice_profiles.get(vp, {}).get("clip")
                        au = synthesise(response[:500], voice_clip=clip)
                yield lh, lh, ctx_info(lh), au, th, tl, em, system_status()

            # Input handlers
            chat_in.submit(_send, inputs=[chat_in, chatbot, file_st, img_st, agent_chk, speak_chk, reason_dd, v_prof_dd, subject_dd, temp_sl, maxt_sl, sys_p], outputs=[chatbot, chatbot, ctx_bar, audio_out, think_box, tool_box, emotion_box, status_box])
            send_btn.click(_send, inputs=[chat_in, chatbot, file_st, img_st, agent_chk, speak_chk, reason_dd, v_prof_dd, subject_dd, temp_sl, maxt_sl, sys_p], outputs=[chatbot, chatbot, ctx_bar, audio_out, think_box, tool_box, emotion_box, status_box])
            
            if Config.ENABLE_VOICE:
                audio_in.change(lambda a: _get_whisper("base").transcribe(a, language="en")["text"].strip() if a else "", audio_in, chat_in)
            file_in.change(lambda f: (process_upload(f.name) if f else "No file", f.name if f else None), file_in, [status_box, file_st])
            img_in.change(lambda p: p, img_in, img_st)
            
            clr_btn.click(lambda: ([], [], "0/32,768 tokens (0%)"), None, [chatbot, chatbot, ctx_bar])
            thu_btn.click(lambda h: (db_add_dpo(h[-2]["content"], h[-1]["content"], ""), str(len(db_get_dpo()))), [chatbot], [dpo_ct, dpo_ct])
            thd_btn.click(lambda h: (db_add_dpo(h[-2]["content"], "", h[-1]["content"]), str(len(db_get_dpo()))), [chatbot], [dpo_ct, dpo_ct])
            export_btn.click(lambda h: export_conversation(h), [chatbot], status_box)
            ref_btn.click(system_status, None, status_box)
            free_btn.click(lambda: (torch.cuda.empty_cache(), "VRAM Cleared"), None, status_box)
            hlth_btn.click(lambda: json.dumps(health_check(), indent=2), None, status_box)

        # ══════════════════════════════════════════════════════════
        # TAB 2: 🌐 BROWSER
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🌐 Browser"):
            if not Config.ENABLE_BROWSER:
                gr.Markdown("### Browser disabled\nSet `Config.ENABLE_BROWSER = True` and restart.")
            else:
                gr.Markdown("### Autonomous VL Browser Agent\nGives the AI a goal — it runs until done and verifies completion.")
                with gr.Row():
                    ba_goal  = gr.Textbox(label="Goal", lines=3, scale=4,
                                           placeholder="Find the top 5 Python ML libraries, compare GitHub stars, and write a summary")
                    ba_steps = gr.Slider(5, 30, value=15, step=1, label="Max steps", scale=1)
                    ba_start = gr.Textbox(label="Start URL (optional)", lines=1, scale=2)
                ba_btn    = gr.Button("🚀 Run Agent", variant="primary")
                ba_result = gr.Textbox(label="Result", lines=10, interactive=False)
                ba_log    = gr.Textbox(label="Step log", lines=8, interactive=False, elem_classes=["logbox"])
                ba_ss     = gr.Gallery(label="Screenshots", columns=3, height=220)

                def _do_ba(goal, steps, start):
                    if not goal.strip(): return "Enter a goal.", "", []
                    log_lines = []
                    r = browser_agent(goal, max_steps=int(steps), start_url=start.strip(),
                                       log_fn=log_lines.append)
                    ver = "✅ Verified" if r.get("verified") else "⚠️ Unverified"
                    return (
                        f"{ver}\n\n{r['result']}",
                        "\n".join(log_lines),
                        [(p,"") for p in r.get("screenshots",[])[-6:]])
                ba_btn.click(_do_ba, [ba_goal,ba_steps,ba_start], [ba_result,ba_log,ba_ss])

                gr.Markdown("""---
### Manual Control""")
                with gr.Row():
                    br_url = gr.Textbox(label="URL", scale=3)
                    br_act = gr.Dropdown(
                        ["navigate","screenshot","get_text","click","type","scroll","run_js","close"],
                        value="navigate", label="Action", scale=2)
                    br_btn = gr.Button("▶", variant="primary", min_width=60)
                with gr.Row():
                    br_sel = gr.Textbox(label="Selector", scale=2)
                    br_inp = gr.Textbox(label="Text/JS", scale=3)
                br_out = gr.Textbox(label="Result", lines=8, interactive=False)
                br_img = gr.Image(label="Screenshot")
                def _br(url,act,sel,inp):
                    r = browser_action_single(act, url=url, selector=sel, text_input=inp, js_code=inp)
                    if r.startswith("__IMAGE__"): return "Screenshot taken.", r.replace("__IMAGE__","")
                    return r, None
                br_btn.click(_br, [br_url,br_act,br_sel,br_inp], [br_out,br_img])

        # ══════════════════════════════════════════════════════════
        # TAB 3: 📝 HOMEWORK
        # ══════════════════════════════════════════════════════════
        with gr.Tab("📝 Homework"):
            with gr.Tabs():
                with gr.Tab("Solve"):
                    hw_q   = gr.Textbox(label="Question", lines=5,
                                         placeholder="Solve 2x² + 5x - 3 = 0 showing all working…")
                    hw_sub = gr.Radio(["auto","maths","coding","science","english","humanities"],
                                       value="auto", label="Subject")
                    hw_btn = gr.Button("📝 Solve", variant="primary")
                    hw_out = gr.Textbox(label="Solution", lines=15, interactive=False)
                    hw_btn.click(solve_homework, [hw_q,hw_sub], hw_out)

                with gr.Tab("Study Notes"):
                    sn_topic = gr.Textbox(label="Topic",
                                           placeholder="Quadratic equations / Photosynthesis / Cold War")
                    sn_sub   = gr.Dropdown(["general","maths","coding","science","english","humanities"],
                                            value="general", label="Subject")
                    sn_btn   = gr.Button("📚 Generate Notes", variant="primary")
                    sn_out   = gr.Textbox(label="Notes", lines=20, interactive=False)
                    sn_btn.click(study_notes, [sn_topic,sn_sub], sn_out)

                with gr.Tab("Flashcards"):
                    fc_topic = gr.Textbox(label="Topic")
                    fc_n     = gr.Slider(5, 30, value=10, step=1, label="Number of cards")
                    fc_btn   = gr.Button("🃏 Generate", variant="primary")
                    fc_out   = gr.Textbox(label="Flashcards", lines=18, interactive=False)
                    fc_btn.click(lambda t,n: generate_flashcards(t,int(n)), [fc_topic,fc_n], fc_out)

                with gr.Tab("Essay"):
                    es_topic = gr.Textbox(label="Essay topic / title", lines=2)
                    with gr.Row():
                        es_type = gr.Dropdown(["analytical","argumentative","narrative","descriptive"],
                                               value="analytical", label="Type")
                        es_wc   = gr.Slider(250, 2000, value=500, step=50, label="Word count")
                    es_btn = gr.Button("✍️ Write Essay", variant="primary")
                    es_out = gr.Textbox(label="Essay", lines=20, interactive=False)
                    es_btn.click(essay_help, [es_topic,es_type,es_wc], es_out)

        # ══════════════════════════════════════════════════════════
        # TAB 4: 💻 CODE
        # ══════════════════════════════════════════════════════════
        with gr.Tab("💻 Code"):
            gr.Markdown("### Persistent Kernel — variables survive between runs")
            with gr.Tabs():
                with gr.Tab("Editor"):
                    with gr.Row():
                        co_la = gr.Dropdown(["python","bash","sql","javascript"],
                                             value="python", label="Language", scale=2)
                        co_t  = gr.Slider(5, 120, value=30, step=5, label="Timeout (s)", scale=1)
                    co_in = gr.Code(language="python", lines=14,
                                     value="x = [1, 2, 3, 4, 5]\nprint('Sum:', sum(x))\nprint('Mean:', sum(x)/len(x))")
                    with gr.Row():
                        co_run = gr.Button("▶ Run",    variant="primary")
                        co_fix = gr.Button("🐛 Auto-fix")
                        co_rev = gr.Button("📄 Review")
                        co_tst = gr.Button("🧪 Tests")
                        co_doc = gr.Button("📖 Docs")
                    co_desc = gr.Textbox(label="Describe code to generate", lines=2)
                    co_gen  = gr.Button("✏️ Generate", variant="secondary")
                    co_out  = gr.Textbox(label="Output", lines=9, interactive=False)
                    co_info = gr.Textbox(label="Info / Review", lines=5, interactive=False)

                    co_run.click(lambda l,c,t: (run_code(l,c,int(t)), ""), [co_la,co_in,co_t], [co_out,co_info])
                    co_fix.click(lambda l,c: (auto_fix_loop(l,c), ""), [co_la,co_in], [co_out,co_info])
                    co_rev.click(lambda l,c: ("", code_review(c,l)), [co_la,co_in], [co_out,co_info])
                    co_tst.click(lambda l,c: (generate_tests(c,l), ""), [co_la,co_in], [co_out,co_info])
                    co_doc.click(lambda l,c: (generate_docs(c,l), ""), [co_la,co_in], [co_out,co_info])
                    co_gen.click(
                        lambda d,l: (re.sub(r"```\w*\n?|```", "",
                            quick(f"Write {l} code for: {d}\nReturn ONLY code.",
                                  system=f"Expert {l} programmer.", max_tokens=600, temp=0.2)).strip(), ""),
                        [co_desc,co_la], [co_in,co_info])

                with gr.Tab("Project Scanner"):
                    ps_path = gr.Textbox(label="Folder path", placeholder="projects/my_app")
                    ps_btn  = gr.Button("🔍 Scan", variant="primary")
                    ps_out  = gr.Textbox(label="Project overview", lines=16, interactive=False)
                    ps_btn.click(
                        lambda p: scan_project(os.path.join(WORKSPACE, p) if not p.startswith("/") else p),
                        ps_path, ps_out)

        # ══════════════════════════════════════════════════════════
        # TAB 5: 📁 WORKSPACE
        # ══════════════════════════════════════════════════════════
        with gr.Tab("📁 Workspace"):
            gr.Markdown(f"### Persistent Workspace\n`{WORKSPACE}` — survives session restarts.")
            with gr.Tabs():
                with gr.Tab("Browse"):
                    with gr.Row():
                        ws_dir = gr.Textbox(label="Subfolder (empty=all)", lines=1)
                        ws_lb  = gr.Button("📂 List", variant="primary")
                    ws_list_out = gr.Textbox(label="", lines=14, interactive=False)
                    ws_lb.click(ws_list, ws_dir, ws_list_out)

                with gr.Tab("Read / Edit"):
                    ws_path = gr.Textbox(label="Relative path",
                                          placeholder="projects/main.py", lines=1)
                    with gr.Row():
                        ws_rb = gr.Button("📖 Read", variant="primary")
                        ws_sb = gr.Button("💾 Save", variant="secondary")
                    ws_content = gr.Code(label="Content", lines=18)
                    ws_msg     = gr.Textbox(label="", lines=1, interactive=False)
                    ws_rb.click(ws_read, ws_path, ws_content)
                    ws_sb.click(lambda p,c: ws_write(p,c), [ws_path,ws_content], ws_msg)

                with gr.Tab("Notes"):
                    notes_out = gr.Textbox(
                        label="workspace/memory.md", lines=14, interactive=False,
                        value=WS_NOTES.read_text() if WS_NOTES.exists() else "")
                    with gr.Row():
                        note_in  = gr.Textbox(label="Add note", lines=3, scale=4)
                        note_btn = gr.Button("➕ Add", variant="primary", scale=1)
                    def _add_note(n):
                        ws_note(n)
                        return WS_NOTES.read_text() if WS_NOTES.exists() else ""
                    note_btn.click(_add_note, note_in, notes_out)

        # ══════════════════════════════════════════════════════════
        # TAB 6: 🔬 RESEARCH
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🔬 Research"):
            with gr.Tabs():
                with gr.Tab("Deep Research"):
                    with gr.Row():
                        dr_t   = gr.Textbox(label="Topic", scale=4)
                        dr_d   = gr.Slider(1, 5, value=3, step=1, label="Depth")
                        dr_btn = gr.Button("🔬 Research", variant="primary")
                    dr_out = gr.Textbox(label="Report", lines=18, interactive=False)
                    dr_btn.click(_t_deep_research, [dr_t,dr_d], dr_out)

                with gr.Tab("Read Paper"):
                    pp_up  = gr.File(label="Upload PDF", file_types=[".pdf"])
                    pp_btn = gr.Button("📄 Read Paper", variant="primary")
                    pp_out = gr.Textbox(label="Analysis", lines=16, interactive=False)
                    def _rp(f):
                        if not f: return "Upload a PDF."
                        r = read_paper(f.name)
                        return (f"**{r.get('title','')}**\n\n"
                                f"**Abstract:**\n{r.get('abstract','')}\n\n"
                                f"**Methodology:**\n{r.get('methodology','')}\n\n"
                                f"**Results:**\n{r.get('results','')}")
                    pp_btn.click(_rp, pp_up, pp_out)

                with gr.Tab("Knowledge Base"):
                    with gr.Row():
                        kb_wi = gr.Textbox(label="Wikipedia query", scale=2)
                        kb_wb = gr.Button("📖", size="sm")
                        kb_ax = gr.Textbox(label="arXiv query", scale=2)
                        kb_ab = gr.Button("📄", size="sm")
                        kb_ur = gr.Textbox(label="URL to crawl", scale=3)
                        kb_ub = gr.Button("🌐", size="sm")
                    kb_tx = gr.Textbox(label="Paste text to add", lines=5)
                    kb_ad = gr.Button("➕ Add to Knowledge Base", variant="primary")
                    kb_ou = gr.Textbox(label="Result", lines=5, interactive=False)
                    kb_wb.click(_t_wikipedia, kb_wi, kb_ou)
                    kb_ab.click(_t_arxiv, kb_ax, kb_ou)
                    kb_ub.click(_t_crawl, kb_ur, kb_ou)
                    kb_ad.click(
                        lambda t: f"✅ {rag_add(t,'manual','user')} chunks added." if t.strip() else "Enter text.",
                        kb_tx, kb_ou)

                with gr.Tab("RAG Audit"):
                    ra_k   = gr.Slider(3, 10, value=5, step=1, label="k")
                    ra_btn = gr.Button("🔍 Run Audit", variant="primary")
                    ra_out = gr.Textbox(label="Results", lines=16, interactive=False, elem_classes=["logbox"])
                    ra_btn.click(lambda k: str(rag_audit(k=int(k))), ra_k, ra_out)

        # ══════════════════════════════════════════════════════════
        # TAB 7: 🎙️ VOICE
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🎙️ Voice"):
            if not Config.ENABLE_VOICE:
                gr.Markdown("### Voice disabled\nSet `Config.ENABLE_VOICE = True` and restart.")
            else:
                with gr.Tabs():
                    with gr.Tab("Live"):
                        gr.Markdown(f"**Speak → {APP_NAME} thinks → speaks back** (engine: {_tts_engine})")
                        with gr.Row():
                            lv_au  = gr.Audio(label="🎤 Speak", type="filepath",
                                               sources=["microphone","upload"], scale=3)
                            lv_wsz = gr.Dropdown(["tiny","base","small"], value="base",
                                                   label="Whisper size", scale=1)
                            lv_vp  = gr.Dropdown(["default"]+list(_voice_profiles.keys()),
                                                   value="default", label="Voice", scale=1)
                            lv_btn = gr.Button("▶ Reply", variant="primary", scale=1)
                        lv_tr  = gr.Textbox(label="You said", lines=2, interactive=False)
                        lv_rep = gr.Textbox(label="Reply", lines=4, interactive=False)
                        lv_out = gr.Audio(label="🔊", autoplay=True)
                        lv_btn.click(lambda a,wsz,vp: voice_turn(a,wsz,vp),
                                      [lv_au,lv_wsz,lv_vp], [lv_tr,lv_rep,lv_out])

                    with gr.Tab("TTS Test"):
                        tts_t   = gr.Textbox(label="Text to speak", lines=5)
                        with gr.Row():
                            tts_e = gr.Dropdown(["auto","neutral","happy","sad","shocked","angry","excited"],
                                                  value="auto", label="Emotion")
                            tts_v = gr.Dropdown(["default"]+list(_voice_profiles.keys()),
                                                  value="default", label="Voice")
                        tts_btn = gr.Button("🔊 Speak", variant="primary")
                        tts_out = gr.Audio(label="", autoplay=True)
                        tts_btn.click(
                            lambda t,e,v: synthesise(t, e, _voice_profiles.get(v,{}).get("clip")),
                            [tts_t,tts_e,tts_v], tts_out)

                    with gr.Tab("Voice Profiles"):
                        gr.Markdown(f"Clone any voice from a 3–30s audio clip. Engine: **{_tts_engine}**")
                        with gr.Row():
                            vp_n = gr.Textbox(label="Profile name", scale=2)
                            vp_d = gr.Textbox(label="Description", scale=3)
                        vp_c  = gr.Audio(label="Reference clip", type="filepath", sources=["upload"])
                        vp_btn = gr.Button("✅ Create Profile", variant="primary")
                        vp_res = gr.Textbox(label="", lines=2, interactive=False)
                        vp_lst = gr.Textbox(
                            label="Existing profiles", lines=5, interactive=False,
                            value="\n".join(f"• {n}: {p['desc']}" for n,p in _voice_profiles.items()) or "None.")
                        def _cvp(n,d,c):
                            if not n.strip(): return "Enter name.", "—"
                            if not c: return "Upload clip.", "—"
                            r = create_voice_profile(n.strip(), d, c)
                            lst = "\n".join(f"• {n}: {p['desc']}" for n,p in _voice_profiles.items())
                            return r, lst
                        vp_btn.click(_cvp, [vp_n,vp_d,vp_c], [vp_res,vp_lst])

        # ══════════════════════════════════════════════════════════
        # TAB 8: 📊 TRAIN / ALIGN
        # ══════════════════════════════════════════════════════════
        with gr.Tab("📊 Train"):
            gr.Markdown("""### Streaming Trainer — 0 bytes stored to disk during training
"
                        "Streams → quality filter → curriculum sort → Unsloth LoRA → eval gate → auto-rollback if worse.""")
            with gr.Row():
                with gr.Column(scale=2):
                    ds_chk  = gr.CheckboxGroup(choices=list(DATASET_CATALOGUE.keys()),
                                                label="Datasets")
                    ds_cust = gr.Textbox(label="Custom HF dataset IDs (one per line)", lines=4)
                    ds_synt = gr.Textbox(label="Synthetic data topics (one per line)", lines=3)
                    with gr.Row():
                        ds_mx = gr.Slider(100, 5000, value=500, step=100, label="Max per dataset")
                        ds_qt = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Quality threshold")
                    with gr.Row():
                        ds_unsl = gr.Checkbox(label="🦥 Unsloth", value=True)
                        ds_drv  = gr.Checkbox(label="💾 Save to Drive", value=True)
                        ds_cur  = gr.Checkbox(label="📚 Curriculum", value=True)
                        ds_bch  = gr.Checkbox(label="📊 Eval gate", value=True)
                    with gr.Row():
                        ds_btn  = gr.Button("🚀 Train", variant="primary")
                        dpo_btn = gr.Button("🎯 DPO", variant="secondary")

                    gr.Markdown("""---
**Continual Learning**""")
                    bg_topic  = gr.Textbox(label="Add learning topic", lines=1)
                    bg_add    = gr.Button("➕ Add Topic", size="sm")
                    with gr.Row():
                        bg_start = gr.Button("🌙 Start BG Learning", variant="primary")
                        bg_stop  = gr.Button("⏹ Stop", variant="stop")
                        bg_ref   = gr.Button("🔄 Status", size="sm")
                    bg_status = gr.Textbox(label="", lines=5, interactive=False,
                                            value=learning_status())

                with gr.Column(scale=3):
                    ds_log = gr.Textbox(label="Training log", lines=22, interactive=False,
                                         elem_classes=["logbox"])
                    ds_res = gr.Textbox(label="Result", lines=2, interactive=False)

            # Training runs in background thread so UI doesn't freeze
            def _do_train(sel, cust, synt, mx, qt, unsl, drv, cur, bch):
                ids = list(sel or []) + [l.strip() for l in (cust or "").split("\n") if l.strip()]
                tops = [l.strip() for l in (synt or "").split("\n") if l.strip()]
                if not ids and not tops:
                    return "Select at least one dataset or topic.", "❌"
                log_lines = []
                result = stream_train(
                    ids, max_per_ds=int(mx), save_drive=drv, use_unsloth=unsl,
                    use_curriculum=cur, quality_threshold=float(qt),
                    synthetic_topics=tops or None, run_bench=bch,
                    log_fn=log_lines.append)
                return "\n".join(log_lines), result

            def _do_dpo():
                log_lines = []
                r = dpo_train(log_fn=log_lines.append)
                return "\n".join(log_lines), r

            ds_btn.click(_do_train,
                [ds_chk,ds_cust,ds_synt,ds_mx,ds_qt,ds_unsl,ds_drv,ds_cur,ds_bch],
                [ds_log,ds_res])
            dpo_btn.click(_do_dpo, [], [ds_log,ds_res])
            bg_add.click(add_learning_topic, bg_topic, bg_status)
            bg_start.click(start_learning, None, bg_status)
            bg_stop.click(stop_learning, None, bg_status)
            bg_ref.click(learning_status, None, bg_status)

        # ══════════════════════════════════════════════════════════
        # TAB 9: 🎨 IMAGE STUDIO
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🎨 Studio"):
            if not Config.ENABLE_IMAGE_GEN:
                gr.Markdown("### Image generation disabled\nSet `Config.ENABLE_IMAGE_GEN = True` and restart.")
            else:
                gr.Markdown("""### Local Image Generation (Stable Diffusion Turbo)
Free, local, no API key needed.""")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_prompt = gr.Textbox(label="Prompt", lines=3)
                        img_neg    = gr.Textbox(label="Negative", value="blurry, low quality, distorted")
                        img_steps  = gr.Slider(1, 10, value=4, step=1, label="Steps")
                        img_btn    = gr.Button("🎨 Generate", variant="primary")
                    with gr.Column(scale=2):
                        img_output = gr.Image(label="Output", interactive=False)
                def _gen_img(p, n, s):
                    r = generate_image(p, n, s)
                    return r.replace("__IMAGE__", "") if r.startswith("__IMAGE__") else None
                img_btn.click(_gen_img, [img_prompt,img_neg,img_steps], img_output)

        # ══════════════════════════════════════════════════════════
        # TAB 10: 📊 STATUS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("📊 Status"):
            with gr.Row():
                s_ref  = gr.Button("🔄 Refresh", variant="primary")
                s_vram = gr.Button("🧹 Free VRAM")
                s_fast = gr.Button("⚡ Load Fast Model")
                s_rag  = gr.Button("🗑 Clear RAG", variant="stop")
                s_mem  = gr.Button("🗑 Clear Memory", variant="stop")
                s_hlth = gr.Button("🏥 Health Check")
            s_box  = gr.Textbox(label="System Status", lines=22, interactive=False,
                                 elem_classes=["logbox"], value=system_status())
            s_aud  = gr.Textbox(label="Recent audit events", lines=6, interactive=False)
            s_bench_plot = gr.Plot(label="📊 Eval Score History (auto-updates after DPO/training)")
            s_hlth_out = gr.Textbox(label="Health check", lines=8, interactive=False)

            def _sr():
                al = []
                if Path(AUDIT_LOG).exists():
                    try:
                        al = Path(AUDIT_LOG).read_text().strip().split("\n")[-10:]
                    except Exception:
                        al = []
                try:
                    bp = _bench_plot()
                except Exception:
                    bp = None
                return system_status(), "\n".join(al), bp

            def _clear_rag():
                global _bm25_docs, _bm25_ids, _bm25_index, _seen
                _chroma.delete_collection("minigrok_rag")
                _chroma.get_or_create_collection("minigrok_rag", metadata={"hnsw:space":"cosine"})
                _bm25_docs = []; _bm25_ids = []; _bm25_index = None; _seen = set()
                return _sr()

            def _load_fast():
                ok = _try_load_fast_model()
                return _sr()[0] + f"\nFast model: {'✅ loaded' if ok else '❌ not available'}", _sr()[1]

            s_ref.click(_sr, None, [s_box,s_aud,s_bench_plot])
            s_vram.click(lambda: (torch.cuda.empty_cache(), *_sr()), None, [s_box,s_aud,s_bench_plot])
            s_fast.click(_load_fast, None, [s_box,s_aud])
            s_rag.click(_clear_rag, None, [s_box,s_aud,s_bench_plot])
            s_mem.click(lambda: (_pmem.clear(), *_sr()), None, [s_box,s_aud,s_bench_plot])
            s_hlth.click(lambda: str(health_check()), None, s_hlth_out)
            demo.load(_sr, None, [s_box,s_aud,s_bench_plot])

        # ══════════════════════════════════════════════════════════
        # TAB 11: 🧩 PLUGINS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🧩 Plugins"):
            gr.Markdown(f"## Registered Tools & Plugins\nDrop `.py` files into `{DIRS['ws_plugins']}/`")
            tool_df = gr.Dataframe(
                headers=["Tool","Description","Parameters"],
                value=[[k,v["desc"],str(v.get("params",{}))] for k,v in TOOL_REGISTRY.items()],
                interactive=False)
            with gr.Row():
                pl_path   = gr.Textbox(label="Plugin folder", value=DIRS["ws_plugins"], interactive=False)
                pl_reload = gr.Button("🔄 Reload Plugins")
                pl_out    = gr.Textbox(label="Status", lines=2)
            def _reload_pl():
                n = _load_plugins()
                return (f"Reloaded. {len(TOOL_REGISTRY)} total tools ({n} from plugins)",
                        [[k,v["desc"],str(v.get("params",{}))] for k,v in TOOL_REGISTRY.items()])
            pl_reload.click(_reload_pl, None, [pl_out,tool_df])
    return app

def _auto_save_worker():
    """Background thread: persist conversations and memory every 5 minutes using atomic writes."""
    while True:
        try:
            time.sleep(300)
            if _saved_convs:
                _atomic_write(CONV_PATH, _saved_convs)
            if _pmem:
                _atomic_write(PMEM_PATH, _pmem)
            if _dpo_pairs:
                _atomic_write(DPO_PATH, _dpo_pairs)
            log.debug("Auto-save complete (atomic)")
        except Exception as e:
            log.warning(f"Auto-save error: {e}")



def find_free_port(start: int = 7860) -> int:
    """Find an available TCP port starting from 'start'. Prevents OSError on relaunch."""
    for p in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', p))
                return p
        except OSError:
            continue
    # Let OS pick an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _launch(share=True, port=7860):
    """Start background workers and launch the Gradio UI with optional auth."""
    try:
        from huggingface_hub import login
        if "HF_TOKEN" in os.environ:
            login(os.environ["HF_TOKEN"])
            log.info("HF Token authenticated")
    except Exception:
        pass

    # Auto-generate credentials when sharing without explicit auth
    auth = Config.GRADIO_AUTH
    if share and auth is None:
        auto_user = "minigrok"
        auto_pwd  = secrets.token_urlsafe(12)
        auth = (auto_user, auto_pwd)
        print(f"\n  🔐 Auto-generated Gradio credentials (share=True):")
        print(f"     Username: {auto_user}")
        print(f"     Password: {auto_pwd}")
        print(f"     (Set Config.GRADIO_AUTH=(user,pass) to use fixed credentials)\n")

    print(f"\n{'='*60}")
    print(f"  🚀  {APP_NAME} v{APP_VERSION}")
    print(f"  GPU:       {GPU_NAME}  ({VRAM_GB:.0f}GB VRAM)")
    print(f"  Model:     {Config.MODEL_ID}")
    print(f"  Embedder:  {_EMBEDDER_NAME}")
    print(f"  TTS:       {_tts_engine if Config.ENABLE_VOICE else 'disabled'}")
    print(f"  Tools:     {len(TOOL_REGISTRY)}")
    print(f"  Hybrid RAG:{len(_bm25_docs)} BM25 | {_col.count()} semantic chunks")
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Auth:      {'✅ enabled' if auth else '❌ none (local only)'}")
    print(f"{'='*60}\n")

    threading.Thread(target=_auto_save_worker, daemon=True).start()
    actual_port = find_free_port(port)
    if actual_port != port:
        log.warning(f"Port {port} in use — using {actual_port}")
        print(f"  ⚠️  Port {port} busy → using {actual_port}")

    app = create_ui()
    app.queue(max_size=20).launch(
        share=share,
        server_port=actual_port,
        show_error=True,
        auth=auth,
        allowed_paths=[WORKSPACE, DIRS["outputs"]],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--no-share", action="store_false", dest="share")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--auth", type=str, default=None, help="user:password")
    args, _ = parser.parse_known_args()
    if args.auth:
        user, pwd = args.auth.split(":", 1)
        Config.GRADIO_AUTH = (user, pwd)
    _launch(share=args.share, port=args.port)
else:
    _launch()
