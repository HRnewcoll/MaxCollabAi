import os, re

FILENAME = "c:/Users/hasan/Random/truemod/minigrok_v13.2.py"

with open(FILENAME, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Fix nest_asyncio
text = text.replace(
    "nest_asyncio.apply()",
    "try: nest_asyncio.apply()\nexcept Exception: pass"
)

# 2. Premium UI Replacement
# We target the range from MOBILE_CSS definition to the start of the Search tab.
MOBILE_CSS_START = 'MOBILE_CSS = """'
SEARCH_TAB_START = 'with gr.Tab("🔍 Search")'

start_idx = text.find(MOBILE_CSS_START)
end_idx = text.find(SEARCH_TAB_START)

if start_idx == -1 or end_idx == -1:
    print(f"Replacement markers not found! Start: {start_idx}, End: {end_idx}")
    exit(1)

NEW_CHAT_UI = \"\"\"MOBILE_CSS = \"\"\"
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
\"\"\"

def create_ui():
    with gr.Blocks(css=MOBILE_CSS, theme=gr.themes.Default(), title=f"{APP_NAME} v{APP_VERSION}") as app:
        file_st = gr.State(None)
        img_st  = gr.State(None)
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="", height=440, type="messages", autofocus=True)
            ctx_bar = gr.Textbox(label="", lines=1, interactive=False, value=f"0/{MODEL_MAX_TOKENS:,} tokens (0%)")
            subject_dd = gr.Dropdown(choices=["auto","maths","coding","science","english","humanities"], value="auto", label="📚 Subject", container=False)

            gr.HTML(\"\"\"<div class="chips">
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
            <script>function cht(t){const tb=document.querySelector("textarea");if(tb){tb.value=t;tb.dispatchEvent(new Event("input",{bubbles:true}));tb.focus();}}</script>\"\"\")

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
                audio_in.change(lambda a: _get_whisper(\"base\").transcribe(a, language=\"en\")[\"text\"].strip() if a else \"\", audio_in, chat_in)
            file_in.change(lambda f: (process_upload(f.name) if f else \"No file\", f.name if f else None), file_in, [status_box, file_st])
            img_in.change(lambda p: p, img_in, img_st)
            
            clr_btn.click(lambda: ([], [], \"0/32,768 tokens (0%)\"), None, [chatbot, chatbot, ctx_bar])
            thu_btn.click(lambda h: (db_add_dpo(h[-2][\"content\"], h[-1][\"content\"], \"\"), str(len(db_get_dpo()))), [chatbot], [dpo_ct, dpo_ct])
            thd_btn.click(lambda h: (db_add_dpo(h[-2][\"content\"], \"\", h[-1][\"content\"]), str(len(db_get_dpo()))), [chatbot], [dpo_ct, dpo_ct])
            export_btn.click(lambda h: export_conversation(h), [chatbot], status_box)
            ref_btn.click(system_status, None, status_box)
            free_btn.click(lambda: (torch.cuda.empty_cache(), \"VRAM Cleared\"), None, status_box)
            hlth_btn.click(lambda: json.dumps(health_check(), indent=2), None, status_box)

        # ── Rest of Tabs are preserved below ──
        \"\"\"

final_text = text[:start_idx] + NEW_CHAT_UI + text[end_idx:]

with open(FILENAME, "w", encoding="utf-8", newline="\n") as f:
    f.write(final_text)

print("v13.2.py patched successfully.")
