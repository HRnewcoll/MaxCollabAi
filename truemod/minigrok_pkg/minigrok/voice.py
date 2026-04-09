"""
minigrok.voice
AI voice: TTS (F5/Kokoro/gTTS), Whisper STT
"""


# Cross-module imports
import os, json, re, logging, threading, io, wave, base64
import warnings
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_SILENT", "true")

from .base import log
from .config import Config, DIRS
from .models import quick, VRAMJuggler


# ════════════════════════════════════════════════════════════════════
# § 11  AI VOICE  (F5-TTS → Kokoro → gTTS)
# ════════════════════════════════════════════════════════════════════

_tts_engine = "none"
_f5 = None
_kokoro = None
_voice_profiles: dict = {}
VP_PATH = Path(f"{DR}/voices/profiles.json")
if VP_PATH.exists():
    try:
        _voice_profiles = json.loads(VP_PATH.read_text())
    except Exception:
        pass

_EMOTION_KEYWORDS = {
    "happy": ["great", "amazing", "excellent", "love", "perfect", "wonderful", "fantastic"],
    "sad": ["sorry", "loss", "failed", "regret", "unfortunately", "sad"],
    "shocked": ["impossible", "no way", "omg", "wow", "unbelievable"],
    "angry": ["wrong", "ridiculous", "broken", "terrible", "awful"],
    "excited": ["can't wait", "yes!", "brilliant", "love it", "awesome"]
}
_EMOTION_SPEEDS = {"happy": 1.15, "sad": 0.85, "shocked": 1.25, "angry": 1.2, "excited": 1.2, "neutral": 1.0}

def detect_emotion(text: str) -> str:
    """Detect emotion from text using keyword matching."""
    tl = text.lower()
    scores = {e: sum(1 for k in kws if k in tl) for e, kws in _EMOTION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"

def _init_tts():
    global _f5, _kokoro, _tts_engine
    try:
        from f5_tts.api import F5TTS
        _f5 = F5TTS()
        _tts_engine = "f5"
        log.info("  F5-TTS (AI voice cloning)")
        return
    except Exception as e:
        log.debug(f"  F5: {e}")
    try:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        _tts_engine = "kokoro"
        log.info("  Kokoro TTS")
        return
    except Exception as e:
        log.debug(f"  Kokoro: {e}")
    try:
        from gtts import gTTS
        _tts_engine = "gtts"
        log.info("  gTTS fallback")
    except Exception:
        _tts_engine = "espeak"

_init_tts()

def synthesise(text, emotion="auto", voice_clip=None, lang="en") -> Optional[str]:
    """Generate speech audio from text."""
    if not Config.ENABLE_VOICE:
        return None
    if emotion == "auto":
        emotion = detect_emotion(text)
    clean = re.sub(r"[*#`_\[\]<>{}]", "", text)
    clean = re.sub(r"\n+", " ", clean).strip()[:700]
    if not clean:
        return None
    speed = _EMOTION_SPEEDS.get(emotion, 1.0)
    out = os.path.join(DIRS["outputs"], f"tts_{int(time.time()*1000)}.wav")

    if _tts_engine == "f5" and _f5:
        try:
            import soundfile as sf
            kw = {"gen_text": clean, "speed": speed}
            if voice_clip and Path(voice_clip).exists():
                kw["ref_audio_path"] = voice_clip
                kw["ref_text"] = ""
            wav, sr, _ = _f5.infer(**kw)
            sf.write(out, wav, sr)
            return out
        except Exception as e:
            log.warning(f"F5 TTS: {e}")
    if _kokoro:
        try:
            import soundfile as sf
            voice_map = {"happy": "af_bella", "sad": "af", "shocked": "af_sky",
                         "angry": "am_adam", "neutral": "af_bella", "excited": "af_sky"}
            wav, sr = _kokoro.create(clean, voice=voice_map.get(emotion, "af_bella"), speed=speed, lang="en-us")
            sf.write(out, wav, sr)
            return out
        except Exception as e:
            log.warning(f"Kokoro TTS: {e}")
    try:
        from gtts import gTTS
        mp3 = out.replace(".wav", ".mp3")
        gTTS(text=clean, lang=lang[:2], slow=(emotion == "sad")).save(mp3)
        return mp3
    except Exception:
        pass
    try:
        subprocess.run(["espeak", f"--speed={int(speed*150)}", "-w", out, clean],
                       capture_output=True, timeout=15)
        return out
    except Exception:
        return None

def create_voice_profile(name, desc, clip_path) -> str:
    """Create a voice clone profile from a reference audio clip."""
    if not Path(clip_path).exists():
        return f"❌ Not found: {clip_path}"
    dest = os.path.join(f"{DR}/voices", f"{name}{Path(clip_path).suffix}")
    shutil.copy(clip_path, dest)
    _voice_profiles[name] = {
        "desc": desc, "clip": dest, "engine": _tts_engine,
        "created": datetime.now().isoformat()
    }
    VP_PATH.write_text(json.dumps(_voice_profiles, indent=2))
    return f"✅ Voice profile '{name}' created ({_tts_engine})"

_whisper_models: dict = {}
_cv2_module = None   # Lazy

def _get_cv2():
    """Lazy-load cv2 on first use (saves startup VRAM)."""
    global _cv2_module
    if _cv2_module is None:
        try:
            import cv2 as _cv2
            _cv2_module = _cv2
        except ImportError:
            raise ToolError("opencv-python-headless not installed. Run: pip install opencv-python-headless")
    return _cv2_module

def _get_whisper(size="base"):
    """Lazy-load Whisper on first use (saves ~300MB VRAM at boot)."""
    if size not in _whisper_models:
        if not Config.ENABLE_VOICE:
            raise ToolError("Voice is disabled. Set Config.ENABLE_VOICE=True.")
        try:
            import whisper as _whisper
        except ImportError:
            raise ToolError("openai-whisper not installed.")
        _whisper_models[size] = _whisper.load_model(size)
    return _whisper_models[size]

def voice_turn(audio_path, whisper_size="base", voice_profile="default", history=None) -> tuple:
    """Process voice input: transcribe → generate reply → synthesise speech."""
    if not audio_path:
        return "", "", None
    transcript = _get_whisper(whisper_size).transcribe(audio_path, language="en")["text"].strip()
    if not transcript:
        return "", "", None
    msgs = _build_msgs(history or [], transcript)
    msgs[0]["content"] += "\n\n[VOICE MODE: 2-3 sentences. No markdown. Natural speech.]"
    reply = "".join(stream_gen(msgs, max_new_tokens=200, temperature=0.7))
    reply = _strip_think(reply)
    clip = _voice_profiles.get(voice_profile, {}).get("clip")
    return transcript, reply, synthesise(reply, voice_clip=clip)

log.info(f"AI TTS ({_tts_engine})")


