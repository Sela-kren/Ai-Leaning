import glob
import os
import re
import tempfile
import threading
import wave
import winsound
from tkinter import BOTH, END, LEFT, RIGHT, BooleanVar, Text, Tk, ttk
from typing import Optional

from piper.config import SynthesisConfig
from piper.voice import PiperVoice

# Default voice: jenny_dioco medium (set VOICE_PATH env to override)
DEFAULT_VOICE = "voices/en_GB-jenny_dioco-medium.onnx"
VOICE_PATH = os.environ.get("VOICE_PATH", DEFAULT_VOICE)
_play_lock = threading.Lock()  # serialize playback to avoid duplicate plays

# Talk speed: lower = faster, higher = slower (default 1.0)
TALK_SPEED = max(float(os.environ.get("TALK_SPEED", "0.9")), 0.1)


def ensure_voice_files(model_path: str) -> str:
    """Ensure the matching config (.onnx.json) exists next to the model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Voice model not found: {model_path}")

    config_path = f"{model_path}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing config: {config_path}\n"
            "Download the matching .onnx.json file from the Piper voice release and place it in the same folder."
        )
    return model_path


def resolve_voice_path() -> str:
    """Return a voice model path (env VOICE_PATH, else first voices/*.onnx)."""
    if VOICE_PATH:
        return ensure_voice_files(VOICE_PATH)

    candidates = sorted(glob.glob(os.path.join("voices", "*.onnx")))
    if not candidates:
        raise FileNotFoundError("No .onnx voice found in voices/ and VOICE_PATH is not set.")
    return ensure_voice_files(candidates[0])


def _clean_text_for_tts(text: str) -> str:
    """
    Trim stray trailing quotes/smart quotes that can cause artifacts.
    If the cleaned text is empty, fall back to the original stripped text.
    """
    cleaned = text.strip()
    cleaned = re.sub(r'[\\s\"“”\'’‘]+$', "", cleaned)
    return cleaned or text.strip()


def text_to_speech(
    text: str,
    out_wav: Optional[str] = None,
    voice: Optional[PiperVoice] = None,
    speed: Optional[float] = None,
) -> str:
    """
    Synthesize text to a WAV file and return the path.

    speed: length_scale override for Piper (lower=faster, higher=slower).
    """
    voice = voice or PiperVoice.load(resolve_voice_path())
    out_wav = out_wav or tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    length_scale = max(speed if speed is not None else TALK_SPEED, 0.1)
    synth_cfg = SynthesisConfig(length_scale=length_scale)
    text = _clean_text_for_tts(text)

    with wave.open(out_wav, "wb") as f:
        voice.synthesize_wav(text, f, syn_config=synth_cfg, set_wav_format=True)

    return out_wav


def play_text(text: str, voice: PiperVoice, status_cb=None):
    """Run TTS and playback in a thread-safe way."""
    if not _play_lock.acquire(blocking=False):
        if status_cb:
            status_cb("Already playing, please wait...")
        return

    try:
        if status_cb:
            status_cb("Synthesizing...")
        wav_path = text_to_speech(text, voice=voice)
        if status_cb:
            status_cb("Playing...")
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    finally:
        if "wav_path" in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        _play_lock.release()
        if status_cb:
            status_cb("Ready")


def build_gui():
    model_path = resolve_voice_path()
    voice = PiperVoice.load(model_path)

    root = Tk()
    root.title("Piper TTS")
    root.geometry("420x280")

    ttk.Label(root, text=f"Voice: {os.path.basename(model_path)}").pack(pady=6)

    text_box = Text(root, wrap="word", height=10)
    text_box.pack(fill=BOTH, expand=True, padx=10)

    status_var = ttk.Label(root, text="Ready")
    status_var.pack(pady=4)

    def set_status(msg: str):
        status_var.config(text=msg)
        status_var.update_idletasks()

    def do_read(current_text: str):
        if not current_text.strip():
            set_status("Enter text to read")
            return
        threading.Thread(target=play_text, args=(current_text, voice, set_status), daemon=True).start()

    def on_read_clicked():
        current_text = text_box.get("1.0", END)
        do_read(current_text)

    def on_paste_read_clicked():
        try:
            clip = root.clipboard_get()
            text_box.delete("1.0", END)
            text_box.insert("1.0", clip)
            do_read(clip)
        except Exception as exc:  # best-effort clipboard read
            set_status(f"Clipboard error: {exc}")

    button_frame = ttk.Frame(root)
    button_frame.pack(fill=BOTH, padx=10, pady=6)

    read_btn = ttk.Button(button_frame, text="Read Text", command=on_read_clicked)
    read_btn.pack(side=LEFT, expand=True, fill=BOTH, padx=(0, 5))

    paste_read_btn = ttk.Button(button_frame, text="Paste && Read", command=on_paste_read_clicked)
    paste_read_btn.pack(side=RIGHT, expand=True, fill=BOTH, padx=(5, 0))

    root.mainloop()


if __name__ == "__main__":
    build_gui()
