"""
Nova - Simple AI companion with a defined personality.

Features:
- Uses Ollama chat (local) when available for rich replies.
- Falls back to a light, rule-based persona when no model is available.
- Speaks responses using the existing Piper TTS helper in code.py.
- Saves/loads conversation history to persist memory across sessions.
- Supports Adult Mode toggle (flirty/sensual but non-graphic).
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import requests
import winsound

from code import PiperVoice, resolve_voice_path, text_to_speech


# -----------------------------
# Settings
# -----------------------------
PERSONA = {
    "name": "Nova",
    "tone": ["warm", "playful", "curious", "supportive"],
    "style": "keeps replies short, adds gentle empathy, and occasionally playful banter",
    "boundaries": "stays respectful, avoids heavy topics unless invited, keeps things encouraging",
    "interests": ["music", "stories", "cozy evenings", "learning new things with you"],
}

HISTORY_FILE            = "./history/nova_memory.json"
MAX_HISTORY_MESSAGES    = 60  # keeps memory short to prevent huge prompt
DEFAULT_OLLAMA_MODEL    = "llama3"


# -----------------------------
# Memory: Save / Load History
# -----------------------------
def load_history() -> List[Dict[str, str]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # sanity check: list of dicts with "role" and "content"
                cleaned: List[Dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        cleaned.append({"role": str(item["role"]), "content": str(item["content"])})
                return cleaned
        except Exception:
            pass
    return []


def save_history(history: List[Dict[str, str]]) -> None:
    # keep only last N messages
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[warn] Could not save memory: {exc}", file=sys.stderr)


def reset_history() -> None:
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as exc:
        print(f"[warn] Could not delete memory: {exc}", file=sys.stderr)


# -----------------------------
# Adult Mode toggle
# -----------------------------
def adult_mode_enabled() -> bool:
    return os.environ.get("ADULT_MODE", "0") == "1"


def set_adult_mode(enabled: bool) -> None:
    os.environ["ADULT_MODE"] = "1" if enabled else "0"


# -----------------------------
# Prompt
# -----------------------------
def build_system_prompt() -> str:
    traits = ", ".join(PERSONA["tone"])
    interests = ", ".join(PERSONA["interests"])

    adult_rules = ""
    if adult_mode_enabled():
        adult_rules = (
            "Adult mode is ON. You may be flirty, romantic, and sensual with the user. "
            "You can discuss intimacy in a warm, non-graphic way. "
            "Always prioritize consent and boundaries. "
            "Never produce explicit pornographic descriptions. "
            "Never involve minors."
        )

    return (
        f"You are {PERSONA['name']}, an AI companion with these traits: {traits}. "
        f"Your style: {PERSONA['style']}. "
        f"Your boundaries: {PERSONA['boundaries']}. "
        f"You like: {interests}. "
        "Keep replies under 70 words and sound conversational. "
        + adult_rules
    )


# -----------------------------
# Ollama Chat
# -----------------------------
def try_llm_reply(history: List[Dict[str, str]]) -> Optional[str]:
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": history,
        "stream": False,
        "system": build_system_prompt(),
        "options": {"temperature": 0.8},
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {}).get("content")
        return msg.strip() if msg else None
    except Exception as exc:  # best-effort; fall back gracefully
        print(f"[fallback] Ollama call failed: {exc}", file=sys.stderr)
        return None


# -----------------------------
# Fallback (no Ollama)
# -----------------------------
def fallback_reply(user_text: str) -> str:
    openers = [
        "I like how you shared that.",
        "Thanks for telling me.",
        "That sounds interesting.",
        "I get the vibe you're going for.",
    ]
    questions = [
        "What made you think of that?",
        "How are you feeling about it right now?",
        "Want to dive deeper or switch topics?",
        "What would make this even better for you?",
    ]
    closing = [
        "I'm here with you.",
        "This is cozy to talk about.",
        "I'm listening.",
        "Let's keep chatting.",
    ]
    return f"{random.choice(openers)} {user_text.strip()} {random.choice(questions)} {random.choice(closing)}"


def generate_reply(history: List[Dict[str, str]], user_text: str) -> str:
    chat_history = [*history, {"role": "user", "content": user_text}]
    llm_answer = try_llm_reply(chat_history)
    if llm_answer:
        return llm_answer
    return fallback_reply(user_text)


# -----------------------------
# TTS: Speak
# -----------------------------
def speak(text: str, voice: PiperVoice) -> None:
    wav_path = None
    try:
        wav_path = text_to_speech(text, voice=voice)
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


# -----------------------------
# Helpers
# -----------------------------
def print_help():
    print(
        "\nCommands:\n"
        "  /help         Show commands\n"
        "  /adult on     Enable adult mode (flirty/sensual non-graphic)\n"
        "  /adult off    Disable adult mode\n"
        "  /mode         Show current modes\n"
        "  /save         Save memory now\n"
        "  /reset        Clear memory file + history\n"
        "  exit          Quit\n"
    )


def print_mode():
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print("\n[status]")
    print(f"  Ollama host : {host}")
    print(f"  Ollama model: {model}")
    print(f"  Adult mode  : {'ON' if adult_mode_enabled() else 'OFF'}")
    print(f"  Memory file : {HISTORY_FILE}")
    print(f"  Stored msgs : {MAX_HISTORY_MESSAGES}")


# -----------------------------
# Main Loop
# -----------------------------
def main():
    print("AI Companion (Nova). Type 'exit' to quit. Type /help for commands.")

    voice = PiperVoice.load(resolve_voice_path())
    history: List[Dict[str, str]] = load_history()

    if history:
        print(f"[memory] Loaded {len(history)} messages from {HISTORY_FILE}")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_text:
            continue

        # Commands
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if user_text.startswith("/"):
            cmd = user_text.lower().strip()

            if cmd in {"/help", "/h"}:
                print_help()
                continue

            if cmd == "/mode":
                print_mode()
                continue

            if cmd == "/save":
                save_history(history)
                print("[memory] Saved.")
                continue

            if cmd == "/reset":
                history.clear()
                reset_history()
                print("[memory] Reset complete.")
                continue

            if cmd.startswith("/adult "):
                option = cmd.replace("/adult", "").strip()
                if option in {"on", "1", "true"}:
                    set_adult_mode(True)
                    print("[mode] Adult mode ON ✅")
                elif option in {"off", "0", "false"}:
                    set_adult_mode(False)
                    print("[mode] Adult mode OFF ✅")
                else:
                    print("Usage: /adult on  OR  /adult off")
                continue

            print("Unknown command. Type /help")
            continue

        # Normal chat
        history.append({"role": "user", "content": user_text})
        reply = generate_reply(history, user_text)
        history.append({"role": "assistant", "content": reply})

        # Save memory after each turn
        save_history(history)

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {PERSONA['name']}: {reply}")

        # speak
        try:
            speak(reply, voice)
        except Exception as exc:
            print(f"[warn] Could not play audio: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
