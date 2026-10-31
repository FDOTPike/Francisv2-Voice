from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import wave
from urllib.parse import urlparse
import requests

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================= CONFIGURATION =================
# Use "http://127.0.0.1:9880" if running on the same PC as GPT-SoVITS,
# or "http://<PC_LAN_IP>:9880" when connecting from a Mac or another computer on LAN.
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:9880")
GPT_SOVITS_ROOT = os.environ.get("GPT_SOVITS_ROOT", r"C:\GPT-SoVITS")
MODEL_NAME = os.environ.get("VOICE_MODEL_NAME", "Francisv2")

# Default relative paths inside GPT-SoVITS
DEFAULT_GPT_WEIGHT = f"GPT_weights_v2/{MODEL_NAME}-e15.ckpt"
DEFAULT_SOVITS_WEIGHT = f"SoVITS_weights_v2/{MODEL_NAME}_e8_s344.pth"

# Reference audio and transcript bundled in this repo
LOCAL_REF_AUDIO = os.path.join(REPO_ROOT, "ref", "ref.wav")
LOCAL_REF_TRANSCRIPT_FILE = os.path.join(REPO_ROOT, "ref", "ref_transcript.txt")

DEFAULT_REF_TRANSCRIPT = "2017 he finished. If I went to uni one year early, which I could have."
if os.path.exists(LOCAL_REF_TRANSCRIPT_FILE):
    with open(LOCAL_REF_TRANSCRIPT_FILE, "r", encoding="utf-8") as _f:
        REF_TRANSCRIPT = _f.read().strip() or DEFAULT_REF_TRANSCRIPT
else:
    REF_TRANSCRIPT = DEFAULT_REF_TRANSCRIPT

# Get a free key at https://console.groq.com/keys (or set GROQ_API_KEY in environment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_your_free_groq_key_here")
# =================================================


def is_local_server(server_url: str) -> bool:
    host = (urlparse(server_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def resolve_paths(server_url: str, root_dir: str, model_name: str) -> tuple[str, str, str]:
    """
    Resolves GPT weight, SoVITS weight, and reference audio paths.
    If connecting to a local server (127.0.0.1), uses local repo weights/ref directly
    (auto-downloading them from GitHub Releases if missing).
    If connecting to a remote LAN server, uses the server-side paths inside GPT-SoVITS.
    """
    local_gpt = os.path.join(REPO_ROOT, "weights", f"{model_name}-e15.ckpt")
    local_sovits = os.path.join(REPO_ROOT, "weights", f"{model_name}_e8_s344.pth")

    if is_local_server(server_url):
        if not (os.path.exists(local_gpt) and os.path.exists(local_sovits)):
            try:
                from download_weights import ensure_weights

                print("Local weights not found in weights/; downloading from GitHub Release v1.0.0...")
                ensure_weights(repo_root=REPO_ROOT, gpt_sovits_dir=root_dir)
            except Exception as e:
                print(f"[Warning] Auto-download skipped ({e}); falling back to GPT-SoVITS folder.")

        if os.path.exists(local_gpt) and os.path.exists(local_sovits) and os.path.exists(LOCAL_REF_AUDIO):
            return (
                local_gpt.replace("\\", "/"),
                local_sovits.replace("\\", "/"),
                LOCAL_REF_AUDIO.replace("\\", "/"),
            )

    # Fallback: resolve inside GPT_SOVITS_ROOT or use server-relative paths
    gpt_rel = DEFAULT_GPT_WEIGHT
    sovits_rel = DEFAULT_SOVITS_WEIGHT
    if os.path.isdir(root_dir):
        gpt_candidates = []
        sovits_candidates = []
        for folder in ["GPT_weights_v2", "GPT_weights_v2Pro", "GPT_weights"]:
            gpt_candidates.extend(glob.glob(os.path.join(root_dir, folder, f"{model_name}*.ckpt")))
        for folder in ["SoVITS_weights_v2", "SoVITS_weights_v2Pro", "SoVITS_weights"]:
            sovits_candidates.extend(glob.glob(os.path.join(root_dir, folder, f"{model_name}*.pth")))
        if gpt_candidates:
            gpt_rel = os.path.relpath(max(gpt_candidates, key=os.path.getmtime), root_dir).replace("\\", "/")
        if sovits_candidates:
            sovits_rel = os.path.relpath(max(sovits_candidates, key=os.path.getmtime), root_dir).replace("\\", "/")

    ref_server_path = "ref.wav"
    if os.path.isdir(root_dir) and os.path.exists(os.path.join(root_dir, "ref.wav")):
        ref_server_path = os.path.join(root_dir, "ref.wav").replace("\\", "/")

    return gpt_rel, sovits_rel, ref_server_path


def load_weights_on_server(server_url: str, gpt_weight: str, sovits_weight: str):
    """Loads the fine-tuned Francisv2 GPT and SoVITS weights onto the api_v2.py server."""
    print(f"Loading Francisv2 voice weights on {server_url} ...")
    print(f"  GPT weight:    {gpt_weight}")
    print(f"  SoVITS weight: {sovits_weight}")
    r1 = requests.get(
        f"{server_url}/set_gpt_weights",
        params={"weights_path": gpt_weight},
        timeout=60,
    )
    r1.raise_for_status()
    r2 = requests.get(
        f"{server_url}/set_sovits_weights",
        params={"weights_path": sovits_weight},
        timeout=60,
    )
    r2.raise_for_status()
    print("Francisv2 voice weights loaded successfully!")


def open_audio_stream():
    """Initializes PyAudio for real-time 32000 Hz, 16-bit mono PCM playback."""
    try:
        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=32000,
            output=True,
        )
        return p, stream
    except Exception as e:
        print(f"[Info] Live speaker stream unavailable ({e}); audio will be saved to WAV if --out is set.")
        return None, None


def speak_realtime(
    text_to_speak: str,
    server_url: str = TTS_SERVER_URL,
    audio_stream=None,
    save_wav_path: str | None = None,
    ref_audio_path: str = LOCAL_REF_AUDIO,
    ref_transcript: str = REF_TRANSCRIPT,
):
    """Streams raw 32kHz 16-bit PCM audio chunks from GPT-SoVITS straight to speakers and/or WAV file."""
    payload = {
        "text": text_to_speak,
        "text_lang": "en",
        "ref_audio_path": ref_audio_path,
        "prompt_text": ref_transcript,
        "prompt_lang": "en",
        "streaming_mode": 2,          # Low-latency token streaming (True/1/2/3 supported)
        "media_type": "raw",          # Raw 16-bit 32kHz mono PCM bytes for zero-pop streaming
        "text_split_method": "cut5",  # Splits by punctuation for fast first-chunk response
    }

    pcm_bytes = bytearray()
    with requests.post(f"{server_url}/tts", json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                if audio_stream is not None:
                    audio_stream.write(chunk)
                if save_wav_path is not None:
                    pcm_bytes.extend(chunk)

    if save_wav_path and pcm_bytes:
        os.makedirs(os.path.dirname(os.path.abspath(save_wav_path)), exist_ok=True)
        with wave.open(save_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(32000)
            wf.writeframes(bytes(pcm_bytes))
        print(f"Saved synthesized speech ({len(pcm_bytes) / 64000:.2f}s) to {save_wav_path}")


def run_conversation_loop(server_url: str, audio_stream, ref_audio_path: str, ref_transcript: str):
    """Runs the real-time microphone -> Whisper STT -> Llama 3.3 -> Francisv2 TTS loop."""
    import speech_recognition as sr
    from groq import Groq

    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_free_groq_key_here":
        print(
            "\n[!] Set your free Groq API key via environment variable GROQ_API_KEY "
            "or inside voice_agent.py (https://console.groq.com/keys)."
        )
        sys.exit(1)

    groq_client = Groq(api_key=GROQ_API_KEY)
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8

    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are speaking with the Francisv2 Australian English voice model in a real-time "
                "voice conversation. Keep replies concise (1 to 3 sentences), natural, and conversational."
            ),
        }
    ]

    print("\n=== Francisv2 Real-Time Voice Agent Ready! Speak into your microphone (Ctrl+C to exit) ===")
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        while True:
            try:
                print("\n[Listening...]")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=20)
                wav_bytes = audio.get_wav_data()

                # 1. Instant STT via Groq Whisper
                transcription = groq_client.audio.transcriptions.create(
                    file=("mic.wav", io.BytesIO(wav_bytes)),
                    model="whisper-large-v3-turbo",
                    language="en",
                )
                user_text = transcription.text.strip()
                if not user_text:
                    continue
                print(f"You: {user_text}")
                conversation_history.append({"role": "user", "content": user_text})

                # 2. Fast LLM reply via Groq Llama 3.3
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=180,
                )
                reply_text = completion.choices[0].message.content.strip()
                print(f"Francisv2: {reply_text}")
                conversation_history.append({"role": "assistant", "content": reply_text})

                # 3. Stream Francisv2 voice in real time
                speak_realtime(
                    reply_text,
                    server_url=server_url,
                    audio_stream=audio_stream,
                    ref_audio_path=ref_audio_path,
                    ref_transcript=ref_transcript,
                )

            except KeyboardInterrupt:
                print("\nExiting Francisv2 voice agent.")
                break
            except Exception as e:
                print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Francisv2 Real-Time Voice Clone & AI Voice Agent (GPT-SoVITS v2)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=TTS_SERVER_URL,
        help="GPT-SoVITS api_v2.py server URL (default: http://127.0.0.1:9880).",
    )
    parser.add_argument(
        "--say",
        type=str,
        default=None,
        help="Synthesize a specific text string with Francisv2 and exit.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="francisv2_output.wav",
        help="Output WAV path when using --say.",
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Disable live speaker playback (useful for headless WAV generation).",
    )
    args = parser.parse_args()

    gpt_w, sovits_w, ref_audio = resolve_paths(args.server, GPT_SOVITS_ROOT, MODEL_NAME)
    load_weights_on_server(args.server, gpt_w, sovits_w)

    p, audio_stream = (None, None) if args.no_speaker else open_audio_stream()
    try:
        if args.say:
            print(f"Synthesizing with Francisv2: {args.say!r}")
            speak_realtime(
                args.say,
                server_url=args.server,
                audio_stream=audio_stream,
                save_wav_path=args.out,
                ref_audio_path=ref_audio,
                ref_transcript=REF_TRANSCRIPT,
            )
        else:
            run_conversation_loop(args.server, audio_stream, ref_audio, REF_TRANSCRIPT)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        if p is not None:
            p.terminate()


if __name__ == "__main__":
    main()
