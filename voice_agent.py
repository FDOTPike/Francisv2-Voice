from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys
import wave
from urllib.parse import urlparse
import numpy as np
import requests
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from studio_mastering import master_synthesized_voice

# ================= CONFIGURATION =================
# Use "http://127.0.0.1:9880" if running on the same PC as GPT-SoVITS,
# or "http://<PC_LAN_IP>:9880" when connecting from a Mac or another computer on LAN.
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:9880")
GPT_SOVITS_ROOT = os.environ.get("GPT_SOVITS_ROOT", r"C:\GPT-SoVITS")
MODEL_NAME = os.environ.get("VOICE_MODEL_NAME", "Francisv2")

# Default relative paths inside GPT-SoVITS
DEFAULT_GPT_WEIGHT = f"GPT_weights_v2ProPlus/{MODEL_NAME}-e15.ckpt"
DEFAULT_SOVITS_WEIGHT = f"SoVITS_weights_v2ProPlus/{MODEL_NAME}_e8_s344.pth"

# Reference audio and transcript bundled in this repo
LOCAL_REF_AUDIO = os.path.join(REPO_ROOT, "ref", "ref.wav")
LOCAL_REF_TRANSCRIPT_FILE = os.path.join(REPO_ROOT, "ref", "ref_transcript.txt")
LOCAL_AUX_REFS = [
    os.path.join(REPO_ROOT, "ref", f"ref_aux_{i}.wav").replace("\\", "/")
    for i in (1, 2, 3)
    if os.path.exists(os.path.join(REPO_ROOT, "ref", f"ref_aux_{i}.wav"))
]

DEFAULT_REF_TRANSCRIPT = "You think this guy is telling the truth? He is young as hell, Sarah, why?"
if os.path.exists(LOCAL_REF_TRANSCRIPT_FILE):
    with open(LOCAL_REF_TRANSCRIPT_FILE, "r", encoding="utf-8") as _f:
        REF_TRANSCRIPT = _f.read().strip() or DEFAULT_REF_TRANSCRIPT
else:
    REF_TRANSCRIPT = DEFAULT_REF_TRANSCRIPT

# Get a free key at https://console.groq.com/keys (or set GROQ_API_KEY in environment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_your_free_groq_key_here")
# =================================================


def normalize_tts_text(text: str) -> str:
    """
    Normalizes Australian English greetings and technical acronyms/version strings
    so GPT-SoVITS's English frontend pronounces them naturally without slurring
    or splitting 1-word exclamations into unstable fragments.
    """
    t = text.strip()
    t = re.sub(r"\bG'day!\s*", "G'day, ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFrancisv2\b", "Francis version two", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFrancis\s+v2\b", "Francis version two", t, flags=re.IGNORECASE)
    t = re.sub(r"\bGPT-SoVITS\b", "G P T So Vits", t, flags=re.IGNORECASE)
    return t


def is_local_server(server_url: str) -> bool:
    host = (urlparse(server_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def resolve_paths(server_url: str, root_dir: str, model_name: str) -> tuple[str, str, str, list[str]]:
    """
    Resolves GPT weight, SoVITS weight, primary reference audio, and auxiliary reference paths.
    If connecting to a local server (127.0.0.1), uses local repo weights/ref directly
    (auto-downloading them from GitHub Releases if missing).
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
                LOCAL_AUX_REFS,
            )

    gpt_rel = DEFAULT_GPT_WEIGHT
    sovits_rel = DEFAULT_SOVITS_WEIGHT
    if os.path.isdir(root_dir):
        gpt_candidates = []
        sovits_candidates = []
        for folder in ["GPT_weights_v2ProPlus", "GPT_weights_v2", "GPT_weights_v2Pro", "GPT_weights"]:
            gpt_candidates.extend(glob.glob(os.path.join(root_dir, folder, f"{model_name}*.ckpt")))
        for folder in ["SoVITS_weights_v2ProPlus", "SoVITS_weights_v2", "SoVITS_weights_v2Pro", "SoVITS_weights"]:
            sovits_candidates.extend(glob.glob(os.path.join(root_dir, folder, f"{model_name}*.pth")))
        if gpt_candidates:
            gpt_rel = os.path.relpath(max(gpt_candidates, key=os.path.getmtime), root_dir).replace("\\", "/")
        if sovits_candidates:
            sovits_rel = os.path.relpath(max(sovits_candidates, key=os.path.getmtime), root_dir).replace("\\", "/")

    ref_server_path = "ref.wav"
    if os.path.isdir(root_dir) and os.path.exists(os.path.join(root_dir, "ref.wav")):
        ref_server_path = os.path.join(root_dir, "ref.wav").replace("\\", "/")

    return gpt_rel, sovits_rel, ref_server_path, []


def load_weights_on_server(server_url: str, gpt_weight: str, sovits_weight: str):
    """Loads the fine-tuned Francisv2 GPT (s1v3) and SoVITS (v2ProPlus) weights onto api_v2.py."""
    print(f"Loading Francisv2 Studio Master weights on {server_url} ...")
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
    print("Francisv2 v2ProPlus weights loaded successfully!")


def open_audio_stream(sample_rate: int = 48000):
    """Initializes PyAudio for 48000 Hz, 16-bit mono PCM playback."""
    try:
        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
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
    aux_ref_audio_paths: list[str] | None = None,
    ref_transcript: str = REF_TRANSCRIPT,
):
    """
    Synthesizes speech with Francisv2 v2ProPlus (multi-reference SV fusion + full-context decode)
    and runs the 48kHz Studio Broadcast Mastering DSP chain (formant-preserving harmonic comb
    de-fuzzing + studio EQ + downward expander noise gate).
    """
    norm_text = normalize_tts_text(text_to_speak)
    payload = {
        "text": norm_text,
        "text_lang": "en",
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths or LOCAL_AUX_REFS,
        "prompt_text": ref_transcript,
        "prompt_lang": "en",
        "top_k": 10,
        "top_p": 0.88,
        "temperature": 0.75,
        "text_split_method": "cut4",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.25,
        "seed": 42,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "streaming_mode": False,
        "media_type": "wav",
    }

    r = requests.post(f"{server_url}/tts", json=payload, timeout=180)
    r.raise_for_status()

    y_raw, sr_raw = sf.read(io.BytesIO(r.content), dtype="float32")
    y_master, sr_master = master_synthesized_voice(y_raw, orig_sr=sr_raw, target_sr=48000)

    if audio_stream is not None:
        pcm16 = (np.clip(y_master, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        audio_stream.write(pcm16)

    if save_wav_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_wav_path)), exist_ok=True)
        sf.write(save_wav_path, y_master, sr_master, subtype="PCM_16")
        print(f"Saved Studio Master speech ({len(y_master) / sr_master:.2f}s @ {sr_master}Hz) to {save_wav_path}")


def run_conversation_loop(
    server_url: str,
    audio_stream,
    ref_audio_path: str,
    aux_ref_paths: list[str],
    ref_transcript: str,
):
    """Runs the real-time microphone -> Whisper STT -> Llama 3.3 -> Francisv2 Studio Master TTS loop."""
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

    print("\n=== Francisv2 Studio Master Voice Agent Ready! Speak into your microphone (Ctrl+C to exit) ===")
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        while True:
            try:
                print("\n[Listening...]")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=20)
                wav_bytes = audio.get_wav_data()

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

                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=180,
                )
                reply_text = completion.choices[0].message.content.strip()
                print(f"Francisv2: {reply_text}")
                conversation_history.append({"role": "assistant", "content": reply_text})

                speak_realtime(
                    reply_text,
                    server_url=server_url,
                    audio_stream=audio_stream,
                    ref_audio_path=ref_audio_path,
                    aux_ref_audio_paths=aux_ref_paths,
                    ref_transcript=ref_transcript,
                )

            except KeyboardInterrupt:
                print("\nExiting Francisv2 voice agent.")
                break
            except Exception as e:
                print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Francisv2 Studio Master Real-Time Voice Clone & AI Voice Agent (GPT-SoVITS v2ProPlus)"
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
        help="Synthesize a specific text string with Francisv2 Studio Master and exit.",
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

    gpt_w, sovits_w, ref_audio, aux_refs = resolve_paths(args.server, GPT_SOVITS_ROOT, MODEL_NAME)
    load_weights_on_server(args.server, gpt_w, sovits_w)

    p, audio_stream = (None, None) if args.no_speaker else open_audio_stream(48000)
    try:
        if args.say:
            print(f"Synthesizing with Francisv2 Studio Master: {args.say!r}")
            speak_realtime(
                args.say,
                server_url=args.server,
                audio_stream=audio_stream,
                save_wav_path=args.out,
                ref_audio_path=ref_audio,
                aux_ref_audio_paths=aux_refs,
                ref_transcript=REF_TRANSCRIPT,
            )
        else:
            run_conversation_loop(args.server, audio_stream, ref_audio, aux_refs, REF_TRANSCRIPT)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        if p is not None:
            p.terminate()


if __name__ == "__main__":
    main()
