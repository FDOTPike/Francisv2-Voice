from __future__ import annotations

import json
import os
import re
import sys
import numpy as np
import soundfile as sf

os.chdir(r"C:\GPT-SoVITS")
sys.path.insert(0, r"C:\GPT-SoVITS")
sys.path.insert(0, r"C:\GPT-SoVITS\GPT_SoVITS")
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation\scripts")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from evaluate_quality import evaluate_wav, TARGET_SENTENCE


def normalize_tts_text(text: str) -> str:
    """
    Normalizes Australian English greetings and technical acronyms/version strings
    so GPT-SoVITS's English frontend pronounces them naturally without slurring
    or splitting 1-word exclamations into unstable fragments.
    """
    t = text.strip()
    # Prevent 1-word exclamation fragment split on "G'day!"
    t = re.sub(r"\bG'day!\s*", "G'day, ", t, flags=re.IGNORECASE)
    # Expand version shorthand that G2P reads awkwardly as "vtwo"
    t = re.sub(r"\bFrancisv2\b", "Francis version two", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFrancis\s+v2\b", "Francis version two", t, flags=re.IGNORECASE)
    t = re.sub(r"\bGPT-SoVITS\b", "G P T So Vits", t, flags=re.IGNORECASE)
    return t


def main():
    ref_dir = r"C:\Users\fpike\VoiceCreation\ref"
    with open(os.path.join(ref_dir, "ref_metadata.json"), "r", encoding="utf-8") as f:
        ref_meta = json.load(f)

    ref_audio = os.path.join(ref_dir, "ref.wav").replace("\\", "/")
    ref_text = open(os.path.join(ref_dir, "ref_transcript.txt"), "r", encoding="utf-8").read().strip()
    aux_refs = ref_meta["aux_ref_paths"]

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2"
    tts_config.t2s_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2-e15.ckpt"
    tts_config.vits_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2_e8_s344.pth"

    tts = TTS(tts_config)

    norm_target = normalize_tts_text(TARGET_SENTENCE)
    inputs = {
        "text": norm_target,
        "text_lang": "en",
        "ref_audio_path": ref_audio,
        "aux_ref_audio_paths": aux_refs,
        "prompt_text": ref_text,
        "prompt_lang": "en",
        "top_k": 12,
        "top_p": 0.9,
        "temperature": 0.8,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.28,
        "seed": 42,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "streaming_mode": False,
    }

    sr, audio = next(tts.run(inputs))
    out_path = r"C:\Users\fpike\VoiceCreation\samples\iterations\iter1_sample.wav"
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"Saved Iteration 1 sample to {out_path} (sr={sr}, len={len(audio)/sr:.2f}s)")

    res = evaluate_wav(out_path, "Iteration 1 (Studio Ref Conditioning + Multi-Ref Fusion + Full-Context Decode)")
    print("=== ITERATION 1 EVALUATION ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
