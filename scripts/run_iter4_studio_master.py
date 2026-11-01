from __future__ import annotations

import functools
import json
import os
import sys
import numpy as np
import scipy.signal as signal
import soundfile as sf

GPT_SOVITS_ROOT = r"C:\GPT-SoVITS"
os.chdir(GPT_SOVITS_ROOT)
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation")
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation\scripts")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from studio_mastering import master_synthesized_voice
from evaluate_quality import evaluate_wav


def main():
    studio_dir = r"C:\GPT-SoVITS\output\slicer_opt_studio"
    asr_list_path = r"C:\GPT-SoVITS\output\asr_opt\slicer_opt.list"
    transcripts = {}
    with open(asr_list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 4:
                transcripts[os.path.basename(parts[0])] = parts[3].strip()

    # Primary prompt: 5.78s natural conversational cadence (SV=0.9326)
    # Aux prompts: Top-3 highest-HNR clips (+2.89dB, +2.14dB, +0.48dB) for acoustic tone fusion
    primary_name = "francis_v2_raw.wav_0012707840_0012892800.wav"
    aux_names = [
        "francis_v2_raw.wav_0000955840_0001150400.wav",
        "francis_v2_raw.wav_0014018560_0014187520.wav",
        "francis_v2_raw.wav_0014738880_0014886720.wav",
    ]

    ref_dir = r"C:\Users\fpike\VoiceCreation\ref"
    os.makedirs(ref_dir, exist_ok=True)

    primary_src = os.path.join(studio_dir, primary_name)
    primary_text = transcripts[primary_name]

    y_ref, sr_ref = sf.read(primary_src, dtype="float32")
    if sr_ref != 32000:
        y_ref = signal.resample_poly(y_ref, 32000, sr_ref).astype(np.float32)
    ref_wav_path = os.path.join(ref_dir, "ref.wav")
    sf.write(ref_wav_path, y_ref, 32000, subtype="PCM_16")
    sf.write(r"C:\GPT-SoVITS\ref.wav", y_ref, 32000, subtype="PCM_16")
    with open(os.path.join(ref_dir, "ref_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(primary_text + "\n")
    with open(r"C:\GPT-SoVITS\ref_transcript.txt", "w", encoding="utf-8") as f:
        f.write(primary_text + "\n")

    aux_paths = []
    for idx, aname in enumerate(aux_names, start=1):
        asrc = os.path.join(studio_dir, aname)
        if os.path.exists(asrc):
            y_a, sr_a = sf.read(asrc, dtype="float32")
            if sr_a != 32000:
                y_a = signal.resample_poly(y_a, 32000, sr_a).astype(np.float32)
            ap = os.path.join(ref_dir, f"ref_aux_{idx}.wav")
            sf.write(ap, y_a, 32000, subtype="PCM_16")
            aux_paths.append(ap.replace("\\", "/"))

    meta = {
        "primary_ref_name": primary_name,
        "primary_ref_text": primary_text,
        "aux_ref_names": aux_names,
        "aux_ref_paths": aux_paths,
    }
    with open(os.path.join(ref_dir, "ref_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2ProPlus"
    tts_config.t2s_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2-e15.ckpt"
    tts_config.vits_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2_e8_s344.pth"

    tts = TTS(tts_config)

    orig_decode = tts.vits_model.decode

    @functools.wraps(orig_decode)
    def clean_decode(*args, **kwargs):
        kwargs["noise_scale"] = 0.45
        return orig_decode(*args, **kwargs)

    tts.vits_model.decode = clean_decode

    norm_target = (
        "G'day, this is Francis version two running locally on G P T So Vits. "
        "Feel free to use this voice in your own projects as long as you credit Francis version two."
    )
    inputs = {
        "text": norm_target,
        "text_lang": "en",
        "ref_audio_path": ref_wav_path.replace("\\", "/"),
        "aux_ref_audio_paths": aux_paths,
        "prompt_text": primary_text,
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
    }

    sr_raw, audio_raw = next(tts.run(inputs))
    audio_master, sr_master = master_synthesized_voice(audio_raw, orig_sr=sr_raw, target_sr=48000)

    out_path = r"C:\Users\fpike\VoiceCreation\samples\iterations\iter4_sample.wav"
    sf.write(out_path, audio_master, sr_master, subtype="PCM_16")
    print(f"Saved Iteration 4 Studio Master sample to {out_path} (sr={sr_master}, len={len(audio_master)/sr_master:.2f}s)")

    res = evaluate_wav(out_path, "Iteration 4 (v2ProPlus + Pitch-Sync Comb De-Fuzz + STFT Harmonic Contrast + 48kHz Studio Master)")
    print("=== ITERATION 4 EVALUATION ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
