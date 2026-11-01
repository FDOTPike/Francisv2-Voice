from __future__ import annotations

import functools
import json
import os
import sys
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

GPT_SOVITS_ROOT = r"C:\GPT-SoVITS"
os.chdir(GPT_SOVITS_ROOT)
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation\scripts")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
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

    # Use the highest-HNR clips in the entire dataset (3.71s HNR=+2.14dB, 4.09s HNR=+2.89dB, 5.78s SV=0.9326, 6.08s)
    best_ref_names = [
        "francis_v2_raw.wav_0000609280_0000728000.wav",  # 3.71s, HNR=2.14dB, CPP=0.744 (#1 highest clarity)
        "francis_v2_raw.wav_0012707840_0012892800.wav",  # 5.78s, SV=0.9326
        "francis_v2_raw.wav_0000955840_0001150400.wav",  # 6.08s, HNR=0.48dB, SNR=34.8dB
        "francis_v2_raw.wav_0001439680_0001611200.wav",  # 5.36s, HNR=0.79dB
    ]

    ref_dir = r"C:\Users\fpike\VoiceCreation\ref"
    os.makedirs(ref_dir, exist_ok=True)

    primary_name = best_ref_names[0]
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
    for idx, aname in enumerate(best_ref_names[1:], start=1):
        asrc = os.path.join(studio_dir, aname)
        if os.path.exists(asrc):
            y_a, sr_a = sf.read(asrc, dtype="float32")
            if sr_a != 32000:
                y_a = signal.resample_poly(y_a, 32000, sr_a).astype(np.float32)
            ap = os.path.join(ref_dir, f"ref_aux_{idx}.wav")
            sf.write(ap, y_a, 32000, subtype="PCM_16")
            aux_paths.append(ap.replace("\\", "/"))

    # Initialize v2ProPlus TTS pipeline
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2ProPlus"
    tts_config.t2s_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2-e15.ckpt"
    tts_config.vits_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2_e8_s344.pth"

    tts = TTS(tts_config)

    # Lower VITS prior noise_scale from 0.50 -> 0.25 to suppress stochastic latent phase fuzz in vowels
    orig_decode = tts.vits_model.decode

    @functools.wraps(orig_decode)
    def clean_decode(*args, **kwargs):
        kwargs["noise_scale"] = 0.25
        return orig_decode(*args, **kwargs)

    tts.vits_model.decode = clean_decode

    norm_target = (
        "G'day! This is Francis version two running locally on G P T So Vits. "
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
        "top_p": 0.85,
        "temperature": 0.72,
        "text_split_method": "cut4",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.35,
        "seed": 42,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "streaming_mode": False,
    }

    sr, audio_32k = next(tts.run(inputs))
    audio_float = audio_32k.astype(np.float32) / 32768.0 if audio_32k.dtype == np.int16 else audio_32k.astype(np.float32)

    # Apply AP_BWE Neural Audio Super-Resolution (32kHz -> 48kHz Bandwidth Extension)
    print("Applying AP_BWE 48kHz neural bandwidth extension...")
    tts.init_sr_model()
    audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).to("cuda")
    audio_48k, sr_48k = tts.sr_model(audio_tensor, orig_sampling_rate=sr)
    audio_48k = np.clip(audio_48k / (np.max(np.abs(audio_48k)) + 1e-8) * 0.88, -0.98, 0.98).astype(np.float32)

    out_path = r"C:\Users\fpike\VoiceCreation\samples\iterations\iter3_sample.wav"
    sf.write(out_path, audio_48k, sr_48k, subtype="PCM_16")
    print(f"Saved Iteration 3 sample to {out_path} (sr={sr_48k}, len={len(audio_48k)/sr_48k:.2f}s)")

    res = evaluate_wav(out_path, "Iteration 3 (Low-Noise Prior Decode noise_scale=0.25 + High-HNR Ref + AP_BWE 48kHz)")
    print("=== ITERATION 3 EVALUATION ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
