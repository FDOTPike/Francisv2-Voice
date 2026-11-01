from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

GPT_SOVITS_ROOT = r"C:\GPT-SoVITS"
os.chdir(GPT_SOVITS_ROOT)
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation")
sys.path.insert(0, r"C:\Users\fpike\VoiceCreation\scripts")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import studio_mastering as mastering
from evaluate_quality import evaluate_wav


def main():
    root = Path(r"C:\Users\fpike\VoiceCreation")
    iter_dir = root / "samples" / "iterations"
    abl_dir = iter_dir / "ablations"
    abl_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = root / "ref"
    with open(ref_dir / "ref_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    ref_wav_path = str(ref_dir / "ref.wav").replace("\\", "/")
    aux_paths = meta["aux_ref_paths"]
    primary_text = meta["primary_ref_text"]

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2ProPlus"
    tts_config.t2s_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2-e15.ckpt"
    tts_config.vits_weights_path = r"C:/Users/fpike/VoiceCreation/weights/Francisv2_e8_s344.pth"

    tts = TTS(tts_config)

    # Match voice_agent.py / api_v2.py exact default inference settings (no hidden monkeypatch)
    norm_target = (
        "G'day, this is Francis version two running locally on G P T So Vits. "
        "Feel free to use this voice in your own projects as long as you credit Francis version two."
    )
    inputs = {
        "text": norm_target,
        "text_lang": "en",
        "ref_audio_path": ref_wav_path,
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
    y_raw = audio_raw.astype(np.float64)
    if audio_raw.dtype == np.int16 or np.max(np.abs(y_raw)) > 2.0:
        y_raw = y_raw / 32768.0
    if y_raw.ndim > 1:
        y_raw = np.mean(y_raw, axis=1)

    # Trim any leading/trailing digital zeros from raw synthesis before saving pre-master
    nz = np.flatnonzero(np.abs(y_raw) > 1e-5)
    if len(nz) > 0:
        y_raw = y_raw[nz[0] : nz[-1] + 1]

    premaster_path = iter_dir / "iter5_premaster_raw.wav"
    sf.write(premaster_path, y_raw.astype(np.float32), sr_raw, subtype="PCM_16")
    print(f"Saved Round 5 pre-master raw synthesis to {premaster_path} (sr={sr_raw}, len={len(y_raw)/sr_raw:.2f}s)")

    # Controlled Stage-by-Stage Ablation on the EXACT same pre-master waveform
    gcd = np.gcd(sr_raw, 48000)
    y_48k = signal.resample_poly(y_raw, 48000 // gcd, sr_raw // gcd)
    sos_hp = signal.butter(2, 80.0 / 24000.0, btype="highpass", output="sos")
    y_hp = signal.sosfiltfilt(sos_hp, y_48k)
    sf.write(abl_dir / "0_premaster_48k_hp.wav", y_hp.astype(np.float32), 48000, subtype="FLOAT")

    y_c1 = mastering.formant_preserving_harmonic_comb(y_hp, 48000, alpha=0.65, gamma=1.40)
    sf.write(abl_dir / "1_comb_1pass.wav", y_c1.astype(np.float32), 48000, subtype="FLOAT")

    y_c2 = mastering.formant_preserving_harmonic_comb(y_c1, 48000, alpha=0.50, gamma=1.25)
    sf.write(abl_dir / "2_comb_2pass.wav", y_c2.astype(np.float32), 48000, subtype="FLOAT")

    y_eq = mastering.studio_broadcast_eq_and_air(y_c2, 48000)
    sf.write(abl_dir / "3_comb_plus_eq.wav", y_eq.astype(np.float32), 48000, subtype="FLOAT")

    y_master = mastering.apply_studio_noise_gate_and_limiter(y_eq, 48000)
    out_path = iter_dir / "iter5_sample.wav"
    sf.write(out_path, y_master, 48000, subtype="PCM_16")
    sf.write(root / "samples" / "francisv2_sample.wav", y_master, 48000, subtype="PCM_16")
    print(f"Saved Round 5 Astra-Remediated Studio Master to {out_path} (sr=48000, len={len(y_master)/48000:.2f}s)")

    # Save per-round metrics JSON files (iter0_metrics.json .. iter5_metrics.json)
    rounds = [
        (0, iter_dir / "iter0_baseline.wav", "Round 0 Baseline (v2)"),
        (1, iter_dir / "iter1_sample.wav", "Round 1 (Curated & Conditioned References)"),
        (2, iter_dir / "iter2_sample.wav", "Round 2 (Retrained v2ProPlus + s1v3)"),
        (3, iter_dir / "iter3_sample.wav", "Round 3 (Low-Noise Prior + AP-BWE)"),
        (4, iter_dir / "iter4_sample.wav", "Round 4 (Initial Studio Master)"),
        (5, iter_dir / "iter5_sample.wav", "Round 5 (Astra-Remediated Studio Master v2.1)"),
    ]
    for r_idx, wav_p, label in rounds:
        if wav_p.exists():
            m = evaluate_wav(str(wav_p), label)
            with open(iter_dir / f"iter{r_idx}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)

    print("Exported iter0_metrics.json .. iter5_metrics.json successfully.")


if __name__ == "__main__":
    main()
