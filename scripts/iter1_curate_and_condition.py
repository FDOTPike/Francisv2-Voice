from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import numpy as np
import soundfile as sf
import scipy.signal as signal

sys.path.append(r"C:\Users\fpike\VoiceCreation\scripts")
from evaluate_quality import compute_sv_embedding, get_speaker_centroid, evaluate_wav


def studio_condition_audio(
    y: np.ndarray,
    sr: int,
    mud_cut_db: float = -3.5,
    presence_boost_db: float = 4.2,
    air_boost_db: float = 3.0,
    denoise_strength: float = 1.4,
) -> np.ndarray:
    """
    Applies studio vocal conditioning to remove low-mid mud, strip inter-harmonic room/AAC fuzz,
    and restore 2.5 - 8 kHz vocal presence and clarity.
    """
    y = y.astype(np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 1. 85 Hz 2nd-order Butterworth high-pass filter (removes rumble & plosive mud)
    sos_hp = signal.butter(2, 85.0 / (sr / 2.0), btype="highpass", output="sos")
    y = signal.sosfiltfilt(sos_hp, y)

    # 2. STFT domain EQ + Spectral Wiener De-Fuzz
    n_fft = 2048
    hop = 512
    freqs, times, Zxx = signal.stft(y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise floor per frequency bin from the quietest 12% of frames
    frame_energy = np.sum(mag**2, axis=0)
    quiet_thresh = np.percentile(frame_energy, 12)
    quiet_frames = mag[:, frame_energy <= quiet_thresh]
    if quiet_frames.shape[1] >= 2:
        noise_profile = np.median(quiet_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.percentile(mag, 10, axis=1, keepdims=True)

    # Soft Wiener spectral gating (stronger in >1.5 kHz where m4a/vocoder fuzz lives)
    freq_weight = np.clip((freqs - 800.0) / 3000.0, 0.35, 1.0)[:, None]
    snr_post = (mag**2) / (denoise_strength * freq_weight * (noise_profile**2) + 1e-12)
    wiener_gain = np.clip(snr_post / (1.0 + snr_post), 0.10, 1.0)
    mag_clean = mag * wiener_gain

    # Parametric Studio EQ curve in frequency domain:
    # - Cut boxiness/mud around 240 Hz (Gaussian bell in log-freq)
    # - Boost vocal presence around 3200 Hz
    # - Smooth air shelf 5000 - 11000 Hz, gentle roll-off above 13500 Hz
    f_safe = np.maximum(freqs, 20.0)
    mud_bell = mud_cut_db * np.exp(-0.5 * ((np.log2(f_safe / 240.0)) / 0.65) ** 2)
    presence_bell = presence_boost_db * np.exp(-0.5 * ((np.log2(f_safe / 3300.0)) / 0.85) ** 2)
    air_shelf = air_boost_db / (1.0 + np.exp(-(freqs - 5200.0) / 1200.0))
    hf_rolloff = -6.0 / (1.0 + np.exp(-(freqs - min(13800.0, sr * 0.44)) / 900.0))

    total_eq_db = mud_bell + presence_bell + air_shelf + hf_rolloff
    eq_gain = (10.0 ** (total_eq_db / 20.0))[:, None]

    Zxx_out = (mag_clean * eq_gain) * np.exp(1j * phase)
    _, y_out = signal.istft(Zxx_out, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y_out = y_out[: len(y)]

    # 3. Gentle soft-knee dynamic leveling + -1.2 dBTP true peak normalization
    rms = np.sqrt(np.mean(y_out**2) + 1e-12)
    target_rms = 10.0 ** (-18.5 / 20.0)
    y_out = y_out * (target_rms / rms)
    # Soft tanh saturation above 0.75 to prevent harsh clipping
    peak = np.max(np.abs(y_out))
    if peak > 0.87:
        y_out = np.tanh(y_out / 0.87) * 0.87

    return y_out.astype(np.float32)


def score_raw_clip(wav_path: str, centroid: np.ndarray) -> dict:
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    dur = len(y) / sr
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame_len),
        strides=(y.strides[0] * hop_len, y.strides[0]),
    )
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12))
    snr_db = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))

    active = frames[rms_db > np.percentile(rms_db, 50)]
    if len(active) == 0:
        active = frames
    spec = np.abs(np.fft.rfft(active * np.hanning(frame_len), n=2048, axis=1)) + 1e-10
    ac = np.fft.irfft(spec**2, n=2048, axis=1)
    ac_norm = ac / (ac[:, :1] + 1e-12)
    q_min, q_max = int(sr / 340.0), int(sr / 65.0)
    r_max = np.clip(np.max(ac_norm[:, q_min:q_max], axis=1), 0.01, 0.999)
    hnr_db = float(np.mean(10 * np.log10(r_max / (1.0 - r_max))))

    cep = np.abs(np.fft.irfft(20 * np.log10(spec), n=2048, axis=1))
    cpp_db = float(np.mean(np.max(cep[:, q_min:q_max], axis=1) - np.median(cep[:, q_min:q_max], axis=1)))

    emb = compute_sv_embedding(y, sr, "cuda")
    sv_sim = float(np.dot(emb, centroid))

    composite = hnr_db + 0.35 * snr_db + 18.0 * cpp_db + 15.0 * sv_sim
    return {
        "path": wav_path,
        "name": os.path.basename(wav_path),
        "duration": round(dur, 2),
        "snr_db": round(snr_db, 2),
        "hnr_db": round(hnr_db, 2),
        "cpp_db": round(cpp_db, 3),
        "sv_sim": round(sv_sim, 4),
        "composite": round(composite, 2),
    }


def main():
    asr_list_path = r"C:\GPT-SoVITS\output\asr_opt\slicer_opt.list"
    slicer_dir = r"C:\GPT-SoVITS\output\slicer_opt"
    conditioned_dir = r"C:\GPT-SoVITS\output\slicer_opt_studio"
    os.makedirs(conditioned_dir, exist_ok=True)

    # Load ASR transcripts
    transcripts = {}
    with open(asr_list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 4:
                wpath, spk, lang, txt = parts
                transcripts[os.path.basename(wpath)] = txt.strip()

    centroid = get_speaker_centroid("cuda")
    clip_paths = sorted(glob.glob(os.path.join(slicer_dir, "*.wav")))
    scored_clips = []
    for cp in clip_paths:
        bname = os.path.basename(cp)
        txt = transcripts.get(bname, "")
        if len(txt.split()) < 4:
            continue
        info = score_raw_clip(cp, centroid)
        info["text"] = txt
        scored_clips.append(info)

    scored_clips.sort(key=lambda x: x["composite"], reverse=True)
    print(f"Scored {len(scored_clips)} clips with valid transcripts.")
    print("Top 8 cleanest clips for Reference & Training:")
    for item in scored_clips[:8]:
        print(
            f"  {item['name']} ({item['duration']}s): score={item['composite']}, "
            f"HNR={item['hnr_db']}dB, SNR={item['snr_db']}dB, SV={item['sv_sim']} | {item['text']!r}"
        )

    # Keep top 75 highest-clarity clips (filtering out the bottom noisy/muffled segments)
    kept_clips = [c for c in scored_clips[:75] if c["snr_db"] >= 18.0 and c["sv_sim"] >= 0.72]
    print(f"Selected {len(kept_clips)} studio-grade clips (rejected {len(scored_clips) - len(kept_clips)} noisy clips).")

    # Condition and save all kept clips to C:\GPT-SoVITS\output\slicer_opt_studio
    curated_lines = []
    total_dur = 0.0
    for c in kept_clips:
        y, sr = sf.read(c["path"], dtype="float32")
        y_cond = studio_condition_audio(y, sr)
        out_path = os.path.join(conditioned_dir, c["name"])
        sf.write(out_path, y_cond, sr, subtype="PCM_16")
        curated_lines.append(f"{out_path}|Francisv2|en|{c['text']}")
        total_dur += c["duration"]

    studio_list_path = r"C:\GPT-SoVITS\output\asr_opt\francisv2_studio.list"
    with open(studio_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(curated_lines) + "\n")
    print(f"Wrote {len(curated_lines)} studio-conditioned clips ({total_dur / 60.0:.2f} mins) to {studio_list_path}")

    # Pick the best primary reference clip (duration between 4.5s and 7.0s) and top 3 aux reference clips
    ref_candidates = [c for c in kept_clips if 4.5 <= c["duration"] <= 7.0]
    best_ref = ref_candidates[0]
    aux_refs = ref_candidates[1:4]

    ref_dir = r"C:\Users\fpike\VoiceCreation\ref"
    os.makedirs(ref_dir, exist_ok=True)

    # Save conditioned primary ref.wav (32kHz PCM_16)
    y_ref, sr_ref = sf.read(os.path.join(conditioned_dir, best_ref["name"]), dtype="float32")
    if sr_ref != 32000:
        y_ref = signal.resample_poly(y_ref, 32000, sr_ref).astype(np.float32)
    sf.write(os.path.join(ref_dir, "ref.wav"), y_ref, 32000, subtype="PCM_16")
    sf.write(r"C:\GPT-SoVITS\ref.wav", y_ref, 32000, subtype="PCM_16")
    with open(os.path.join(ref_dir, "ref_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(best_ref["text"] + "\n")
    with open(r"C:\GPT-SoVITS\ref_transcript.txt", "w", encoding="utf-8") as f:
        f.write(best_ref["text"] + "\n")

    aux_paths = []
    for idx, aux in enumerate(aux_refs, start=1):
        y_a, sr_a = sf.read(os.path.join(conditioned_dir, aux["name"]), dtype="float32")
        if sr_a != 32000:
            y_a = signal.resample_poly(y_a, 32000, sr_a).astype(np.float32)
        ap = os.path.join(ref_dir, f"ref_aux_{idx}.wav")
        sf.write(ap, y_a, 32000, subtype="PCM_16")
        aux_paths.append(ap.replace("\\", "/"))

    meta = {
        "primary_ref": best_ref,
        "aux_refs": aux_refs,
        "aux_ref_paths": aux_paths,
    }
    with open(os.path.join(ref_dir, "ref_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Updated primary ref.wav -> {best_ref['name']} ({best_ref['duration']}s): {best_ref['text']!r}")
    print(f"Saved {len(aux_paths)} auxiliary reference clips for multi-reference tone fusion.")


if __name__ == "__main__":
    main()
