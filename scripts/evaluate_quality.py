from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torchaudio

sys.path.append(r"C:\GPT-SoVITS")
sys.path.append(r"C:\GPT-SoVITS\GPT_SoVITS")
sys.path.append(r"C:\GPT-SoVITS\GPT_SoVITS\eres2net")

from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi

_SV_MODEL = None
_CENTROID_EMB = None
_WHISPER_MODEL = None

TARGET_SENTENCE = (
    "G'day! This is Francis v2 running locally on GPT-SoVITS. "
    "Feel free to use this voice in your own projects as long as you credit Francisv2."
)


def get_sv_model(device: str = "cuda"):
    global _SV_MODEL
    if _SV_MODEL is None:
        sv_path = r"C:\GPT-SoVITS\GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt"
        pretrained_state = torch.load(sv_path, map_location="cpu")
        model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        model.load_state_dict(pretrained_state)
        model.eval().to(device)
        _SV_MODEL = model
    return _SV_MODEL


def compute_sv_embedding(wav_np: np.ndarray, sr: int, device: str = "cuda") -> np.ndarray:
    model = get_sv_model(device)
    with torch.no_grad():
        wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0).to(device)
        if sr != 16000:
            wav_t = torchaudio.transforms.Resample(sr, 16000).to(device)(wav_t)
        feat = torch.stack(
            [Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav_t]
        )
        emb = model.forward(feat)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0).cpu().numpy()


def get_speaker_centroid(device: str = "cuda") -> np.ndarray:
    """Computes the ground-truth speaker embedding centroid from Francis's top-10 cleanest raw clips."""
    global _CENTROID_EMB
    if _CENTROID_EMB is not None:
        return _CENTROID_EMB

    top_clips = [
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0000609280_0000728000.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0000312320_0000443200.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0012707840_0012892800.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0000955840_0001150400.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0014018560_0014187520.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0014738880_0014886720.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0001439680_0001611200.wav",
        r"C:\GPT-SoVITS\output\slicer_opt\francis_v2_raw.wav_0014218560_0014354880.wav",
    ]
    embs = []
    for p in top_clips:
        if os.path.exists(p):
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            embs.append(compute_sv_embedding(y, sr, device))
    centroid = np.mean(np.stack(embs, axis=0), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    _CENTROID_EMB = centroid
    return _CENTROID_EMB


def evaluate_asr_clarity(wav_path: str) -> dict:
    """Uses Faster-Whisper large-v3-turbo to measure intelligibility and articulation confidence."""
    global _WHISPER_MODEL
    from faster_whisper import WhisperModel

    if _WHISPER_MODEL is None:
        model_path = r"C:\GPT-SoVITS\tools\asr\models\faster-whisper-large-v3-turbo"
        _WHISPER_MODEL = WhisperModel(model_path, device="cuda", compute_type="float16")

    segments, _ = _WHISPER_MODEL.transcribe(wav_path, beam_size=5, language="en")
    segs = list(segments)
    if not segs:
        return {"transcript": "", "avg_logprob": -2.0, "no_speech_prob": 1.0}
    text = " ".join(s.text.strip() for s in segs).strip()
    avg_logprob = float(np.mean([s.avg_logprob for s in segs]))
    no_speech_prob = float(np.mean([s.no_speech_prob for s in segs]))
    return {
        "transcript": text,
        "avg_logprob": round(avg_logprob, 4),
        "no_speech_prob": round(no_speech_prob, 5),
    }


def clamp(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return float(max(lo, min(hi, val)))


def _word_error_rate(ref_text: str, hyp_text: str) -> float:
    def _norm(s: str) -> list[str]:
        s = s.lower().replace("gpt-sovits", "gpt sovits").replace("g p t so vits", "gpt sovits")
        s = s.replace("francisv2", "francis v2").replace("francis version two", "francis v2")
        return re.findall(r"[a-z0-9']+", s)

    r = _norm(ref_text)
    h = _norm(hyp_text)
    if not r:
        return 0.0
    dp = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        dp[i, 0] = i
    for j in range(len(h) + 1):
        dp[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len(r), len(h)] / len(r))


def compute_unbiased_diagnostics(y: np.ndarray, sr: int, transcript: str = "") -> dict:
    """
    Unbiased acoustic diagnostics per GPT 6 Astra Audit:
      - Excludes outer exact-zero padding before computing pause floor
      - Uses Praat autocorrelation pitch & harmonicity (no FFT/sample-rate bias)
      - Measures stationary voiced valley-to-harmonic power density in 0.5-4 kHz and 4-8 kHz
      - Computes normalized Word Error Rate (WER) against TARGET_SENTENCE
    """
    nz = np.flatnonzero(y != 0)
    y_trim = y[nz[0] : nz[-1] + 1] if len(nz) > 0 else y
    snd = parselmouth.Sound(y_trim.astype(np.float64), float(sr))
    pitch = snd.to_pitch_ac(
        time_step=0.01, pitch_floor=65.0, pitch_ceiling=350.0, very_accurate=True, voicing_threshold=0.5
    )
    f0 = pitch.selected_array["frequency"]
    strength = pitch.selected_array["strength"]
    hnr = snd.to_harmonicity_ac(time_step=0.01, minimum_pitch=65.0, silence_threshold=0.03, periods_per_window=4.5)
    hv = hnr.values[0]
    hv = hv[(hv > -100) & np.isfinite(hv)]
    good_p = (f0 > 0) & (strength >= 0.6)
    adj = good_p[1:] & good_p[:-1]
    steps = np.abs(12.0 * np.log2(np.maximum(f0[1:], 1.0) / np.maximum(f0[:-1], 1.0)))[adj]

    # Stationary inter-harmonic valley density (60 ms window, 10 ms hop)
    n_win, hop = round(0.06 * sr), round(0.01 * sr)
    frames_60 = np.lib.stride_tricks.sliding_window_view(y_trim, n_win)[::hop]
    t_60 = (np.arange((len(y_trim) - n_win) // hop + 1) * hop + n_win / 2) / sr
    hi = np.clip(np.round((t_60 - pitch.x1) / pitch.dx).astype(int), 0, max(pitch.nx - 1, 0))
    hp = f0[hi]
    hs = strength[hi]
    rms_60_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames_60**2, axis=1)), 1e-15))
    thresh_db = float(np.percentile(rms_60_db, 95) - 30.0)
    st_mask = np.r_[False, np.abs(np.diff(hp)) < 2.0] & np.r_[np.abs(np.diff(hp)) < 2.0, False]
    good_st = (hp > 0) & (hs > 0.7) & st_mask & (rms_60_db > thresh_db)

    nfft = 2 ** int(np.ceil(np.log2(n_win * 4)))
    hz = np.fft.rfftfreq(nfft, 1.0 / sr)
    hspec = np.abs(np.fft.rfft(frames_60[good_st] * np.hanning(n_win), n=nfft, axis=1)) ** 2
    valleys = {}
    for lo, hi_f in [(500, 4000), (4000, 8000)]:
        ratios = []
        for row, fund in zip(hspec, hp[good_st]):
            dist = np.abs(hz / fund - np.round(hz / fund))
            band = (hz >= lo) & (hz < hi_f)
            harm = band & (dist <= 0.18)
            inter = band & (dist >= 0.32)
            ratios.append(10.0 * np.log10((np.mean(row[inter]) + 1e-20) / (np.mean(row[harm]) + 1e-20)))
        valleys[f"valley_to_harmonic_{lo}_{hi_f}_median_db"] = round(float(np.median(ratios)), 2) if ratios else None

    quiet_frames = rms_60_db[rms_60_db <= thresh_db]
    return {
        "praat_hnr_median_db": round(float(np.median(hv)), 2) if len(hv) else None,
        "f0_median_hz": round(float(np.median(f0[good_p])), 1) if np.any(good_p) else None,
        "adjacent_f0_steps_over_3st_pct": round(float(100.0 * np.mean(steps > 3.0)), 2) if len(steps) else 0.0,
        **valleys,
        "interior_quiet_median_dbfs": round(float(np.median(quiet_frames)), 2) if len(quiet_frames) else None,
        "leading_exact_zero_s": round(float(nz[0] / sr), 3) if len(nz) else 0.0,
        "trailing_exact_zero_s": round(float((len(y) - 1 - nz[-1]) / sr), 3) if len(nz) else 0.0,
        "normalized_wer": round(_word_error_rate(TARGET_SENTENCE, transcript), 4) if transcript.strip() else 1.0,
    }


def evaluate_wav(wav_path: str, label: str = "") -> dict:
    """
    Computes objective studio speech quality metrics and a calibrated 0-10 Studio Quality Score
    where Baseline v1.0.0 (`iter0_baseline.wav`) is anchored at ~5.0 / 10.0 and 10.0 is studio master quality.
    """
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    duration = len(y) / sr
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame_len),
        strides=(y.strides[0] * hop_len, y.strides[0]),
    )
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms)

    # 1. Dynamic Range / Silence Floor SNR (92nd percentile active speech vs 8th percentile silence floor)
    p92_db = float(np.percentile(rms_db, 92))
    p08_db = float(np.percentile(rms_db, 8))
    snr_db = p92_db - p08_db

    # Active voiced speech frames (top 45% RMS energy)
    active_mask = rms_db > np.percentile(rms_db, 55)
    active_frames = frames[active_mask] if np.any(active_mask) else frames
    window = np.hanning(frame_len)
    n_fft = 2048 if sr <= 32000 else 4096
    spec = np.abs(np.fft.rfft(active_frames * window, n=n_fft, axis=1)) + 1e-10
    power = spec**2
    log_power = 20 * np.log10(spec)

    # 2. Cepstral Peak Prominence (CPP) & Harmonic-to-Noise Ratio (HNR)
    cepstrum = np.abs(np.fft.irfft(log_power, n=n_fft, axis=1))
    q_min = int(sr / 340.0)
    q_max = min(int(sr / 65.0), cepstrum.shape[1] // 2)
    peak_vals = np.max(cepstrum[:, q_min:q_max], axis=1)
    baseline_vals = np.median(cepstrum[:, q_min:q_max], axis=1)
    cpp_db = float(np.mean(peak_vals - baseline_vals))

    ac = np.fft.irfft(power, n=n_fft, axis=1)
    ac_norm = ac / (ac[:, :1] + 1e-12)
    r_max = np.clip(np.max(ac_norm[:, q_min:q_max], axis=1), 0.01, 0.999)
    hnr_db = float(np.mean(10 * np.log10(r_max / (1.0 - r_max))))

    # 3. Spectral Presence & High-Frequency Fuzz Analysis
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    low_mid_mask = (freqs >= 100) & (freqs < 1500)
    presence_mask = (freqs >= 2000) & (freqs <= 8000)
    hf_mask = (freqs >= 4000) & (freqs <= 12000)

    e_low_mid = float(np.mean(np.sum(power[:, low_mid_mask], axis=1))) + 1e-12
    e_presence = float(np.mean(np.sum(power[:, presence_mask], axis=1))) + 1e-12
    presence_ratio_db = 10 * np.log10(e_presence / e_low_mid)

    # HF Spectral Flatness (lower in voiced frames = structured overtones; higher = white-noise fuzz)
    hf_power = power[:, hf_mask]
    geo_mean = np.exp(np.mean(np.log(hf_power + 1e-15), axis=1))
    arith_mean = np.mean(hf_power, axis=1) + 1e-15
    hf_flatness = float(np.mean(geo_mean / arith_mean))

    # 95% Spectral Rolloff
    cumsum_e = np.cumsum(power, axis=1)
    tot_e = cumsum_e[:, -1:] + 1e-12
    rolloff_idx = np.argmax(cumsum_e >= 0.95 * tot_e, axis=1)
    rolloff_hz = float(np.mean(freqs[rolloff_idx]))

    # 4. Speaker Verification Similarity vs Clean Centroid
    centroid = get_speaker_centroid("cuda")
    emb = compute_sv_embedding(y, sr, "cuda")
    sv_sim = float(np.dot(emb, centroid))

    # 5. ASR Intelligibility & Articulation Confidence
    asr_res = evaluate_asr_clarity(wav_path)

    # ================= CALIBRATED 0-10 SUB-SCORES =================
    # Subscore 1: Harmonic Clarity & Periodicity (30% weight)
    # Baseline: HNR = -0.67 dB, CPP = 0.566 -> ~4.4 / 10
    # Studio Target: HNR >= +3.5 dB, CPP >= 0.68 -> 9.0 - 10.0 / 10
    s_hnr = clamp(5.0 + (hnr_db - (-0.67)) * 1.05, 1.0, 10.0)
    s_cpp = clamp(4.0 + (cpp_db - 0.566) * 42.0, 1.0, 10.0)
    score_harmonic = 0.55 * s_hnr + 0.45 * s_cpp

    # Subscore 2: De-Fuzz & Silence Floor Purity (25% weight)
    # Baseline: hf_flatness = 0.0050, SNR = 41.3 dB -> ~4.5 / 10
    # Lower hf_flatness during voiced speech = less white-noise vocoder fuzz
    s_fuzz = clamp(9.5 - (hf_flatness / 0.0050) * 5.0, 1.0, 10.0)
    s_snr = clamp(4.5 + (snr_db - 41.0) * 0.22, 1.0, 10.0)
    score_defuzz = 0.60 * s_fuzz + 0.40 * s_snr

    # Subscore 3: Studio Bandwidth & Presence Balance (20% weight)
    # Baseline: rolloff = 1615 Hz (muffled), sr = 32000 Hz -> ~4.2 / 10
    # Studio sweet spot: rolloff 3000-5500 Hz, presence_ratio_db around -13 to -7 dB, sr = 48000 Hz
    s_rolloff = clamp(3.8 + (min(rolloff_hz, 5200.0) - 1615.0) / 550.0, 1.0, 10.0)
    s_sr_bonus = 1.0 if sr >= 48000 else 0.0
    s_presence = clamp(9.2 - abs(presence_ratio_db - (-10.5)) * 0.45, 1.0, 10.0)
    score_bandwidth = clamp(0.50 * s_rolloff + 0.40 * s_presence + s_sr_bonus, 1.0, 10.0)

    # Subscore 4: Speaker Timbre Fidelity (15% weight)
    # Baseline vs clean centroid: ~0.78 -> ~6.5 / 10; >=0.85 -> 9.2+ / 10
    score_speaker = clamp(6.2 + (sv_sim - 0.78) * 40.0, 1.0, 10.0)

    # Subscore 5: Articulation & Whisper Confidence (10% weight)
    # avg_logprob: -0.35 -> 6.5, -0.15 -> 9.2+
    score_asr = clamp(9.8 + (asr_res["avg_logprob"] + 0.10) * 10.0, 1.0, 10.0)

    overall_score = (
        0.30 * score_harmonic
        + 0.25 * score_defuzz
        + 0.20 * score_bandwidth
        + 0.15 * score_speaker
        + 0.10 * score_asr
    )

    result = {
        "label": label or os.path.basename(wav_path),
        "wav_path": wav_path,
        "sample_rate": sr,
        "duration_s": round(duration, 2),
        "metrics": {
            "hnr_db": round(hnr_db, 2),
            "cpp_db": round(cpp_db, 3),
            "snr_db": round(snr_db, 2),
            "hf_flatness_fuzz": round(hf_flatness, 5),
            "rolloff_95_hz": round(rolloff_hz, 1),
            "presence_ratio_db": round(presence_ratio_db, 2),
            "sv_cosine_sim": round(sv_sim, 4),
            "whisper_avg_logprob": asr_res["avg_logprob"],
            "whisper_transcript": asr_res["transcript"],
        },
        "subscores_out_of_10": {
            "1_harmonic_clarity": round(score_harmonic, 2),
            "2_defuzz_and_snr": round(score_defuzz, 2),
            "3_studio_bandwidth_presence": round(score_bandwidth, 2),
            "4_speaker_timbre_fidelity": round(score_speaker, 2),
            "5_articulation_intelligibility": round(score_asr, 2),
        },
        "overall_quality_score": round(overall_score, 2),
    }
    if "compute_unbiased_diagnostics" in globals():
        result["unbiased_diagnostics"] = compute_unbiased_diagnostics(y, sr, asr_res["transcript"])
    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\fpike\VoiceCreation\samples\francisv2_sample.wav"
    lbl = sys.argv[2] if len(sys.argv) > 2 else "Sample"
    res = evaluate_wav(target, lbl)
    print(json.dumps(res, indent=2))
