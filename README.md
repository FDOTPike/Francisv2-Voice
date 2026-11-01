# Francisv2 — Studio Master Australian English Voice Clone & AI Voice Agent

[![License: CC BY 4.0](https://img.shields.io/badge/Voice_Model_License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-green.svg)](./LICENSE)
[![Release](https://img.shields.io/badge/Release-v2.0.0_Studio_Master-orange.svg)](https://github.com/FDOTPike/Francisv2-Voice/releases)

**Francisv2** is an open, studio-mastered **GPT-SoVITS v2ProPlus** Australian English voice model (`ERes2NetV2` speaker verification + `s1v3` prosody + 48 kHz Formant-Preserving Harmonic Comb Mastering) and real-time AI voice agent created by **Francis Pike** (`@FDOTPike`).

It is **100% free to use** on your own computer (for personal projects, content creation, research, or commercial apps) **as long as you reference/credit the product**.

---

## Mandatory Attribution (Free to Use with Reference)

You may use the **Francisv2** voice model and code free of charge in any project provided you include the following credit in your project README, video/audio description, app credits, or documentation:

> **Plain text:**
> Voice powered by **Francisv2** by Francis Pike (https://github.com/FDOTPike/Francisv2-Voice)

```markdown
<!-- Markdown Attribution -->
Voice powered by [**Francisv2**](https://github.com/FDOTPike/Francisv2-Voice) by Francis Pike (`@FDOTPike`).
```

---

## Multi-Iteration Clarity Optimization (`4.99 / 10` → `8.60 / 10`)

The `Francisv2` pipeline was iteratively engineered across 4 generations to eliminate neural vocoder fuzziness and achieve **48,000 Hz studio broadcast clarity**:

| Iteration | Architecture & DSP Upgrades | Harmonic Clarity | De-Fuzz & SNR | Speaker Fidelity | Articulation | **Overall Score** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iteration 0 (Baseline)** | Raw `v2` (`s2G2333k`), unfiltered clips, single muffled `ref.wav`, `streaming_mode=2` (32 kHz) | `4.58 / 10` | `4.61 / 10` | `5.75 / 10` | `7.55 / 10` | **`4.99 / 10`** |
| **Iteration 1 (`v1.1`)** | STFT Wiener dataset de-fuzzing, top-75 clarity curation, multi-reference fusion (`ref_aux_1..3`), full-context decode | `6.17 / 10` | `6.30 / 10` | `8.33 / 10` | `7.93 / 10` | **`6.70 / 10`** |
| **Iteration 2 (`v1.2`)** | Upgraded to **`GPT-SoVITS v2ProPlus`** (`s2Gv2ProPlus` + `s1v3` + 20,480-dim `ERes2NetV2` Speaker Verification) | `7.53 / 10` | `5.54 / 10` | `7.92 / 10` | `8.99 / 10` | **`6.95 / 10`** |
| **Iteration 3 (`v1.3`)** | Low-noise prior decode (`noise_scale=0.25`) + high-HNR reference fusion + 48 kHz neural BWE | `6.01 / 10` | `5.45 / 10` | `10.0 / 10` | `8.38 / 10` | **`6.84 / 10`** |
| **Iteration 4 (`v2.0 Studio Master`)** | **`v2ProPlus` + Two-Pass Formant-Preserving Frequency-Domain Harmonic Comb Filter + 48 kHz Studio Broadcast EQ & Expander Gate** | **`8.63 / 10`** | **`9.48 / 10`** | **`9.23 / 10`** | **`8.47 / 10`** | **`8.60 / 10`** |

- **93.2% Reduction in High-Frequency Vocoder Fuzz (`hf_flatness_fuzz`):** `0.00546` → **`0.00037`**
- **Harmonic-to-Noise Ratio (`HNR`) & Cepstral Peak Prominence (`CPP`):** `HNR` improved from `-0.41 dB` → **`+1.72 dB`**, `CPP` improved from `0.560` → **`1.156`**
- **Dynamic Studio Silence Floor (`SNR`):** Improved from `45.4 dB` → **`104.3 dB`** at native **`48,000 Hz`**

---

## Repository Structure

```text
VoiceCreation/
├── voice_agent.py              # Studio Master TTS CLI + conversational voice agent client
├── studio_mastering.py         # 48kHz Formant-Preserving Harmonic Comb & Studio Broadcast DSP chain
├── download_weights.py         # Auto-downloader & SHA256 verifier for Francisv2 v2ProPlus weights
├── start_francisv2_server.bat  # One-click Windows launcher for GPT-SoVITS api_v2.py
├── requirements.txt            # Python client dependencies
├── ref/
│   ├── ref.wav                 # 5.78s studio-conditioned primary reference anchor (SV=0.9326)
│   ├── ref_aux_1..3.wav        # Top-3 auxiliary studio reference clips for multi-ref tone fusion
│   └── ref_transcript.txt      # Exact transcript of ref.wav
├── samples/
│   ├── francisv2_sample.wav    # 48kHz Studio Master sample audio (8.60/10 score)
│   └── iterations/             # Archived samples from Iteration 0 through Iteration 4
├── weights/                    # Local directory for pretrained weights (auto-populated)
│   ├── Francisv2-e15.ckpt      # Stage 1 AR GPT s1v3 prosody/accent model (155.3 MB)
│   └── Francisv2_e8_s344.pth   # Stage 2 SoVITS v2ProPlus + ERes2NetV2 vocoder model (172.8 MB)
└── scripts/                    # Complete training, curation, and objective quality evaluation suite
    ├── prepare_audio.py        # FFmpeg 48kHz highpass + EBU R128 loudness normalization
    ├── run_slicer.py           # Energy-based silence slicer (3.5s - 10.0s clip curator)
    ├── iter1_curate_and_condition.py # STFT Wiener de-fuzzing & clarity-ranked dataset curation
    ├── iter2_train_v2proplus.py      # Automated v2ProPlus + s1v3 + ERes2NetV2 trainer
    ├── run_iter4_studio_master.py    # End-to-end Studio Master synthesis & evaluation runner
    └── evaluate_quality.py     # Objective 0-10 Studio Quality Benchmark (HNR, CPP, SNR, SV, Whisper)
```

---

## Quick Start (Run Francisv2 on Your Computer)

### 1. Prerequisites
1. **GPT-SoVITS v2Pro (`20250604` or newer)** installed on a computer with an NVIDIA GPU (8 GB+ VRAM recommended, e.g. RTX 3070 Ti) or Apple Silicon Mac:
   - Download the official integrated package from [GPT-SoVITS Releases](https://github.com/RVC-Boss/GPT-SoVITS/releases) and extract it (e.g. to `C:\GPT-SoVITS`).
2. **Python 3.9+** for running the client agent (`voice_agent.py`).

### 2. Clone This Repository & Download the Weights
```bash
git clone https://github.com/FDOTPike/Francisv2-Voice.git
cd Francisv2-Voice
pip install -r requirements.txt

# Downloads Francisv2-e15.ckpt (155.3 MB) and Francisv2_e8_s344.pth (172.8 MB)
# from GitHub Releases, verifies SHA256 hashes, and syncs to C:\GPT-SoVITS
python download_weights.py --gpt-sovits-dir "C:\GPT-SoVITS"
```

### 3. Start the Headless GPT-SoVITS Server
On Windows, double-click **`start_francisv2_server.bat`** or run:
```bat
start_francisv2_server.bat
```

### 4. Generate 48 kHz Studio Master Speech or Run the Real-Time Voice Agent

#### Option A: Synthesize Any Text to 48 kHz Studio Master WAV (CLI)
```bash
python voice_agent.py --say "G'day! This is Francis v2 running locally on GPT-SoVITS. Feel free to use this voice in your own projects as long as you credit Francisv2." --out my_output.wav
```

#### Option B: Interactive Real-Time Conversational Voice Agent (Mic -> Whisper -> Llama 3.3 -> Francisv2)
1. Get a free API key at [console.groq.com](https://console.groq.com/keys).
2. Set `GROQ_API_KEY` in your environment and run:
```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
python voice_agent.py
```

---

## Model Specifications & Checksums (`v2.0.0 Studio Master`)

| Asset | File | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Stage 1 GPT (`s1v3` Base)** | `Francisv2-e15.ckpt` | 155.3 MB | `718A3A16F720736CCDA1FA96D1DF1F5F0319CFA5B323E14EE2E067EA5F43D6E9` |
| **Stage 2 SoVITS (`v2ProPlus` + SV)** | `Francisv2_e8_s344.pth` | 172.8 MB | `197658D82E7805F66FA7577F81870E63D386565C22ACBCCEA6F0345391182162` |
| **Primary Reference Anchor** | `ref/ref.wav` | 369 KB | 5.78s, 32 kHz studio-conditioned mono PCM (`SV = 0.9326`) |
| **Auxiliary Reference Anchors** | `ref/ref_aux_1..3.wav` | 1.02 MB | Top-3 studio-conditioned clips for multi-reference tone fusion |

---

## Citation

If you use **Francisv2** in academic work, software, or media, please cite:

```bibtex
@software{pike2026francisv2,
  author       = {Pike, Francis},
  title        = {{Francisv2: Studio Master Australian English Voice Model \& AI Voice Agent (GPT-SoVITS v2ProPlus)}},
  year         = {2026},
  version      = {2.0.0},
  url          = {https://github.com/FDOTPike/Francisv2-Voice},
  note         = {Licensed under CC BY 4.0 (Attribution Required)}
}
```
