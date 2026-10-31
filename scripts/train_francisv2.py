import json
import os
import shutil
import subprocess
import sys
import wave

GPT_SOVITS_ROOT = os.environ.get("GPT_SOVITS_ROOT", r"C:\GPT-SoVITS")
RUNTIME_DIR = os.path.join(GPT_SOVITS_ROOT, "runtime")
PYTHON_EXEC = os.path.join(RUNTIME_DIR, "python.exe")
RAW_WAV = r"C:\voice_data\francis_v2_raw.wav"
EXP_NAME = "Francisv2"
VERSION = "v2"  # Uses 32000 Hz sample rate and saves into GPT_weights_v2 / SoVITS_weights_v2
TARGET_DATASET_SECONDS = 420.0  # ~7 minutes of the cleanest 3-10.5s speech slices
MAX_ASR_CLIPS = 110             # ~10 minutes of 3-10.5s clips to transcribe via Faster-Whisper


def configure_runtime_paths():
    """Writes runtime/Lib/site-packages/users.pth just like webui.py does on startup."""
    now_dir = GPT_SOVITS_ROOT.replace("\\", "/")
    users_pth = os.path.join(RUNTIME_DIR, "Lib", "site-packages", "users.pth")
    content = (
        f"{now_dir}\n"
        f"{now_dir}/GPT_SoVITS/BigVGAN\n"
        f"{now_dir}/tools\n"
        f"{now_dir}/tools/asr\n"
        f"{now_dir}/GPT_SoVITS\n"
        f"{now_dir}/tools/uvr5\n"
    )
    with open(users_pth, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Init] Configured {users_pth} for {now_dir}")


def run_step(cmd: list[str], env: dict[str, str] | None = None, desc: str = ""):
    print(f"\n=== {desc} ===")
    print("Command:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    merged_env = os.environ.copy()
    merged_env["PATH"] = f"{RUNTIME_DIR};{merged_env.get('PATH', '')}"
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONUTF8"] = "1"
    merged_env["TEMP"] = os.path.join(GPT_SOVITS_ROOT, "TEMP")
    os.makedirs(merged_env["TEMP"], exist_ok=True)
    if env:
        merged_env.update(env)
    res = subprocess.run(cmd, cwd=GPT_SOVITS_ROOT, env=merged_env)
    if res.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with exit code {res.returncode}")


def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def limit_slicer_clips_for_fast_asr(slicer_opt: str):
    """
    Keeps the best MAX_ASR_CLIPS (3.5s - 10.0s) in slicer_opt so Faster-Whisper
    transcribes ~9 minutes of clean speech in ~30s instead of all 28 minutes,
    archiving the rest into output/slicer_opt_backup.
    """
    backup_dir = os.path.join(GPT_SOVITS_ROOT, "output", "slicer_opt_backup")
    os.makedirs(backup_dir, exist_ok=True)
    wavs = sorted(f for f in os.listdir(slicer_opt) if f.endswith(".wav"))
    if len(wavs) <= MAX_ASR_CLIPS:
        return

    kept = 0
    moved = 0
    for fname in wavs:
        fpath = os.path.join(slicer_opt, fname)
        dur = get_wav_duration(fpath)
        if 3.5 <= dur <= 10.0 and kept < MAX_ASR_CLIPS:
            kept += 1
        else:
            shutil.move(fpath, os.path.join(backup_dir, fname))
            moved += 1
    print(f"[Pre-ASR Filter] Kept {kept} ideal 3.5-10.0s clips in {slicer_opt} (archived {moved} extra clips to {backup_dir}).")


def curate_and_pick_reference(asr_list_path: str, curated_list_path: str):
    """
    Reads Faster-Whisper's slicer_opt.list:
      wav_path|speaker|lang|text
    Filters to clean 3.0s - 10.5s spoken clips up to ~7 minutes total for fast, high-fidelity
    training on the 8 GB RTX 3070 Ti, and selects the best 5-7s clip as C:/GPT-SoVITS/ref.wav.
    """
    with open(asr_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    valid_entries = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        wav_path, speaker, lang, text = parts[0], parts[1], parts[2], "|".join(parts[3:]).strip()
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(GPT_SOVITS_ROOT, wav_path)
        if not os.path.exists(wav_path):
            continue
        if len(text) < 8 or text.startswith("[") or text.startswith("("):
            continue
        dur = get_wav_duration(wav_path)
        if 3.0 <= dur <= 10.5:
            valid_entries.append((dur, wav_path, speaker, lang, text, line))

    if not valid_entries:
        raise RuntimeError("No valid 3-10.5s transcribed audio slices found in slicer_opt.list!")

    # Pick the best 5.0 - 7.5s clip with natural word density (~2.0 to 3.6 words/sec) for ref.wav
    ref_candidates = []
    for dur, wav_path, speaker, lang, text, raw_line in valid_entries:
        words = len(text.split())
        wps = words / dur
        if 4.5 <= dur <= 7.8 and 1.8 <= wps <= 3.8 and text.endswith((".", "!", "?")):
            score = -abs(dur - 6.0) - abs(wps - 2.6)
            ref_candidates.append((score, dur, wav_path, text))

    if ref_candidates:
        ref_candidates.sort(reverse=True)
        _, ref_dur, best_ref_wav, best_ref_text = ref_candidates[0]
    else:
        best_ref_wav = valid_entries[0][1]
        ref_dur = valid_entries[0][0]
        best_ref_text = valid_entries[0][4]

    ref_dest = os.path.join(GPT_SOVITS_ROOT, "ref.wav")
    shutil.copy2(best_ref_wav, ref_dest)
    ref_txt_dest = os.path.join(GPT_SOVITS_ROOT, "ref_transcript.txt")
    with open(ref_txt_dest, "w", encoding="utf-8") as f:
        f.write(best_ref_text)

    print(f"\n[Reference Clip Selected]")
    print(f"  File:       {ref_dest} ({ref_dur:.2f}s, from {os.path.basename(best_ref_wav)})")
    print(f"  Transcript: {best_ref_text!r}")

    # Curate up to TARGET_DATASET_SECONDS (~7 minutes) of clean clips
    curated_lines = []
    accumulated_sec = 0.0
    for dur, wav_path, speaker, lang, text, raw_line in valid_entries:
        curated_lines.append(f"{wav_path}|{EXP_NAME}|en|{text}")
        accumulated_sec += dur
        if accumulated_sec >= TARGET_DATASET_SECONDS:
            break

    with open(curated_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(curated_lines) + "\n")

    print(
        f"[Dataset Curated] Selected {len(curated_lines)} clips "
        f"({accumulated_sec / 60.0:.2f} minutes total) -> {curated_list_path}"
    )


def main():
    if not os.path.exists(PYTHON_EXEC):
        raise FileNotFoundError(f"GPT-SoVITS runtime python not found at {PYTHON_EXEC}")
    if not os.path.exists(RAW_WAV):
        raise FileNotFoundError(f"Normalized WAV not found at {RAW_WAV}")

    configure_runtime_paths()

    slicer_opt = os.path.join(GPT_SOVITS_ROOT, "output", "slicer_opt")
    asr_opt = os.path.join(GPT_SOVITS_ROOT, "output", "asr_opt")
    os.makedirs(slicer_opt, exist_ok=True)
    os.makedirs(asr_opt, exist_ok=True)

    # 1. Step 0a: Audio Slicing
    existing_wavs = [f for f in os.listdir(slicer_opt) if f.endswith(".wav")]
    if len(existing_wavs) < 10:
        run_step(
            [
                PYTHON_EXEC,
                "-s",
                "tools/slice_audio.py",
                RAW_WAV,
                slicer_opt,
                "-34",
                "4000",
                "300",
                "10",
                "500",
                "0.9",
                "0.25",
                "0",
                "1",
            ],
            desc="Phase 2.3 (Tab 0a): Slicing francis_v2_raw.wav into 4-10s clips",
        )
    else:
        print(f"Found {len(existing_wavs)} existing sliced clips in {slicer_opt}.")

    limit_slicer_clips_for_fast_asr(slicer_opt)

    # 2. Step 0b: Faster-Whisper ASR Transcription
    asr_list_path = os.path.join(asr_opt, "slicer_opt.list")
    if not os.path.exists(asr_list_path) or os.path.getsize(asr_list_path) < 100:
        run_step(
            [
                PYTHON_EXEC,
                "tools/asr/fasterwhisper_asr.py",
                "-i", slicer_opt,
                "-o", asr_opt,
                "-s", "large-v3-turbo",
                "-l", "en",
                "-p", "float16",
            ],
            desc="Phase 2.4 (Tab 0b): Auto-Transcribing Clips with Faster-Whisper (large-v3-turbo, float16)",
        )

    # 3. Step 0c: Curate dataset & select reference clip
    curated_list_path = os.path.join(asr_opt, "francisv2_curated.list")
    curate_and_pick_reference(asr_list_path, curated_list_path)

    # 4. Step 1A: One-Click Dataset Formatting (1a: BERT text, 1b: HuBERT + 32k WAV, 1c: Semantic tokens)
    opt_dir = os.path.join(GPT_SOVITS_ROOT, "logs", EXP_NAME)
    os.makedirs(opt_dir, exist_ok=True)

    bert_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ssl_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    s2g_pretrained = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    s2d_pretrained = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    s1_pretrained = (
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
        "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    )

    base_env = {
        "inp_text": curated_list_path,
        "inp_wav_dir": slicer_opt,
        "exp_name": EXP_NAME,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": bert_dir,
        "cnhubert_base_dir": ssl_dir,
        "pretrained_s2G": s2g_pretrained,
        "s2config_path": "GPT_SoVITS/configs/s2.json",
        "is_half": "True",
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
    }

    # 1A-1: Text/phoneme extraction
    run_step(
        [PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        env=base_env,
        desc="Phase 2.5 (Tab 1A - Step 1/3): Extracting phoneme & BERT features",
    )
    part_txt = os.path.join(opt_dir, "2-name2text-0.txt")
    final_txt = os.path.join(opt_dir, "2-name2text.txt")
    if os.path.exists(part_txt):
        with open(part_txt, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().strip("\n").split("\n") if l]
        with open(final_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.remove(part_txt)

    # 1A-2: HuBERT & 32kHz WAV extraction
    run_step(
        [PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        env=base_env,
        desc="Phase 2.5 (Tab 1A - Step 2/3): Extracting HuBERT features & 32kHz audio",
    )

    # 1A-3: Semantic token extraction
    run_step(
        [PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
        env=base_env,
        desc="Phase 2.5 (Tab 1A - Step 3/3): Extracting semantic tokens",
    )
    part_sem = os.path.join(opt_dir, "6-name2semantic-0.tsv")
    final_sem = os.path.join(opt_dir, "6-name2semantic.tsv")
    if os.path.exists(part_sem):
        with open(part_sem, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().strip("\n").split("\n") if l]
        with open(final_sem, "w", encoding="utf-8") as f:
            f.write("item_name\tsemantic_audio\n" + "\n".join(lines) + "\n")
        os.remove(part_sem)

    # 5. Step 1B-1: Fine-tune SoVITS (8 epochs, batch_size=4)
    tmp_dir = os.path.join(GPT_SOVITS_ROOT, "TEMP")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(opt_dir, f"logs_s2_{VERSION}"), exist_ok=True)
    os.makedirs(os.path.join(GPT_SOVITS_ROOT, "SoVITS_weights_v2"), exist_ok=True)
    os.makedirs(os.path.join(GPT_SOVITS_ROOT, "GPT_weights_v2"), exist_ok=True)

    with open(os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS", "configs", "s2.json"), "r", encoding="utf-8") as f:
        s2_data = json.load(f)

    s2_data["train"]["fp16_run"] = True
    s2_data["train"]["batch_size"] = 4
    s2_data["train"]["epochs"] = 8
    s2_data["train"]["text_low_lr_rate"] = 0.4
    s2_data["train"]["pretrained_s2G"] = s2g_pretrained
    s2_data["train"]["pretrained_s2D"] = s2d_pretrained
    s2_data["train"]["if_save_latest"] = True
    s2_data["train"]["if_save_every_weights"] = True
    s2_data["train"]["save_every_epoch"] = 4
    s2_data["train"]["gpu_numbers"] = "0"
    s2_data["train"]["grad_ckpt"] = False
    s2_data["train"]["lora_rank"] = "32"
    s2_data["model"]["version"] = VERSION
    s2_data["data"]["exp_dir"] = s2_data["s2_ckpt_dir"] = opt_dir
    s2_data["save_weight_dir"] = "SoVITS_weights_v2"
    s2_data["name"] = EXP_NAME
    s2_data["version"] = VERSION

    tmp_s2_json = os.path.join(tmp_dir, "tmp_s2.json")
    with open(tmp_s2_json, "w", encoding="utf-8") as f:
        json.dump(s2_data, f, indent=2)

    run_step(
        [PYTHON_EXEC, "GPT_SoVITS/s2_train.py", "--config", tmp_s2_json],
        desc="Phase 2.5 (Tab 1B): Fine-tuning SoVITS acoustic model (8 epochs, batch_size=4)",
    )

    # 6. Step 1B-2: Fine-tune GPT (15 epochs, batch_size=4)
    s1_config_helper = f"""
import os, yaml
with open("GPT_SoVITS/configs/s1longer-v2.yaml", "r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
s1_dir = {opt_dir!r}
os.makedirs(os.path.join(s1_dir, "logs_s1"), exist_ok=True)
data["train"]["batch_size"] = 4
data["train"]["epochs"] = 15
data["pretrained_s1"] = {s1_pretrained!r}
data["train"]["save_every_n_epoch"] = 5
data["train"]["if_save_every_weights"] = True
data["train"]["if_save_latest"] = True
data["train"]["if_dpo"] = False
data["train"]["half_weights_save_dir"] = "GPT_weights_v2"
data["train"]["exp_name"] = {EXP_NAME!r}
data["train_semantic_path"] = os.path.join(s1_dir, "6-name2semantic.tsv")
data["train_phoneme_path"] = os.path.join(s1_dir, "2-name2text.txt")
data["output_dir"] = os.path.join(s1_dir, "logs_s1_v2")
with open(r"{os.path.join(tmp_dir, 'tmp_s1.yaml')}", "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False)
"""
    run_step(
        [PYTHON_EXEC, "-c", s1_config_helper],
        desc="Preparing GPT (s1) training config",
    )
    run_step(
        [PYTHON_EXEC, "GPT_SoVITS/s1_train.py", "--config_file", os.path.join(tmp_dir, "tmp_s1.yaml")],
        env={"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"},
        desc="Phase 2.5 (Tab 1B): Fine-tuning GPT prosody/accent model (15 epochs, batch_size=4)",
    )

    print("\n=== Francisv2 Training Complete! ===")
    print("SoVITS weights:", os.listdir(os.path.join(GPT_SOVITS_ROOT, "SoVITS_weights_v2")))
    print("GPT weights:   ", os.listdir(os.path.join(GPT_SOVITS_ROOT, "GPT_weights_v2")))


if __name__ == "__main__":
    main()
