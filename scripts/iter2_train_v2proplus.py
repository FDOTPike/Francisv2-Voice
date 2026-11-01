from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import yaml
import soundfile as sf

GPT_SOVITS_ROOT = r"C:\GPT-SoVITS"
PYTHON_EXE = os.path.join(GPT_SOVITS_ROOT, "runtime", "python.exe")
EXP_NAME = "Francisv2"
VERSION = "v2ProPlus"
OPT_DIR = os.path.join(GPT_SOVITS_ROOT, "logs", EXP_NAME)
STUDIO_WAV_DIR = os.path.join(GPT_SOVITS_ROOT, "output", "slicer_opt_studio")
STUDIO_LIST_FILE = os.path.join(GPT_SOVITS_ROOT, "output", "asr_opt", "francisv2_studio.list")


def run_step(cmd: list[str], env_updates: dict[str, str], desc: str):
    print(f"\n=== [Iteration 2: {desc}] ===")
    env = os.environ.copy()
    env["PATH"] = os.path.join(GPT_SOVITS_ROOT, "runtime") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = GPT_SOVITS_ROOT + os.pathsep + os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.update(env_updates)
    proc = subprocess.run(cmd, cwd=GPT_SOVITS_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {proc.returncode}")


def main():
    # Clean previous feature extraction logs so v2ProPlus extracts fresh studio-conditioned features & SV embeddings
    if os.path.exists(OPT_DIR):
        shutil.rmtree(OPT_DIR)
    os.makedirs(OPT_DIR, exist_ok=True)

    base_env = {
        "inp_text": STUDIO_LIST_FILE,
        "inp_wav_dir": STUDIO_WAV_DIR,
        "exp_name": EXP_NAME,
        "opt_dir": OPT_DIR,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
        "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "s2config_path": "GPT_SoVITS/configs/s2v2ProPlus.json",
    }

    # 1a. Text / BERT phoneme extraction
    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], base_env, "1a-get-text")
    part_txt = os.path.join(OPT_DIR, "2-name2text-0.txt")
    final_txt = os.path.join(OPT_DIR, "2-name2text.txt")
    if os.path.exists(part_txt):
        shutil.move(part_txt, final_txt)

    # 1b. HuBERT + 32kHz studio-conditioned wavs
    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], base_env, "1b-get-hubert-wav32k")

    # 1b-pro. ERes2NetV2 Speaker Verification embeddings (7-sv_cn) for v2ProPlus
    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], base_env, "1b-get-sv (ERes2NetV2)")

    # 1c. Semantic tokens using s2Gv2ProPlus.pth
    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], base_env, "1c-get-semantic (v2ProPlus)")
    part_sem = os.path.join(OPT_DIR, "6-name2semantic-0.tsv")
    final_sem = os.path.join(OPT_DIR, "6-name2semantic.tsv")
    if os.path.exists(part_sem):
        shutil.move(part_sem, final_sem)

    # 2. Train Stage 2 SoVITS v2ProPlus (10 epochs on studio-conditioned clips)
    with open(os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS/configs/s2v2ProPlus.json"), "r", encoding="utf-8") as f:
        s2_cfg = json.load(f)

    os.makedirs(os.path.join(OPT_DIR, f"logs_s2_{VERSION}"), exist_ok=True)
    os.makedirs(os.path.join(GPT_SOVITS_ROOT, "SoVITS_weights_v2ProPlus"), exist_ok=True)
    os.makedirs(os.path.join(GPT_SOVITS_ROOT, "TEMP"), exist_ok=True)

    s2_cfg["train"]["batch_size"] = 4
    s2_cfg["train"]["epochs"] = 10
    s2_cfg["train"]["text_low_lr_rate"] = 0.4
    s2_cfg["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    s2_cfg["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
    s2_cfg["train"]["if_save_latest"] = True
    s2_cfg["train"]["if_save_every_weights"] = True
    s2_cfg["train"]["save_every_epoch"] = 5
    s2_cfg["train"]["gpu_numbers"] = "0"
    s2_cfg["train"]["grad_ckpt"] = False
    s2_cfg["train"]["lora_rank"] = 32
    s2_cfg["model"]["version"] = VERSION
    s2_cfg["data"]["exp_dir"] = s2_cfg["s2_ckpt_dir"] = OPT_DIR
    s2_cfg["save_weight_dir"] = "SoVITS_weights_v2ProPlus"
    s2_cfg["name"] = EXP_NAME
    s2_cfg["version"] = VERSION

    tmp_s2 = os.path.join(GPT_SOVITS_ROOT, "TEMP", "tmp_s2_v2proplus.json")
    with open(tmp_s2, "w", encoding="utf-8") as f:
        json.dump(s2_cfg, f, indent=2)

    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_s2], base_env, "Stage 2 SoVITS v2ProPlus Training")

    # 3. Train Stage 1 AR GPT using s1v3.ckpt base (15 epochs)
    with open(os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml"), "r", encoding="utf-8") as f:
        s1_cfg = yaml.safe_load(f)

    os.makedirs(os.path.join(OPT_DIR, f"logs_s1_{VERSION}"), exist_ok=True)
    os.makedirs(os.path.join(GPT_SOVITS_ROOT, "GPT_weights_v2ProPlus"), exist_ok=True)

    s1_cfg["train"]["batch_size"] = 4
    s1_cfg["train"]["epochs"] = 15
    s1_cfg["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    s1_cfg["train"]["save_every_n_epoch"] = 5
    s1_cfg["train"]["if_save_every_weights"] = True
    s1_cfg["train"]["if_save_latest"] = True
    s1_cfg["train"]["if_dpo"] = False
    s1_cfg["train"]["half_weights_save_dir"] = "GPT_weights_v2ProPlus"
    s1_cfg["train"]["exp_name"] = EXP_NAME
    s1_cfg["train_semantic_path"] = final_sem
    s1_cfg["train_phoneme_path"] = final_txt
    s1_cfg["output_dir"] = os.path.join(OPT_DIR, f"logs_s1_{VERSION}")

    tmp_s1 = os.path.join(GPT_SOVITS_ROOT, "TEMP", "tmp_s1_v2proplus.yaml")
    with open(tmp_s1, "w", encoding="utf-8") as f:
        yaml.safe_dump(s1_cfg, f, default_flow_style=False)

    s1_env = base_env.copy()
    s1_env["hz"] = "25hz"
    run_step([PYTHON_EXE, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_s1], s1_env, "Stage 1 AR GPT (s1v3) Training")

    # Find the newly trained v2ProPlus weights and copy to C:\Users\fpike\VoiceCreation\weights
    sovits_cands = glob.glob(os.path.join(GPT_SOVITS_ROOT, "SoVITS_weights_v2ProPlus", f"{EXP_NAME}*.pth"))
    gpt_cands = glob.glob(os.path.join(GPT_SOVITS_ROOT, "GPT_weights_v2ProPlus", f"{EXP_NAME}*.ckpt"))
    latest_sovits = max(sovits_cands, key=os.path.getmtime)
    latest_gpt = max(gpt_cands, key=os.path.getmtime)
    print(f"Trained v2ProPlus SoVITS weight: {latest_sovits}")
    print(f"Trained v2ProPlus GPT weight:    {latest_gpt}")

    # Copy into VoiceCreation/weights (both under their exact names and standard names)
    weights_dir = r"C:\Users\fpike\VoiceCreation\weights"
    shutil.copy2(latest_sovits, os.path.join(weights_dir, os.path.basename(latest_sovits)))
    shutil.copy2(latest_gpt, os.path.join(weights_dir, os.path.basename(latest_gpt)))
    # Also overwrite Francisv2-e15.ckpt and Francisv2_e8_s344.pth so existing scripts auto-load v2ProPlus!
    shutil.copy2(latest_sovits, os.path.join(weights_dir, "Francisv2_e8_s344.pth"))
    shutil.copy2(latest_gpt, os.path.join(weights_dir, "Francisv2-e15.ckpt"))

    # Synthesize & Evaluate Iteration 2 Sample
    sys.path.insert(0, GPT_SOVITS_ROOT)
    sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))
    sys.path.insert(0, r"C:\Users\fpike\VoiceCreation\scripts")
    os.chdir(GPT_SOVITS_ROOT)

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from evaluate_quality import evaluate_wav, TARGET_SENTENCE

    ref_dir = r"C:\Users\fpike\VoiceCreation\ref"
    with open(os.path.join(ref_dir, "ref_metadata.json"), "r", encoding="utf-8") as f:
        ref_meta = json.load(f)

    ref_audio = os.path.join(ref_dir, "ref.wav").replace("\\", "/")
    ref_text = open(os.path.join(ref_dir, "ref_transcript.txt"), "r", encoding="utf-8").read().strip()
    aux_refs = ref_meta["aux_ref_paths"]

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2ProPlus"
    tts_config.t2s_weights_path = latest_gpt.replace("\\", "/")
    tts_config.vits_weights_path = latest_sovits.replace("\\", "/")

    tts = TTS(tts_config)

    # Clean text for synthesis using cut4 (splits only on periods so "G'day," stays in sentence 1!)
    norm_target = (
        "G'day, this is Francis version two running locally on G P T So Vits. "
        "Feel free to use this voice in your own projects as long as you credit Francis version two."
    )
    inputs = {
        "text": norm_target,
        "text_lang": "en",
        "ref_audio_path": ref_audio,
        "aux_ref_audio_paths": aux_refs,
        "prompt_text": ref_text,
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

    sr, audio = next(tts.run(inputs))
    out_path = r"C:\Users\fpike\VoiceCreation\samples\iterations\iter2_sample.wav"
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"Saved Iteration 2 sample to {out_path} (sr={sr}, len={len(audio)/sr:.2f}s)")

    res = evaluate_wav(out_path, "Iteration 2 (GPT-SoVITS v2ProPlus + s1v3 + ERes2NetV2 SV + Studio Dataset)")
    print("=== ITERATION 2 EVALUATION ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
