import os
import subprocess
import sys

FFMPEG_PATH = r"C:\Program Files\Shutter Encoder\Library\ffmpeg.exe"
INPUT_M4A = r"C:\Users\fpike\.gemini\antigravity\scratch\francis_voice_agent\voice_data\Glenroi_Av_2.m4a"
OUTPUT_DIR = r"C:\voice_data"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "francis_v2_raw.wav")


def main():
    if not os.path.exists(FFMPEG_PATH):
        raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_PATH}")
    if not os.path.exists(INPUT_M4A):
        raise FileNotFoundError(f"Input audio not found at {INPUT_M4A}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/2] Converting and normalizing {INPUT_M4A} -> {OUTPUT_WAV} ...")
    # Apply 80Hz high-pass filter to remove rumble/handling noise and EBU R128 loudnorm (-16 LUFS)
    # so GPT-SoVITS's -34 dB slicer cuts cleanly between speech segments.
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", INPUT_M4A,
        "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        OUTPUT_WAV,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("FFmpeg error:\n", res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    size_mb = os.path.getsize(OUTPUT_WAV) / (1024 * 1024)
    print(f"[2/2] Created {OUTPUT_WAV} ({size_mb:.1f} MB). Verifying volume levels...")

    verify_cmd = [
        FFMPEG_PATH,
        "-t", "300",
        "-i", OUTPUT_WAV,
        "-af", "volumedetect",
        "-f", "null",
        "NUL",
    ]
    v_res = subprocess.run(verify_cmd, capture_output=True, text=True)
    for line in v_res.stderr.splitlines():
        if "volumedetect" in line and ("mean_volume" in line or "max_volume" in line):
            print("  ", line.strip())


if __name__ == "__main__":
    main()
