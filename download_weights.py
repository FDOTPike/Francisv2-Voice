from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

REPO_SLUG = "FDOTPike/Francisv2-Voice"
RELEASE_TAG = "v1.0.0"

WEIGHT_FILES = [
    {
        "filename": "Francisv2-e15.ckpt",
        "url": f"https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}/Francisv2-e15.ckpt",
        "sha256": "718A3A16F720736CCDA1FA96D1DF1F5F0319CFA5B323E14EE2E067EA5F43D6E9",
        "gpt_sovits_subdir": "GPT_weights_v2ProPlus",
    },
    {
        "filename": "Francisv2_e8_s344.pth",
        "url": f"https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}/Francisv2_e8_s344.pth",
        "sha256": "197658D82E7805F66FA7577F81870E63D386565C22ACBCCEA6F0345391182162",
        "gpt_sovits_subdir": "SoVITS_weights_v2ProPlus",
    },
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def download_with_progress(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {os.path.basename(dest_path)} from {url} ...")

    def _report(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = min(block_num * block_size, total_size)
            pct = (downloaded / total_size) * 100
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb_done:6.1f} / {mb_total:6.1f} MB ({pct:5.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_report)
    print()


def ensure_weights(repo_root: str | None = None, gpt_sovits_dir: str | None = None) -> dict[str, str]:
    if repo_root is None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(repo_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    resolved_paths = {}
    for item in WEIGHT_FILES:
        local_path = os.path.join(weights_dir, item["filename"])
        need_download = True
        if os.path.exists(local_path):
            actual_hash = sha256_file(local_path)
            if actual_hash == item["sha256"]:
                print(f"[OK] {item['filename']} already present and verified (SHA256 match).")
                need_download = False
            else:
                print(f"[!] Checksum mismatch for {item['filename']}, re-downloading...")

        if need_download:
            download_with_progress(item["url"], local_path)
            actual_hash = sha256_file(local_path)
            if actual_hash != item["sha256"]:
                raise RuntimeError(
                    f"SHA256 verification failed for {item['filename']}: "
                    f"expected {item['sha256']}, got {actual_hash}"
                )
            print(f"[OK] Verified {item['filename']} ({actual_hash[:12]}...)")

        resolved_paths[item["filename"]] = local_path

        # If a GPT-SoVITS installation exists, sync weights into its expected folders
        if gpt_sovits_dir and os.path.isdir(gpt_sovits_dir):
            for subdir in [item["gpt_sovits_subdir"], item["gpt_sovits_subdir"].replace("v2ProPlus", "v2")]:
                target_dir = os.path.join(gpt_sovits_dir, subdir)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, item["filename"])
                shutil.copy2(local_path, target_path)
            print(f"[Sync] Installed {item['filename']} into {gpt_sovits_dir}")

    # Sync reference audio & transcript to GPT-SoVITS root if present
    if gpt_sovits_dir and os.path.isdir(gpt_sovits_dir):
        for ref_name in ["ref.wav", "ref_transcript.txt", "ref_aux_1.wav", "ref_aux_2.wav", "ref_aux_3.wav"]:
            src = os.path.join(repo_root, "ref", ref_name)
            dst = os.path.join(gpt_sovits_dir, ref_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    return resolved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify Francisv2 (v2ProPlus Studio Master) GPT-SoVITS voice model weights."
    )
    parser.add_argument(
        "--gpt-sovits-dir",
        type=str,
        default=os.environ.get("GPT_SOVITS_ROOT", r"C:\GPT-SoVITS"),
        help="Optional path to local GPT-SoVITS folder to auto-install weights into.",
    )
    args = parser.parse_args()
    ensure_weights(gpt_sovits_dir=args.gpt_sovits_dir)
    print("\nFrancisv2 Studio Master weights are ready! Please remember to credit Francisv2 (by Francis Pike) in your project.")


if __name__ == "__main__":
    main()
