import os
import subprocess
import numpy as np
from scipy.io import wavfile

FFMPEG_PATH = r"C:\Program Files\Shutter Encoder\Library\ffmpeg.exe"
INPUT_WAV = r"C:\voice_data\francis_v2_raw.wav"
OPT_ROOT = r"C:\GPT-SoVITS\output\slicer_opt"


def load_audio_32k(file_path: str) -> np.ndarray:
    cmd = [
        FFMPEG_PATH,
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", "32000",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).flatten()


def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class Slicer:
    def __init__(self, sr: int, threshold: float = -34.0, min_length: int = 4000, min_interval: int = 300, hop_size: int = 10, max_sil_kept: int = 500):
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        samples = waveform
        if samples.shape[0] <= self.min_length:
            return [[waveform, 0, samples.shape[0]]]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [[waveform, 0, int(total_frames * self.hop_size)]]
        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]), 0, int(sil_tags[0][0] * self.hop_size)])
        for i in range(len(sil_tags) - 1):
            chunks.append([
                self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
                int(sil_tags[i][1] * self.hop_size),
                int(sil_tags[i + 1][0] * self.hop_size),
            ])
        if sil_tags[-1][1] < total_frames:
            chunks.append([
                self._apply_slice(waveform, sil_tags[-1][1], total_frames),
                int(sil_tags[-1][1] * self.hop_size),
                int(total_frames * self.hop_size),
            ])
        return chunks


def main():
    os.makedirs(OPT_ROOT, exist_ok=True)
    print(f"Loading and slicing {INPUT_WAV} into {OPT_ROOT} ...")
    audio = load_audio_32k(INPUT_WAV)
    slicer = Slicer(sr=32000, threshold=-34, min_length=4000, min_interval=300, hop_size=10, max_sil_kept=500)
    _max = 0.9
    alpha = 0.25
    name = os.path.basename(INPUT_WAV)
    count = 0
    good_4_to_10s = 0
    for chunk, start, end in slicer.slice(audio):
        tmp_max = np.abs(chunk).max()
        if tmp_max == 0:
            continue
        if tmp_max > 1:
            chunk /= tmp_max
        chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
        out_path = "%s/%s_%010d_%010d.wav" % (OPT_ROOT, name, start, end)
        wavfile.write(out_path, 32000, (chunk * 32767).astype(np.int16))
        count += 1
        dur = (end - start) / 32000.0
        if 3.0 <= dur <= 10.5:
            good_4_to_10s += 1
    print(f"Slicing complete! Generated {count} total clips ({good_4_to_10s} clips in the ideal 3.0-10.5s range).")


if __name__ == "__main__":
    main()
