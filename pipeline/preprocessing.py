"""
Preprocessing utilities for SSL pretraining on CWRU + Paderborn.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import stft


def load_cwru_mat_dir(root_dir, key_substr="DE_time"):
    records = []
    for fpath in sorted(glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)):
        mat = loadmat(fpath)
        sig_key = next((k for k in mat.keys() if key_substr in k), None)
        if sig_key is None:
            continue
        signal = np.asarray(mat[sig_key]).squeeze().astype(np.float64)
        records.append({
            "signal": signal,
            "file": os.path.basename(fpath),
            "label": os.path.splitext(os.path.basename(fpath))[0],
        })
    if not records:
        raise FileNotFoundError(f"No CWRU .mat files found under {root_dir}.")
    return records


def load_paderborn_mat_dir(root_dir, channel_name="phase_current_1"):
    records = []
    for fpath in sorted(glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True)):
        mat = loadmat(fpath, simplify_cells=True)
        top_keys = [k for k in mat.keys() if not k.startswith("__")]
        if not top_keys:
            continue
        record_struct = mat[top_keys[0]]
        channels = record_struct.get("Y", [])
        signal = None
        context = {}
        for ch in channels:
            name = ch.get("Name", "")
            data = ch.get("Data", None)
            if name == channel_name and data is not None:
                signal = np.asarray(data).squeeze().astype(np.float64)
            elif name in ("speed", "torque", "force") and data is not None:
                context[name] = np.asarray(data).squeeze().astype(np.float64)
        if signal is None:
            continue
        records.append({
            "signal": signal,
            "file": os.path.basename(fpath),
            "label": os.path.splitext(os.path.basename(fpath))[0],
            "context": context,
        })
    if not records:
        raise FileNotFoundError(f"No usable Paderborn signals found under {root_dir}.")
    return records


def inspect_mat_structure(fpath, max_depth=3):
    mat = loadmat(fpath, simplify_cells=True)

    def _walk(entry, prefix, depth):
        if depth > max_depth:
            return
        if isinstance(entry, dict):
            for k, v in entry.items():
                shape = getattr(np.asarray(v), "shape", None)
                print(f"{'  ' * depth}{prefix}{k}  (shape={shape}, type={type(v).__name__})")
                _walk(v, "", depth + 1)

    _walk(mat, "", 0)


def rms_normalise(signal):
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    if rms < 1e-12:
        return signal
    return signal / rms


def random_crop_segment(signal, segment_len, rng=None):
    rng = rng or np.random
    if len(signal) <= segment_len:
        pad = segment_len - len(signal)
        return np.pad(signal, (0, pad), mode="edge")
    start = rng.integers(0, len(signal) - segment_len)
    return signal[start:start + segment_len]


def to_stft(segment, fs, nperseg=256, noverlap=128):
    _, _, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx).astype(np.float32)


def dual_phase_order_augment(multichannel_segment):
    if multichannel_segment.ndim == 1 or multichannel_segment.shape[0] == 1:
        return [multichannel_segment]
    return [multichannel_segment, multichannel_segment[::-1, :]]


class SSLPretrainDataset:
    def __init__(self, records, fs, segment_len=2048, nperseg=256, noverlap=128, seed=0):
        self.records = records
        self.fs = fs
        self.segment_len = segment_len
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def get_segment(self, idx, domain="time"):
        signal = self.records[idx]["signal"]
        signal = rms_normalise(signal)
        seg = random_crop_segment(signal, self.segment_len, rng=self.rng)
        if domain == "time":
            return seg.astype(np.float32)
        elif domain == "stft":
            return to_stft(seg, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        raise ValueError(domain)