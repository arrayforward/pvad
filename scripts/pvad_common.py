# -*- coding: utf-8 -*-
"""PVAD 公共模块: fbank (与 src/fbank.cpp 完全一致), CAM++ embedding, 数据集, 模型。

fbank 参数对齐 C++ 版: 16kHz, 80 mel, 25ms(400)/10ms(160), fft 512,
预加重 0.97, 对称 hamming, low 20Hz, log(max(e,1e-10)), 无中心 padding,
帧数 = 1 + (n-400)//160。embedding 前做 per-bin 均值归一化 (3D-Speaker 做法)。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
FRAME_LEN = 400
FRAME_SHIFT = 160
FFT_SIZE = 512
NUM_BINS = 80
PREEMPH = 0.97
LOW_FREQ = 20.0
FRAME = 160  # 10ms 标签帧长 (= FRAME_SHIFT)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_matrix():
    """与 fbank.cpp 相同的三角滤波器 (bin 四舍五入, 无面积归一化), [80, 257]。"""
    high = SR / 2.0
    mel_lo, mel_hi = _hz_to_mel(LOW_FREQ), _hz_to_mel(high)
    npts = NUM_BINS + 2
    bins = []
    for i in range(npts):
        m = mel_lo + (mel_hi - mel_lo) * i / (npts - 1)
        f = _mel_to_hz(m)
        b = int(np.floor(FFT_SIZE * f / SR + 0.5))
        bins.append(max(0, min(b, FFT_SIZE // 2)))
    mat = np.zeros((NUM_BINS, FFT_SIZE // 2 + 1), dtype=np.float64)
    for b in range(NUM_BINS):
        l, c, r = bins[b], bins[b + 1], bins[b + 2]
        if c <= l:
            c = l + 1
        if r <= c:
            r = c + 1
        if r > FFT_SIZE // 2:
            r = FFT_SIZE // 2
        for k in range(l, min(r, FFT_SIZE // 2) + 1):
            if k < c:
                w = (k - l) / (c - l)
            elif k > c:
                w = (r - k) / (r - c)
            else:
                w = 1.0
            if w > 0:
                mat[b, k] = w
    return mat


_MEL_MAT = _build_mel_matrix()
_WINDOW = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))


def fbank(pcm: np.ndarray) -> np.ndarray:
    """pcm float32 [-1,1] -> [T, 80] log-mel, 与 C++ 版一致。"""
    n = len(pcm)
    if n < FRAME_LEN:
        return np.zeros((0, NUM_BINS), dtype=np.float32)
    n_frames = 1 + (n - FRAME_LEN) // FRAME_SHIFT
    # 分帧
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = pcm[idx].astype(np.float64)
    # 预加重 (C++: re[0]=s[0]*w[0], re[i]=(s[i]-0.97*s[i-1])*w[i])
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0]
    pre[:, 1:] = frames[:, 1:] - PREEMPH * frames[:, :-1]
    pre *= _WINDOW
    spec = np.fft.rfft(pre, n=FFT_SIZE, axis=1)
    power = spec.real ** 2 + spec.imag ** 2  # [T, 257]
    e = power @ _MEL_MAT.T  # [T, 80]
    return np.log(np.maximum(e, 1e-10)).astype(np.float32)


def mean_normalize(feats: np.ndarray) -> np.ndarray:
    return feats - feats.mean(axis=0, keepdims=True)


class CampplusEmbedder:
    """onnxruntime CAM++ 192 维 embedding, 与 src/speaker.cpp 一致。"""

    def __init__(self, model_path=None, intra_threads=1):
        import onnxruntime as ort
        model_path = model_path or str(ROOT / "models" / "campplus.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        feats = mean_normalize(fbank(pcm))
        if len(feats) < 4:
            raise ValueError("audio too short for embedding")
        out = self.sess.run(None, {self.in_name: feats[None, :, :]})[0]
        emb = np.asarray(out, dtype=np.float64).ravel()
        n = np.linalg.norm(emb)
        if n > 1e-8:
            emb /= n
        return emb.astype(np.float32)


def read_wav(path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data[:, 0], sr


def load_labels(labels_path):
    recs = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs
