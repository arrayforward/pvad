// dsp_tse.h - TSE 用 DSP：radix-2 FFT + center=True STFT/ISTFT + ERB 投影
// 逐参数对齐 tse-gitee/android_demo Dsp.kt（16kHz, n_fft=512, win=400, hop=160,
// periodic hamming, center=True 零填充）与 tse_array/erb.py 的 ERB 滤波器组。
#pragma once
#include <vector>

namespace tse_dsp {

constexpr int SR = 16000;
constexpr int N_FFT = 512;
constexpr int WIN = 400;
constexpr int HOP = 160;
constexpr int F = N_FFT / 2 + 1;      // 257
constexpr int ERB_BANDS = 32;
constexpr int PAD = N_FFT / 2;        // 256 (center=True)
constexpr int WIN_OFF = (N_FFT - WIN) / 2;  // 56

void fft(std::vector<float>& re, std::vector<float>& im, bool inverse);

// 输出 [T][F] 实/虚部
void stft(const float* pcm, int n, std::vector<std::vector<float>>& re,
          std::vector<std::vector<float>>& im);
// OLA + 窗平方包络归一化，去两端 PAD
std::vector<float> istft(const std::vector<std::vector<float>>& re,
                         const std::vector<std::vector<float>>& im, int length);
const std::vector<std::vector<float>>& erb_matrix();  // [32][257]
// STFT 功率 [T][257] -> ERB [T][32]
std::vector<std::vector<float>> erb_project(const std::vector<std::vector<float>>& power);

}  // namespace tse_dsp
