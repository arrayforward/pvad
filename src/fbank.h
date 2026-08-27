// fbank.h - Kaldi 风格 fbank 特征（80 维 mel，25ms 窗 / 10ms 移，预加重 0.97）
#pragma once
#include <vector>

struct FbankOptions {
    int sample_rate = 16000;
    int num_bins = 80;
    int frame_len = 400;    // 25ms @ 16k
    int frame_shift = 160;  // 10ms @ 16k
    int fft_size = 512;     // >= frame_len 的 2 的幂
    float preemph = 0.97f;
    float low_freq = 20.f;
    float high_freq = 0.f;  // 0 -> nyquist
};

class Fbank {
public:
    explicit Fbank(const FbankOptions& opt = FbankOptions());
    // 计算 log-mel 特征，输出 out = num_frames * num_bins（行优先），返回帧数
    int compute(const float* pcm, int num_samples, std::vector<float>& out) const;
    // 单帧版本：win400 为 400 采样（25ms）窗口，输出 80 维 log-mel（与 compute 单帧一致）
    void compute_one(const float* win400, float* out80) const;
    int num_bins() const { return opt_.num_bins; }

private:
    FbankOptions opt_;
    std::vector<float> window_;                    // hamming
    std::vector<std::vector<std::pair<int, float>>> mel_filters_;  // (fft_bin, weight)
};

// 对特征做 per-bin 均值归一化（3D-Speaker 做法：当前分析窗内减均值）
void mean_normalize(std::vector<float>& feats, int num_frames, int num_bins);
