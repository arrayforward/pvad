// denoise.h - RNNoise 实时降噪封装
// RNNoise 原生 48kHz/480 采样帧；内部用 speex_resampler 做 16k↔48k 双向重采样。
// 用法：每 10ms 送入 160 采样（16k float [-1,1]），取回降噪后的 160 采样。
#pragma once
#include <deque>

struct DenoiseState;  // rnnoise
struct SpeexResamplerState_;
typedef SpeexResamplerState_ SpeexResamplerState;

class Denoise {
public:
    Denoise();
    ~Denoise();
    Denoise(const Denoise&) = delete;
    Denoise& operator=(const Denoise&) = delete;

    // in/out 各 160 采样（16k, 10ms）。启动初期重采样缓冲未填满时输出补零。
    void process(const float* in, float* out);
    // 新增的稳态流水线延迟（毫秒）
    static double extra_latency_ms() { return 11.4; }  // RNNoise 10ms 帧 + 双向重采样 ~1.4ms

private:
    SpeexResamplerState* up_ = nullptr;    // 16k -> 48k
    SpeexResamplerState* down_ = nullptr;  // 48k -> 16k
    DenoiseState* st_ = nullptr;
    std::deque<float> q48_;   // 上采样后待 RNNoise 处理
    std::deque<float> qout_;  // 降噪+下采样后待输出
};
