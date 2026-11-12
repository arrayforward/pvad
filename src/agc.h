// agc.h - 16kHz 单声道电平测量与 TSE 输入自动增益控制
#pragma once
#include <cstddef>
#include <vector>

struct AudioLevel {
    double seconds = 0.0;
    float rms = 0.f;          // 全段 RMS（含静音）
    float active_rms = 0.f;   // 10ms 帧 RMS>0.003 的有效语音 RMS
    float peak = 0.f;
    size_t clipped = 0;       // |sample|>=0.999 的采样数
    float rms_dbfs() const;
    float active_dbfs() const;
    float peak_dbfs() const;
};

// 流式电平计：可逐帧 add，measure 不清状态；新测量窗口前调用 reset。
class LevelMeter {
public:
    void reset();
    void add(const float* pcm, size_t n);
    AudioLevel measure() const;
private:
    double sum_sq_ = 0.0;
    double active_sum_sq_ = 0.0;
    double frame_sum_sq_ = 0.0;
    size_t samples_ = 0;
    size_t active_samples_ = 0;
    size_t frame_samples_ = 0;
    size_t clipped_ = 0;
    float peak_ = 0.f;
};

struct AgcReport {
    AudioLevel before;
    AudioLevel after;
    float gain = 1.f;
};

// 整段有效语音 RMS -> target_rms（默认 0.045，约 -26.9dBFS）。
// 使用整段恒定增益而非逐帧压缩，保留段内动态；增益限 [0.1,12]，峰值限 0.95。
// 用于已缓冲的 TSE 注册/提取切片，不改变 PVAD 门控输入。
std::vector<float> agc_normalize(const float* pcm, size_t n,
                                 float target_rms = 0.045f,
                                 AgcReport* report = nullptr);
