// agc.cpp
#include "agc.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr float kActiveThreshold = 0.003f;  // -50.5dBFS：低于此电平不抬增益
constexpr size_t kFrameSamples = 160;       // 10ms @16kHz
constexpr float kMaxGain = 12.0f;
constexpr float kMinGain = 0.1f;
constexpr float kPeakTarget = 0.95f;

float dbfs(float value) {
    return 20.f * std::log10(std::max(value, 1e-10f));
}
}  // namespace

float AudioLevel::rms_dbfs() const { return dbfs(rms); }
float AudioLevel::active_dbfs() const { return dbfs(active_rms); }
float AudioLevel::peak_dbfs() const { return dbfs(peak); }

void LevelMeter::reset() {
    sum_sq_ = 0.0;
    active_sum_sq_ = 0.0;
    frame_sum_sq_ = 0.0;
    samples_ = 0;
    active_samples_ = 0;
    frame_samples_ = 0;
    clipped_ = 0;
    peak_ = 0.f;
}

void LevelMeter::add(const float* pcm, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = pcm[i];
        double sq = v * v;
        sum_sq_ += sq;
        frame_sum_sq_ += sq;
        samples_++;
        frame_samples_++;
        float a = std::fabs(pcm[i]);
        peak_ = std::max(peak_, a);
        if (a >= 0.999f) clipped_++;
        if (frame_samples_ == kFrameSamples) {
            float frame_rms = std::sqrt((float)(frame_sum_sq_ / kFrameSamples));
            if (frame_rms > kActiveThreshold) {
                active_sum_sq_ += frame_sum_sq_;
                active_samples_ += kFrameSamples;
            }
            frame_sum_sq_ = 0.0;
            frame_samples_ = 0;
        }
    }
}

AudioLevel LevelMeter::measure() const {
    AudioLevel out;
    out.seconds = samples_ / 16000.0;
    out.rms = samples_ ? std::sqrt((float)(sum_sq_ / samples_)) : 0.f;
    // 不满 10ms 的尾帧不计 active，与 agc_normalize 的帧口径一致。
    out.active_rms = active_samples_
        ? std::sqrt((float)(active_sum_sq_ / active_samples_)) : 0.f;
    out.peak = peak_;
    out.clipped = clipped_;
    return out;
}

std::vector<float> agc_normalize(const float* pcm, size_t n,
                                 float target_rms, AgcReport* report) {
    LevelMeter meter;
    meter.add(pcm, n);
    AudioLevel before = meter.measure();
    float level = before.active_rms > 0.f ? before.active_rms : before.rms;
    float gain = target_rms / std::max(level, 1e-6f);
    gain = std::clamp(gain, kMinGain, kMaxGain);
    if (before.peak > 0.f) gain = std::min(gain, kPeakTarget / before.peak);

    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = std::clamp(pcm[i] * gain, -0.98f, 0.98f);
    if (report) {
        report->before = before;
        LevelMeter after_meter;
        after_meter.add(out.data(), out.size());
        report->after = after_meter.measure();
        report->gain = gain;
    }
    return out;
}
