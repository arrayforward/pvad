// fbank.cpp - 朴素 radix-2 FFT + mel 滤波 + log
#include "fbank.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {
const float kPi = 3.14159265358979323846f;

void fft_inplace(std::vector<float>& re, std::vector<float>& im) {
    int n = (int)re.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.f * kPi / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.f, ci = 0.f;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i + k], ui = im[i + k];
                float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;             im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;   im[i + k + len / 2] = ui - vi;
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

float hz_to_mel(float f) { return 2595.f * log10f(1.f + f / 700.f); }
float mel_to_hz(float m) { return 700.f * (powf(10.f, m / 2595.f) - 1.f); }
}  // namespace

Fbank::Fbank(const FbankOptions& opt) : opt_(opt) {
    // hamming 窗
    window_.resize(opt_.frame_len);
    for (int i = 0; i < opt_.frame_len; i++)
        window_[i] = 0.54f - 0.46f * cosf(2.f * kPi * i / (opt_.frame_len - 1));

    // mel 三角滤波器组
    float low = opt_.low_freq;
    float high = opt_.high_freq > 0 ? opt_.high_freq : opt_.sample_rate / 2.f;
    float mel_lo = hz_to_mel(low), mel_hi = hz_to_mel(high);
    int npts = opt_.num_bins + 2;
    std::vector<int> bins(npts);
    for (int i = 0; i < npts; i++) {
        float m = mel_lo + (mel_hi - mel_lo) * i / (npts - 1);
        float f = mel_to_hz(m);
        int b = (int)floorf(opt_.fft_size * f / opt_.sample_rate + 0.5f);
        bins[i] = std::max(0, std::min(b, opt_.fft_size / 2));
    }
    mel_filters_.resize(opt_.num_bins);
    for (int b = 0; b < opt_.num_bins; b++) {
        int l = bins[b], c = bins[b + 1], r = bins[b + 2];
        if (c <= l) c = l + 1;
        if (r <= c) r = c + 1;
        if (r > opt_.fft_size / 2) r = opt_.fft_size / 2;
        for (int k = l; k <= r && k <= opt_.fft_size / 2; k++) {
            float w = 0.f;
            if (k < c) w = (float)(k - l) / (c - l);
            else if (k > c) w = (float)(r - k) / (r - c);
            else w = 1.f;
            if (w > 0.f) mel_filters_[b].push_back({k, w});
        }
    }
}

int Fbank::compute(const float* pcm, int num_samples, std::vector<float>& out) const {
    if (num_samples < opt_.frame_len) return 0;
    int num_frames = 1 + (num_samples - opt_.frame_len) / opt_.frame_shift;
    out.assign((size_t)num_frames * opt_.num_bins, 0.f);

    std::vector<float> re(opt_.fft_size), im(opt_.fft_size);
    for (int fidx = 0; fidx < num_frames; fidx++) {
        const float* s = pcm + (size_t)fidx * opt_.frame_shift;
        std::fill(re.begin(), re.end(), 0.f);
        std::fill(im.begin(), im.end(), 0.f);
        // 预加重 + 加窗
        re[0] = s[0] * window_[0];
        for (int i = 1; i < opt_.frame_len; i++)
            re[i] = (s[i] - opt_.preemph * s[i - 1]) * window_[i];
        fft_inplace(re, im);
        // 功率谱 -> mel -> log
        for (int b = 0; b < opt_.num_bins; b++) {
            float e = 0.f;
            for (auto& pr : mel_filters_[b]) {
                float p = re[pr.first] * re[pr.first] + im[pr.first] * im[pr.first];
                e += pr.second * p;
            }
            out[(size_t)fidx * opt_.num_bins + b] = logf(std::max(e, 1e-10f));
        }
    }
    return num_frames;
}

void mean_normalize(std::vector<float>& feats, int num_frames, int num_bins) {
    for (int b = 0; b < num_bins; b++) {
        double sum = 0;
        for (int t = 0; t < num_frames; t++) sum += feats[(size_t)t * num_bins + b];
        float mean = (float)(sum / num_frames);
        for (int t = 0; t < num_frames; t++) feats[(size_t)t * num_bins + b] -= mean;
    }
}
