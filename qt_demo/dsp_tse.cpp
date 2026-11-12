// dsp_tse.cpp - 逐参数对齐 tse-gitee/android_demo Dsp.kt
#include "dsp_tse.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tse_dsp {

namespace {
constexpr float kPi = 3.14159265358979323846f;

std::vector<float> make_window() {
    std::vector<float> w(WIN);
    for (int i = 0; i < WIN; i++)
        w[i] = (float)(0.54 - 0.46 * std::cos(2.0 * kPi * i / WIN));  // periodic
    return w;
}
}  // namespace

void fft(std::vector<float>& re, std::vector<float>& im, bool inverse) {
    int n = (int)re.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    float sign = inverse ? 1.f : -1.f;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.f * kPi / len;
        float wr = std::cos(ang), wi = std::sin(ang);
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
    if (inverse) {
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

void stft(const float* pcm, int n, std::vector<std::vector<float>>& re,
          std::vector<std::vector<float>>& im) {
    static const std::vector<float> window = make_window();
    int n_pad = n + 2 * PAD;
    std::vector<float> padded(n_pad, 0.f);
    std::memcpy(padded.data() + PAD, pcm, n * sizeof(float));
    int T = 1 + (n_pad - N_FFT) / HOP;
    re.assign(T, std::vector<float>(F));
    im.assign(T, std::vector<float>(F));
    std::vector<float> fr(N_FFT), fi(N_FFT);
    for (int f = 0; f < T; f++) {
        std::fill(fr.begin(), fr.end(), 0.f);
        std::fill(fi.begin(), fi.end(), 0.f);
        int base = f * HOP;
        for (int i = 0; i < WIN; i++)
            fr[WIN_OFF + i] = padded[base + WIN_OFF + i] * window[i];
        fft(fr, fi, false);
        std::copy(fr.begin(), fr.begin() + F, re[f].begin());
        std::copy(fi.begin(), fi.begin() + F, im[f].begin());
    }
}

std::vector<float> istft(const std::vector<std::vector<float>>& re,
                         const std::vector<std::vector<float>>& im, int length) {
    static const std::vector<float> window = make_window();
    int T = (int)re.size();
    int n = (T - 1) * HOP + N_FFT;
    std::vector<float> out(n, 0.f), env(n, 0.f);
    std::vector<float> fr(N_FFT), fi(N_FFT);
    for (int f = 0; f < T; f++) {
        std::copy(re[f].begin(), re[f].begin() + F, fr.begin());
        std::copy(im[f].begin(), im[f].begin() + F, fi.begin());
        // 共轭对称补全（torch.rfft 频谱的负频率）
        for (int k = F; k < N_FFT; k++) { fr[k] = fr[N_FFT - k]; fi[k] = -fi[N_FFT - k]; }
        fft(fr, fi, true);
        int base = f * HOP;
        for (int i = 0; i < WIN; i++) {
            out[base + WIN_OFF + i] += fr[WIN_OFF + i] * window[i];
            env[base + WIN_OFF + i] += window[i] * window[i];
        }
    }
    for (int i = 0; i < n; i++)
        if (env[i] > 1e-8f) out[i] /= env[i];
    int start = PAD;
    int end = std::min(start + length, n);
    return std::vector<float>(out.begin() + start, out.begin() + end);
}

const std::vector<std::vector<float>>& erb_matrix() {
    static const std::vector<std::vector<float>> m = [] {
        auto hz2erb = [](float hz) { return (float)(21.4 * std::log10(1.0 + 0.00437 * hz)); };
        auto erb2hz = [](float e) { return (float)((std::pow(10.0, e / 21.4) - 1.0) / 0.00437); };
        std::vector<std::vector<float>> mat(ERB_BANDS, std::vector<float>(F));
        float lo = hz2erb(16000.f / N_FFT), hi = hz2erb(8000.f);
        for (int b = 0; b < ERB_BANDS; b++) {
            float l = erb2hz(lo + (hi - lo) * b / (ERB_BANDS + 1));
            float c = erb2hz(lo + (hi - lo) * (b + 1) / (ERB_BANDS + 1));
            float r = erb2hz(lo + (hi - lo) * (b + 2) / (ERB_BANDS + 1));
            float sum = 0.f;
            for (int k = 0; k < F; k++) {
                float hz = k * 16000.f / N_FFT;
                float rise = (hz - l) / std::max(c - l, 1e-8f);
                float fall = (r - hz) / std::max(r - c, 1e-8f);
                float v = std::clamp(std::min(rise, fall), 0.f, 1.f);
                mat[b][k] = v;
                sum += v;
            }
            if (sum > 1e-8f)
                for (int k = 0; k < F; k++) mat[b][k] /= sum;
        }
        return mat;
    }();
    return m;
}

std::vector<std::vector<float>> erb_project(const std::vector<std::vector<float>>& power) {
    const auto& mat = erb_matrix();
    std::vector<std::vector<float>> out(power.size(), std::vector<float>(ERB_BANDS));
    for (size_t t = 0; t < power.size(); t++)
        for (int b = 0; b < ERB_BANDS; b++) {
            float s = 0.f;
            const float* row = mat[b].data();
            const float* p = power[t].data();
            for (int k = 0; k < F; k++) s += p[k] * row[k];
            out[t][b] = s;
        }
    return out;
}

}  // namespace tse_dsp
