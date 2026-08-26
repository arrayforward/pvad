// denoise.cpp
#include "denoise.h"
#include <rnnoise.h>
#include <speex/speex_resampler.h>
#include <stdexcept>

Denoise::Denoise() {
    int err = 0;
    up_ = speex_resampler_init(1, 16000, 48000, 5, &err);
    if (!up_) throw std::runtime_error("speex_resampler_init up failed");
    down_ = speex_resampler_init(1, 48000, 16000, 5, &err);
    if (!down_) throw std::runtime_error("speex_resampler_init down failed");
    st_ = rnnoise_create(nullptr);  // 内置模型
    if (!st_) throw std::runtime_error("rnnoise_create failed");
}

Denoise::~Denoise() {
    if (st_) rnnoise_destroy(st_);
    if (up_) speex_resampler_destroy(up_);
    if (down_) speex_resampler_destroy(down_);
}

void Denoise::process(const float* in, float* out) {
    // 16k -> 48k（160 进 -> 稳态 480 出）
    float up[480];
    spx_uint32_t in_len = 160, out_len = 480;
    speex_resampler_process_float(up_, 0, in, &in_len, up, &out_len);
    q48_.insert(q48_.end(), up, up + out_len);

    // 凑够 480 跑一帧 RNNoise，再下采样回 16k
    while (q48_.size() >= 480) {
        float fin[480], fout[480], dn[160];
        for (int i = 0; i < 480; i++) { fin[i] = q48_.front(); q48_.pop_front(); }
        rnnoise_process_frame(st_, fout, fin);
        spx_uint32_t il = 480, ol = 160;
        speex_resampler_process_float(down_, 0, fout, &il, dn, &ol);
        qout_.insert(qout_.end(), dn, dn + ol);
    }

    for (int i = 0; i < 160; i++) {
        if (qout_.empty()) {
            out[i] = 0.f;
        } else {
            out[i] = qout_.front();
            qout_.pop_front();
        }
    }
}
