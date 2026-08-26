// aec.cpp
#include "aec.h"
#include <speex/speex_echo.h>
#include <speex/speex_preprocess.h>
#include <algorithm>

Aec::Aec(int frame_size, int filter_length, int sample_rate) : n_(frame_size) {
    st_ = speex_echo_state_init(frame_size, filter_length);
    den_ = speex_preprocess_state_init(frame_size, sample_rate);
    speex_echo_ctl(st_, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate);
    speex_preprocess_ctl(den_, SPEEX_PREPROCESS_SET_ECHO_STATE, st_);
    int noise_suppress = -30;
    speex_preprocess_ctl(den_, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &noise_suppress);
    a_.resize(n_);
    b_.resize(n_);
    c_.resize(n_);
}

Aec::~Aec() {
    if (den_) speex_preprocess_state_destroy(den_);
    if (st_) speex_echo_state_destroy(st_);
}

void Aec::process(const float* near_mic, const float* far_end, float* out) {
    for (int i = 0; i < n_; i++) {
        a_[i] = (short)std::max(-32768.f, std::min(32767.f, near_mic[i] * 32768.f));
        b_[i] = (short)std::max(-32768.f, std::min(32767.f, far_end[i] * 32768.f));
    }
    speex_echo_cancellation(st_, a_.data(), b_.data(), c_.data());
    speex_preprocess_run(den_, c_.data());
    for (int i = 0; i < n_; i++) out[i] = c_[i] / 32768.f;
}
