// aec.h - SpeexDSP MDF 回声消除封装
#pragma once
#include <vector>

struct SpeexEchoState_;
struct SpeexPreprocessState_;
typedef SpeexEchoState_ SpeexEchoState;
typedef SpeexPreprocessState_ SpeexPreprocessState;

class Aec {
public:
    // frame_size=160 (10ms@16k), filter_length 覆盖回声尾长（默认 2048 ≈ 128ms）
    explicit Aec(int frame_size = 160, int filter_length = 2048, int sample_rate = 16000);
    ~Aec();
    Aec(const Aec&) = delete;
    Aec& operator=(const Aec&) = delete;

    // near: 麦克风帧, far: 播放参考帧, out: 消回声后的帧（各 frame_size 采样）
    void process(const float* near_mic, const float* far_end, float* out);
    int frame_size() const { return n_; }

private:
    SpeexEchoState* st_ = nullptr;
    SpeexPreprocessState* den_ = nullptr;
    int n_;
    std::vector<short> a_, b_, c_;
};
