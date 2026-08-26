// vad.h - Silero VAD v5 (ONNX) 封装
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

class Vad {
public:
    explicit Vad(const std::string& model_path);
    ~Vad();
    Vad(const Vad&) = delete;
    Vad& operator=(const Vad&) = delete;

    // 送入 160 采样（10ms）帧，内部缓存到 512 采样跑一次模型；
    // 返回最近一次推理的语音概率，尚未推理时返回 -1。
    float process(const float* samples, int n);
    void reset();

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> state_;      // [2, 1, 128]
    std::vector<float> ctx_;        // 64 采样 context（上一帧末尾）
    std::vector<float> buf_;        // 待凑够 512 的采样
    float last_prob_ = -1.f;
    std::string in_name_input_, in_name_state_, in_name_sr_;
    std::string out_name_prob_, out_name_state_;
    void run_512();
};
