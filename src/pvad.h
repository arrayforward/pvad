// pvad.h - PVAD (Personal VAD) ONNX 封装
// 输入 feats [B,T,80]（与 fbank.cpp 同参数 + per-bin 均值归一化）、emb [B,192]
// （CAM++ enrollment embedding，L2 归一化），输出 logits [B,T,3]（0 静音/1 非目标/2 目标）。
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

class Pvad {
public:
    explicit Pvad(const std::string& model_path);
    ~Pvad();
    Pvad(const Pvad&) = delete;
    Pvad& operator=(const Pvad&) = delete;

    // feats: T*80（已 per-bin 均值归一化），emb: 192 维。
    // 返回 T 个 P(target)（对 logits softmax 后的第 2 类概率），帧率 10ms，与 feats 帧对齐。
    std::vector<float> target_probs(const float* feats, int T, const float* emb);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_feats_, in_emb_, out_logits_;
};
