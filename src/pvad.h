// pvad.h - PVAD (Personal VAD) ONNX 封装（双接口）
// v1-v4 单向量接口: feats [B,T,80] + emb [B,192] → logits [B,T,3]
// v5 多帧接口:      feats [B,T,80] + enroll_tokens [B,N,192] + enroll_mask [B,N] → logits [B,T,3]
// 加载时按输入名自动检测；v5 模型须传 tokens（空则回退为 {emb} 单 token）
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

    bool use_tokens() const { return use_tokens_; }

    // feats: T*80（已 per-bin 均值归一化），emb: 192 维质心，tokens: 多帧 enrollment
    // tokens（v5 模型用；空或 nullptr 时回退 {emb}）。返回 T 个 P(target)。
    std::vector<float> target_probs(const float* feats, int T, const float* emb,
                                    const std::vector<std::vector<float>>* tokens = nullptr);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_feats_, in_emb_, out_logits_;
    bool use_tokens_ = false;
    std::string in_tokens_, in_mask_;
};
