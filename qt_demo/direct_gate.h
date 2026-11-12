// direct_gate.h - tse pvad_direct_v3 句级二级复核（ONNX FP32）
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

struct DirectGateDecision {
    float identity = 0.f;
    float confidence = 0.f;
    bool closed = false;  // true=拒绝进入 TSE/ASR；不确定区始终放行
};

class DirectGate {
public:
    // 高召回工作点：当前域 A 放行约 97%/95.5%，牺牲强拦截以尽量少误杀用户。
    explicit DirectGate(const std::string& model_path,
                        float threshold = 0.0f, float uncertain = 0.3f);
    ~DirectGate();
    DirectGate(const DirectGate&) = delete;
    DirectGate& operator=(const DirectGate&) = delete;

    // windows: N 个 3s 窗的 Multiscale deep-192；reference: 注册 deep-192。
    DirectGateDecision decide(const std::vector<std::vector<float>>& windows,
                              const std::vector<float>& reference) const;
    float threshold() const { return threshold_; }
    float uncertain() const { return uncertain_; }
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    float threshold_;
    float uncertain_;
};
