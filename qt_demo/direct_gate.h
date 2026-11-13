// direct_gate.h - tse pvad_direct 句级二级复核（ONNX FP32，当前 direct_v4 官方点）
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
    // 官方工作点 TH=0.23/UNC=0.2；真人 held-out 口径见 docs/REGRESSION.md。
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
    void setThreshold(float t) { threshold_ = t; }  // 界面实时调整（确认门）
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    float threshold_;
    float uncertain_;
};
