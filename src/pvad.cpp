// pvad.cpp
#include "pvad.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

static std::wstring widen3(const std::string& s) {
    std::wstring w(s.begin(), s.end());
    return w;
}

Pvad::Pvad(const std::string& model_path) {
    // ERROR 级别: 屏蔽导出时示例维度 137 带来的输出 shape 校验警告（动态 T 实际正常）
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "pvad");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(2);
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_ = std::make_unique<Ort::Session>(*env_, widen3(model_path).c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto i0 = session_->GetInputNameAllocated(0, alloc);
    auto i1 = session_->GetInputNameAllocated(1, alloc);
    auto o0 = session_->GetOutputNameAllocated(0, alloc);
    in_feats_ = i0.get();
    in_emb_ = i1.get();
    out_logits_ = o0.get();
    // 按名字对齐（不依赖顺序）
    if (in_feats_ == "emb") std::swap(in_feats_, in_emb_);
}

Pvad::~Pvad() = default;

std::vector<float> Pvad::target_probs(const float* feats, int T, const float* emb) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t fshape[3] = {1, T, 80};
    int64_t eshape[2] = {1, 192};
    Ort::Value t_feats = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(feats), (size_t)T * 80, fshape, 3);
    Ort::Value t_emb = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(emb), 192, eshape, 2);
    const char* in_names[2] = {in_feats_.c_str(), in_emb_.c_str()};
    const char* out_names[1] = {out_logits_.c_str()};
    Ort::Value inputs[2] = {std::move(t_feats), std::move(t_emb)};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, T', 3]
    int Tout = (int)shape[1];
    const float* logits = outputs[0].GetTensorData<float>();
    std::vector<float> p2(Tout);
    for (int t = 0; t < Tout; t++) {
        float l0 = logits[t * 3], l1 = logits[t * 3 + 1], l2 = logits[t * 3 + 2];
        float m = std::max(l0, std::max(l1, l2));
        float e0 = expf(l0 - m), e1 = expf(l1 - m), e2 = expf(l2 - m);
        p2[t] = e2 / (e0 + e1 + e2);
    }
    return p2;
}
