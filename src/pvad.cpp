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
    // ERROR 级别: 屏蔽导出时示例维度带来的输出 shape 校验警告（动态 T 实际正常）
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "pvad");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(2);
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_ = std::make_unique<Ort::Session>(*env_, widen3(model_path).c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = session_->GetInputCount();
    for (size_t i = 0; i < n_in; i++) {
        auto name = session_->GetInputNameAllocated(i, alloc);
        std::string n = name.get();
        if (n == "feats") in_feats_ = n;
        else if (n == "emb") in_emb_ = n;
        else if (n == "enroll_tokens") { in_tokens_ = n; use_tokens_ = true; }
        else if (n == "enroll_mask") in_mask_ = n;
        else if (in_feats_.empty()) in_feats_ = n;
        else if (in_emb_.empty()) in_emb_ = n;
    }
    auto o0 = session_->GetOutputNameAllocated(0, alloc);
    out_logits_ = o0.get();
    if (in_feats_.empty()) throw std::runtime_error("pvad: no feats input found");
}

Pvad::~Pvad() = default;

std::vector<float> Pvad::target_probs(const float* feats, int T, const float* emb,
                                      const std::vector<std::vector<float>>* tokens) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t fshape[3] = {1, T, 80};
    Ort::Value t_feats = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(feats), (size_t)T * 80, fshape, 3);

    auto run = [&](std::vector<Ort::Value>& inputs, std::vector<const char*>& names) {
        const char* out_names[1] = {out_logits_.c_str()};
        return session_->Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), out_names, 1);
    };

    std::vector<Ort::Value> outputs;
    if (!use_tokens_) {
        int64_t eshape[2] = {1, 192};
        Ort::Value t_emb = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(emb), 192, eshape, 2);
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(t_feats));
        inputs.push_back(std::move(t_emb));
        std::vector<const char*> names = {in_feats_.c_str(), in_emb_.c_str()};
        outputs = run(inputs, names);
    } else {
        // v5: enroll_tokens [1,N,192] + enroll_mask [1,N]（True=padding；全 False = 无 padding）
        std::vector<float> flat;
        size_t N = tokens && !tokens->empty() ? tokens->size() : 1;
        flat.reserve(N * 192);
        if (tokens && !tokens->empty()) {
            for (auto& t : *tokens) flat.insert(flat.end(), t.begin(), t.end());
        } else {
            flat.insert(flat.end(), emb, emb + 192);  // 回退：质心作单 token
        }
        int64_t tshape[3] = {1, (int64_t)N, 192};
        int64_t mshape[2] = {1, (int64_t)N};
        auto mask = std::make_unique<bool[]>(N);  // vector<bool>::data() 被删除，用数组
        std::fill(mask.get(), mask.get() + N, false);
        Ort::Value t_tokens = Ort::Value::CreateTensor<float>(mem, flat.data(), flat.size(), tshape, 3);
        Ort::Value t_mask = Ort::Value::CreateTensor<bool>(mem, mask.get(), N, mshape, 2);
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(t_feats));
        inputs.push_back(std::move(t_tokens));
        inputs.push_back(std::move(t_mask));
        std::vector<const char*> names = {in_feats_.c_str(), in_tokens_.c_str(), in_mask_.c_str()};
        outputs = run(inputs, names);
    }

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
