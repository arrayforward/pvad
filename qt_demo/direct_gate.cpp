// direct_gate.cpp
#include "direct_gate.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

namespace {
std::wstring widen_direct(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
}

DirectGate::DirectGate(const std::string& model_path, float threshold, float uncertain)
    : threshold_(threshold), uncertain_(uncertain) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "direct_v3");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_ = std::make_unique<Ort::Session>(*env_, widen_direct(model_path).c_str(), so);
}

DirectGate::~DirectGate() = default;

DirectGateDecision DirectGate::decide(
        const std::vector<std::vector<float>>& windows,
        const std::vector<float>& reference) const {
    if (windows.empty()) throw std::runtime_error("direct gate: no windows");
    if (reference.size() != 192) throw std::runtime_error("direct gate: reference dim mismatch");
    std::vector<float> flat;
    flat.reserve(windows.size() * 192);
    for (const auto& w : windows) {
        if (w.size() != 192) throw std::runtime_error("direct gate: window dim mismatch");
        flat.insert(flat.end(), w.begin(), w.end());
    }
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t wshape[2] = {(int64_t)windows.size(), 192};
    int64_t rshape[1] = {192};  // 导出接口 reference 是一维 [192]，不是 [1,192]
    Ort::Value tw = Ort::Value::CreateTensor<float>(
        mem, flat.data(), flat.size(), wshape, 2);
    Ort::Value tr = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(reference.data()), reference.size(), rshape, 1);
    const char* names[2] = {"windows", "reference"};
    const char* outputs[2] = {"identity", "confidence"};
    Ort::Value inputs[2] = {std::move(tw), std::move(tr)};
    auto out = session_->Run(Ort::RunOptions{nullptr}, names, inputs, 2, outputs, 2);
    DirectGateDecision d;
    d.identity = out[0].GetTensorData<float>()[0];
    d.confidence = out[1].GetTensorData<float>()[0];
    d.closed = d.identity < threshold_ && d.confidence >= uncertain_;
    return d;
}
