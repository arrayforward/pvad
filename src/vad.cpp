// vad.cpp
#include "vad.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

static std::wstring widen(const std::string& s) {
    std::wstring w(s.begin(), s.end());
    return w;
}

Vad::Vad(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "vad");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(*env_, widen(model_path).c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    // 输入名
    size_t n_in = session_->GetInputCount();
    for (size_t i = 0; i < n_in; i++) {
        auto name = session_->GetInputNameAllocated(i, alloc);
        std::string n = name.get();
        if (n == "input") in_name_input_ = n;
        else if (n == "state") in_name_state_ = n;
        else if (n == "sr") in_name_sr_ = n;
        else if (in_name_input_.empty()) in_name_input_ = n;
        else if (in_name_state_.empty()) in_name_state_ = n;
        else if (in_name_sr_.empty()) in_name_sr_ = n;
    }
    size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_out; i++) {
        auto name = session_->GetOutputNameAllocated(i, alloc);
        std::string n = name.get();
        if (n == "output") out_name_prob_ = n;
        else if (n == "stateN") out_name_state_ = n;
        else if (out_name_prob_.empty()) out_name_prob_ = n;
        else if (out_name_state_.empty()) out_name_state_ = n;
    }
    reset();
}

Vad::~Vad() = default;

void Vad::reset() {
    state_.assign(2 * 128, 0.f);
    ctx_.assign(64, 0.f);
    buf_.clear();
    last_prob_ = -1.f;
}

float Vad::process(const float* samples, int n) {
    buf_.insert(buf_.end(), samples, samples + n);
    while (buf_.size() >= 512) run_512();
    return last_prob_;
}

void Vad::run_512() {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // silero v5 官方用法：输入 = 64 采样 context（上一帧末尾） + 512 采样新帧
    int64_t in_shape[2] = {1, 576};
    int64_t st_shape[3] = {2, 1, 128};
    int64_t sr_shape[1] = {1};
    int64_t sr_val[1] = {16000};

    float x[576];
    std::copy(ctx_.begin(), ctx_.end(), x);
    std::copy(buf_.begin(), buf_.begin() + 512, x + 64);
    std::copy(x + 512, x + 576, ctx_.begin());

    Ort::Value t_input = Ort::Value::CreateTensor<float>(mem, x, 576, in_shape, 2);
    Ort::Value t_state = Ort::Value::CreateTensor<float>(mem, state_.data(), state_.size(), st_shape, 3);
    Ort::Value t_sr = Ort::Value::CreateTensor<int64_t>(mem, sr_val, 1, sr_shape, 1);

    const char* in_names[3] = {in_name_input_.c_str(), in_name_state_.c_str(), in_name_sr_.c_str()};
    const char* out_names[2] = {out_name_prob_.c_str(), out_name_state_.c_str()};
    Ort::Value inputs[3] = {std::move(t_input), std::move(t_state), std::move(t_sr)};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 2);
    last_prob_ = outputs[0].GetTensorMutableData<float>()[0];
    float* new_state = outputs[1].GetTensorMutableData<float>();
    std::copy(new_state, new_state + state_.size(), state_.begin());
    buf_.erase(buf_.begin(), buf_.begin() + 512);
}
