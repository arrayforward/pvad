// pvad_stream.cpp
#include "pvad_stream.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

static std::wstring widen_s(const std::string& s) {
    std::wstring w(s.begin(), s.end());
    return w;
}

PvadStream::PvadStream(const std::string& model_path, int chunk) : chunk_(chunk) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "pvad_stream");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(2);
    session_ = std::make_unique<Ort::Session>(*env_, widen_s(model_path).c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = session_->GetInputCount();
    for (size_t i = 0; i < n_in; i++) {
        auto name = session_->GetInputNameAllocated(i, alloc);
        std::string n = name.get();
        if (n == "feats_chunk") in_feats_ = n;
        else if (n == "emb") in_emb_ = n;
        else if (n == "h0") in_h0_ = n;
    }
    size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_out; i++) {
        auto name = session_->GetOutputNameAllocated(i, alloc);
        std::string n = name.get();
        if (n == "logits") out_logits_ = n;
        else if (n == "hN") out_hn_ = n;
    }
    if (in_feats_.empty() || in_emb_.empty() || in_h0_.empty() ||
        out_logits_.empty() || out_hn_.empty())
        throw std::runtime_error("pvad_stream: unexpected model io names");
    reset();
}

PvadStream::~PvadStream() = default;

void PvadStream::set_emb(const float* emb192) {
    emb_.assign(emb192, emb192 + 192);
}

void PvadStream::set_cmvn_prior(const float* m80) {
    prior_.assign(m80, m80 + 80);
    m_.assign(prior_.begin(), prior_.end());
}

void PvadStream::reset() {
    h_.assign(2 * 128, 0.f);
    if (prior_.empty()) m_.assign(80, 0.0);
    else m_.assign(prior_.begin(), prior_.end());
    buf_.clear();
    probs_.clear();
    prob_next_ = 0;
    out_frame_ = 0;
}

PvadStream::Out PvadStream::push_frame(const float* fbank80) {
    // EMA CMVN: m = 0.98*m + 0.02*x; feat = x - m（与 python eval_stream.py 一致）
    float feat[80];
    for (int b = 0; b < 80; b++) {
        m_[b] = 0.98 * m_[b] + 0.02 * (double)fbank80[b];
        feat[b] = (float)((double)fbank80[b] - m_[b]);
    }
    buf_.insert(buf_.end(), feat, feat + 80);
    if ((int)buf_.size() >= chunk_ * 80) run_chunk();

    Out o;
    if (prob_next_ < probs_.size()) {
        o.p = probs_[prob_next_++];
        o.valid = true;
        o.frame = out_frame_++;
        o.gated = o.frame >= warmup_;
    }
    return o;
}

void PvadStream::run_chunk() {
    if (emb_.empty()) throw std::runtime_error("pvad_stream: emb not set");
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t fshape[3] = {1, chunk_, 80};
    int64_t eshape[2] = {1, 192};
    int64_t hshape[3] = {2, 1, 128};

    Ort::Value inputs[3] = {
        Ort::Value::CreateTensor<float>(mem, buf_.data(), buf_.size(), fshape, 3),
        Ort::Value::CreateTensor<float>(mem, emb_.data(), emb_.size(), eshape, 2),
        Ort::Value::CreateTensor<float>(mem, h_.data(), h_.size(), hshape, 3),
    };
    const char* in_names[3] = {in_feats_.c_str(), in_emb_.c_str(), in_h0_.c_str()};
    const char* out_names[2] = {out_logits_.c_str(), out_hn_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 2);

    // logits [1, chunk, 3] -> softmax P(2)
    const float* logits = outputs[0].GetTensorData<float>();
    probs_.assign(chunk_, 0.f);
    for (int t = 0; t < chunk_; t++) {
        float l0 = logits[t * 3], l1 = logits[t * 3 + 1], l2 = logits[t * 3 + 2];
        float mx = std::max(l0, std::max(l1, l2));
        float e0 = expf(l0 - mx), e1 = expf(l1 - mx), e2 = expf(l2 - mx);
        probs_[t] = e2 / (e0 + e1 + e2);
    }
    prob_next_ = 0;
    // hN 回传为下一 chunk 的 h0
    const float* hn = outputs[1].GetTensorData<float>();
    std::memcpy(h_.data(), hn, h_.size() * sizeof(float));
    buf_.clear();
}
