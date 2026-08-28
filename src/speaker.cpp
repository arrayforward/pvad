// speaker.cpp
#include "speaker.h"
#include "fbank.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>

static std::wstring widen2(const std::string& s) {
    std::wstring w(s.begin(), s.end());
    return w;
}

SpeakerEmbedder::SpeakerEmbedder(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "spk");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);
    session_ = std::make_unique<Ort::Session>(*env_, widen2(model_path).c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in = session_->GetInputNameAllocated(0, alloc);
    auto out = session_->GetOutputNameAllocated(0, alloc);
    in_name_ = in.get();
    out_name_ = out.get();
    auto shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() >= 2 && shape.back() > 0) dim_ = (int)shape.back();
    else dim_ = 192;
}

SpeakerEmbedder::~SpeakerEmbedder() = default;

std::vector<float> SpeakerEmbedder::embed(const float* pcm, int num_samples) {
    Fbank fbank;
    std::vector<float> feats;
    int T = fbank.compute(pcm, num_samples, feats);
    if (T < 4) throw std::runtime_error("audio too short for embedding");
    mean_normalize(feats, T, 80);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[3] = {1, T, 80};
    Ort::Value t_in = Ort::Value::CreateTensor<float>(mem, feats.data(), feats.size(), shape, 3);
    const char* in_names[1] = {in_name_.c_str()};
    const char* out_names[1] = {out_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &t_in, 1, out_names, 1);

    auto oshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t total = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float* p = outputs[0].GetTensorMutableData<float>();
    std::vector<float> emb(p, p + total);
    if ((int)emb.size() != dim_) dim_ = (int)emb.size();
    l2_normalize(emb);
    return emb;
}

std::vector<std::vector<float>> SpeakerEmbedder::embed_tokens(const float* pcm, int num_samples) {
    std::vector<std::vector<float>> toks;
    int n = num_samples / 16000;  // 整 1s 切分，尾段丢弃
    toks.reserve(n);
    for (int i = 0; i < n; i++) toks.push_back(embed(pcm + (size_t)i * 16000, 16000));
    return toks;
}

void l2_normalize(std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += (double)x * x;
    float n = (float)sqrt(s);
    if (n > 1e-8f) for (float& x : v) x /= n;
}

float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.f;
    return (float)(dot / sqrt(na * nb));
}

float max_neg_sim(const Template& tpl, const std::vector<float>& emb) {
    float s = 0.f;
    for (auto& kv : tpl.neg) s = std::max(s, cosine_sim(kv.second, emb));
    return s;
}

void tnorm_stats(const Template& tpl, const std::vector<float>& emb, int topk, float& mu, float& sigma) {
    if (tpl.cohort.empty()) { mu = 0.f; sigma = 1.f; return; }
    std::vector<float> sims;
    sims.reserve(tpl.cohort.size());
    for (auto& c : tpl.cohort) sims.push_back(cosine_sim(c, emb));
    int k = topk > 0 ? std::min(topk, (int)sims.size()) : (int)sims.size();
    std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), std::greater<float>());
    double m = 0;
    for (int i = 0; i < k; i++) m += sims[i];
    m /= k;
    double v = 0;
    for (int i = 0; i < k; i++) v += (sims[i] - m) * (sims[i] - m);
    v /= k;
    mu = (float)m;
    sigma = (float)sqrt(v);
}

void save_template(const std::string& path, const Template& tpl) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write template: " + path);
    int32_t version = 4, dim = (int32_t)tpl.pos.size();
    fwrite(&version, 4, 1, f);
    fwrite(&dim, 4, 1, f);
    fwrite(tpl.pos.data(), 4, dim, f);
    int32_t n_neg = (int32_t)tpl.neg.size();
    fwrite(&n_neg, 4, 1, f);
    for (auto& kv : tpl.neg) {
        int32_t len = (int32_t)kv.first.size();
        fwrite(&len, 4, 1, f);
        fwrite(kv.first.data(), 1, len, f);
        fwrite(kv.second.data(), 4, dim, f);
    }
    int32_t n_cohort = (int32_t)tpl.cohort.size();
    fwrite(&n_cohort, 4, 1, f);
    for (auto& c : tpl.cohort) fwrite(c.data(), 4, dim, f);
    // v4 追加：多帧 enrollment tokens
    int32_t n_tok = (int32_t)tpl.tokens.size();
    fwrite(&n_tok, 4, 1, f);
    for (auto& t : tpl.tokens) fwrite(t.data(), 4, dim, f);
    fclose(f);
}

Template load_template(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open template: " + path);
    int32_t version = 0, dim = 0;
    if (fread(&version, 4, 1, f) != 1 || (version != 3 && version != 4) ||
        fread(&dim, 4, 1, f) != 1 || dim <= 0 || dim > 4096) {
        fclose(f);
        throw std::runtime_error("bad or old-version template file: " + path);
    }
    Template tpl;
    tpl.pos.resize(dim);
    if (fread(tpl.pos.data(), 4, dim, f) != (size_t)dim) {
        fclose(f);
        throw std::runtime_error("bad template file: " + path);
    }
    int32_t n_neg = 0;
    if (fread(&n_neg, 4, 1, f) != 1 || n_neg < 0 || n_neg > 64) {
        fclose(f);
        throw std::runtime_error("bad template file: " + path);
    }
    for (int32_t i = 0; i < n_neg; i++) {
        int32_t len = 0;
        if (fread(&len, 4, 1, f) != 1 || len <= 0 || len > 256) {
            fclose(f);
            throw std::runtime_error("bad template file: " + path);
        }
        std::string label(len, '\0');
        std::vector<float> c(dim);
        if (fread(&label[0], 1, len, f) != (size_t)len ||
            fread(c.data(), 4, dim, f) != (size_t)dim) {
            fclose(f);
            throw std::runtime_error("bad template file: " + path);
        }
        tpl.neg.push_back({label, std::move(c)});
    }
    int32_t n_cohort = 0;
    if (fread(&n_cohort, 4, 1, f) != 1 || n_cohort < 0 || n_cohort > 4096) {
        fclose(f);
        throw std::runtime_error("bad template file: " + path);
    }
    for (int32_t i = 0; i < n_cohort; i++) {
        std::vector<float> c(dim);
        if (fread(c.data(), 4, dim, f) != (size_t)dim) {
            fclose(f);
            throw std::runtime_error("bad template file: " + path);
        }
        tpl.cohort.push_back(std::move(c));
    }
    if (version == 4) {
        int32_t n_tok = 0;
        if (fread(&n_tok, 4, 1, f) != 1 || n_tok < 0 || n_tok > 4096) {
            fclose(f);
            throw std::runtime_error("bad template file: " + path);
        }
        for (int32_t i = 0; i < n_tok; i++) {
            std::vector<float> t(dim);
            if (fread(t.data(), 4, dim, f) != (size_t)dim) {
                fclose(f);
                throw std::runtime_error("bad template file: " + path);
            }
            tpl.tokens.push_back(std::move(t));
        }
    }
    fclose(f);
    return tpl;
}
