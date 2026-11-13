// tse_chain.cpp
#include "tse_chain.h"
#include "agc.h"
#include "dsp_tse.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

using namespace tse_dsp;

static std::wstring widen_tse(const std::string& s) { return std::wstring(s.begin(), s.end()); }

TseSpeakerEncoder::TseSpeakerEncoder(const std::string& model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "tse_enc");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(2);
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_ = std::make_unique<Ort::Session>(*env_, widen_tse(model_path).c_str(), so);
}

TseSpeakerEncoder::~TseSpeakerEncoder() = default;

std::vector<float> TseSpeakerEncoder::encode_one(const float* pcm, int n, bool apply_agc) {
    // 只归一化音频电平；4288 维嵌入本身保持原样，严禁再做 L2 归一化。
    std::vector<float> normalized = apply_agc
        ? agc_normalize(pcm, (size_t)n) : std::vector<float>(pcm, pcm + n);
    std::vector<std::vector<float>> re, im;
    stft(normalized.data(), n, re, im);
    int T = (int)re.size();
    std::vector<float> flat((size_t)T * ERB_BANDS);
    for (int f = 0; f < T; f++)
        for (int b = 0; b < ERB_BANDS; b++) {
            float p = 0.f;
            for (int k = 0; k < F; k++) p += (re[f][k] * re[f][k] + im[f][k] * im[f][k]) * erb_matrix()[b][k];
            flat[(size_t)f * ERB_BANDS + b] = p;
        }
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t shape[4] = {1, 1, T, ERB_BANDS};
    Ort::Value tin = Ort::Value::CreateTensor<float>(mem, flat.data(), flat.size(), shape, 4);
    const char* in_names[1] = {"erb_power"};
    const char* out_names[1] = {"emb"};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &tin, 1, out_names, 1);
    const float* p = outs[0].GetTensorData<float>();
    size_t total = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (total != EMB_DIM) throw std::runtime_error("tse emb dim mismatch");
    return std::vector<float>(p, p + total);
}

std::vector<float> TseSpeakerEncoder::encode(
        const std::vector<std::vector<float>>& segments, bool apply_agc) {
    if (segments.empty()) throw std::runtime_error("no segments");
    std::vector<float> out(EMB_DIM, 0.f);
    for (auto& s : segments) {
        auto e = encode_one(s.data(), (int)s.size(), apply_agc);
        for (int i = 0; i < EMB_DIM; i++) out[i] += e[i];
    }
    for (int i = 0; i < EMB_DIM; i++) out[i] /= (float)segments.size();
    return out;
}

std::vector<std::vector<float>> TseSpeakerEncoder::encode_windows(const float* pcm, int n) {
    constexpr int W = 3 * SR, H = SR;
    std::vector<int> starts;
    if (n <= W) {
        starts.push_back(0);
    } else {
        for (int s = 0; s + W <= n; s += H) starts.push_back(s);
        if (starts.back() + W < n) starts.push_back(n - W);
    }
    std::vector<std::vector<float>> out;
    for (int s : starts) {
        std::vector<float> seg(W, 0.f);
        int count = std::min(W, std::max(0, n - s));
        if (count > 0) std::copy(pcm + s, pcm + s + count, seg.begin());
        auto e = encode_one(seg.data(), W, false);  // direct_v4 对照路径：原始电平
        e.resize(192);
        out.push_back(std::move(e));
    }
    return out;
}

// ---- TseEngine ----
namespace {
const std::pair<const char*, std::vector<int64_t>> kStates[] = {
    {"fs_cross_r", {1, 3, 4, 257}}, {"fs_cross_i", {1, 3, 4, 257}},
    {"fs_auto_m", {1, 3, 4, 257}},  {"fs_auto_r", {1, 3, 4, 257}},
    {"enc0", {1, 16, 2, 32}},       {"enc1", {1, 16, 2, 32}},
    {"enc2", {1, 16, 2, 16}},       {"enc3", {1, 16, 2, 8}},
    {"df_conv0", {1, 2, 2, 96}},    {"df_conv1", {1, 16, 2, 96}},
    {"df_history_r", {1, 4, 96}},   {"df_history_i", {1, 4, 96}},
    {"gru0_g0", {1, 1, 256}},       {"gru0_g1", {1, 1, 256}},
    {"gru1_g0", {1, 1, 256}},       {"gru1_g1", {1, 1, 256}},
    {"gru2_g0", {1, 1, 256}},       {"gru2_g1", {1, 1, 256}},
    {"df0_g0", {1, 1, 256}},        {"df0_g1", {1, 1, 256}},
    {"df1_g0", {1, 1, 256}},        {"df1_g1", {1, 1, 256}},
    {"tconv0", {1, 16, 2, 4}},      {"tconv1", {1, 16, 2, 8}},
    {"tconv2", {1, 16, 2, 16}},
    {"pconv0", {1, 32, 2, 8}},      {"pconv1", {1, 32, 2, 16}},
    {"pconv2", {1, 32, 2, 32}},
};
constexpr int kNStates = sizeof(kStates) / sizeof(kStates[0]);
}  // namespace

TseEngine::TseEngine(const std::string& model_path, const std::vector<float>& emb)
    : emb_(emb) {
    if ((int)emb.size() != TseSpeakerEncoder::EMB_DIM)
        throw std::runtime_error("tse emb dim mismatch");
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "tse");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(2);
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_ = std::make_unique<Ort::Session>(*env_, widen_tse(model_path).c_str(), so);
    reset();
}

TseEngine::~TseEngine() = default;

void TseEngine::reset() {
    states_.clear();
    for (int i = 0; i < kNStates; i++) {
        int64_t sz = 1;
        for (auto d : kStates[i].second) sz *= d;
        states_.emplace_back((size_t)sz, 0.f);
    }
}

void TseEngine::step(const float* xr, const float* xi, std::vector<float>& sr, std::vector<float>& si) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int n2 = F * CHUNK;
    int64_t xshape[4] = {1, 4, CHUNK, F};
    Ort::Value txr = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(xr), 4 * n2, xshape, 4);
    Ort::Value txi = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(xi), 4 * n2, xshape, 4);
    int64_t eshape[2] = {1, TseSpeakerEncoder::EMB_DIM};
    Ort::Value temb = Ort::Value::CreateTensor<float>(mem, emb_.data(), emb_.size(), eshape, 2);

    std::vector<Ort::Value> inputs;
    std::vector<const char*> in_names;
    inputs.push_back(std::move(txr)); in_names.push_back("Xr");
    inputs.push_back(std::move(txi)); in_names.push_back("Xi");
    inputs.push_back(std::move(temb)); in_names.push_back("emb");
    std::vector<Ort::Value> state_tensors;
    for (int i = 0; i < kNStates; i++) {
        state_tensors.push_back(Ort::Value::CreateTensor<float>(
            mem, states_[i].data(), states_[i].size(), kStates[i].second.data(),
            kStates[i].second.size()));
        in_names.push_back(kStates[i].first);
    }
    for (auto& t : state_tensors) inputs.push_back(std::move(t));

    std::vector<const char*> out_names;
    std::vector<std::string> out_name_storage = {"Sr", "Si"};
    for (int i = 0; i < kNStates; i++)
        out_name_storage.push_back(std::string("new_") + kStates[i].first);
    for (auto& s : out_name_storage) out_names.push_back(s.c_str());
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                              inputs.size(), out_names.data(), out_names.size());
    const float* psr = outs[0].GetTensorData<float>();
    const float* psi = outs[1].GetTensorData<float>();
    sr.assign(psr, psr + n2);
    si.assign(psi, psi + n2);
    for (int i = 0; i < kNStates; i++) {
        const float* p = outs[2 + i].GetTensorData<float>();
        std::copy(p, p + states_[i].size(), states_[i].begin());
    }
}

std::vector<float> tse_extract(TseEngine& eng, const float* pcm, int n,
                               AgcReport* report) {
    std::vector<float> normalized = agc_normalize(pcm, (size_t)n, 0.045f, report);
    std::vector<std::vector<float>> re, im;
    stft(normalized.data(), n, re, im);
    int T = (int)re.size();
    if (T < TseEngine::CHUNK) return {};
    // 尾块零填充凑满 CHUNK（与参考实现一致：因果模型前 k 帧输出不受影响），
    // 只取前 tail 帧——不丢尾部语音
    int n_chunks = (T + TseEngine::CHUNK - 1) / TseEngine::CHUNK;
    // step 返回的是整个 chunk 的平坦 [TseEngine::CHUNK*257] 谱，按帧拆到各帧
    std::vector<std::vector<float>> sre(T), sim(T);
    std::vector<float> xr(4 * TseEngine::CHUNK * F, 0.f), xi(4 * TseEngine::CHUNK * F, 0.f);
    std::vector<float> sr, si;
    for (int c = 0; c < n_chunks; c++) {
        int base = c * TseEngine::CHUNK;
        int cnt = std::min(TseEngine::CHUNK, T - base);
        std::fill(xr.begin(), xr.end(), 0.f);
        std::fill(xi.begin(), xi.end(), 0.f);
        for (int ch = 0; ch < 4; ch++)
            for (int f = 0; f < cnt; f++) {
                std::copy(re[base + f].begin(), re[base + f].end(),
                          xr.begin() + ((size_t)ch * TseEngine::CHUNK + f) * F);
                std::copy(im[base + f].begin(), im[base + f].end(),
                          xi.begin() + ((size_t)ch * TseEngine::CHUNK + f) * F);
            }
        eng.step(xr.data(), xi.data(), sr, si);
        for (int f = 0; f < cnt; f++) {
            sre[base + f].assign(sr.begin() + (size_t)f * F, sr.begin() + (size_t)(f + 1) * F);
            sim[base + f].assign(si.begin() + (size_t)f * F, si.begin() + (size_t)(f + 1) * F);
        }
    }
    return istft(sre, sim, n);
}
