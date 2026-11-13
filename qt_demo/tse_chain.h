// tse_chain.h - TSE 链：多尺度声纹编码器（4288 维）+ e16 step 流式提取
// 移植自 tse-gitee/android_demo 的 SpeakerEncoder.kt / TseEngine.kt（只读参考，不改原仓）。
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

// 多尺度声纹编码器：ERB 功率 [1,1,T,32] -> 4288 维
class TseSpeakerEncoder {
public:
    explicit TseSpeakerEncoder(const std::string& model_path);
    ~TseSpeakerEncoder();
    TseSpeakerEncoder(const TseSpeakerEncoder&) = delete;
    TseSpeakerEncoder& operator=(const TseSpeakerEncoder&) = delete;
    static constexpr int EMB_DIM = 4288;
    // 多段 16k 单声道 -> 逐段提取后均值（与 Kotlin encode() 一致）。
    // apply_agc=false 供 direct_v4 二级复核复现其原始训练输入；TSE 默认 true。
    std::vector<float> encode(const std::vector<std::vector<float>>& segments,
                              bool apply_agc = true);
    // direct_v4：3s 窗 / 1s hop，短句零Padding到3s；返回每窗 deep-192（不做音频 AGC）。
    std::vector<std::vector<float>> encode_windows(const float* pcm, int n);
private:
    std::vector<float> encode_one(const float* pcm, int n, bool apply_agc);
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

// e16 step 流式提取：28 个外置状态 + emb 注入
class TseEngine {
public:
    TseEngine(const std::string& model_path, const std::vector<float>& emb);
    ~TseEngine();
    TseEngine(const TseEngine&) = delete;
    TseEngine& operator=(const TseEngine&) = delete;
    void reset();
    static constexpr int CHUNK = 50;  // 帧 (500ms)
    // 输入 4 通道一个 chunk 的复数谱（[4*CHUNK*257] 实/虚部），返回参考通道提取谱
    void step(const float* xr, const float* xi, std::vector<float>& sr, std::vector<float>& si);
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> emb_;
    std::vector<std::vector<float>> states_;  // 28 个状态，按名称序
};

struct AgcReport;
// 整段提取：单声道 pcm → AGC(有效语音 RMS=0.045) → STFT → 4ch 铺展 → 逐 chunk step → ISTFT
// report 非空时返回 AGC 前后电平与实际增益。
std::vector<float> tse_extract(TseEngine& eng, const float* pcm, int n,
                               AgcReport* report = nullptr);
