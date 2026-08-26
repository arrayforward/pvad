// demo_core.h - Qt 无关的管线核心：注册质心 + PVAD 流式打分 + PvadGate
// 复用 src/ 的 fbank/speaker/pvad/gate。与 src/main.cpp 实时 pvad 路径一致：
// 每 10ms 帧把采样追加到当前流（封顶 8s），整段过 GRU（前缀均值归一化），
// 取最后一帧 P(target) 过迟滞门控。
#pragma once
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "gate.h"

class SpeakerEmbedder;
class Pvad;

struct FrameEvent {
    float p = 0.f;          // 本帧 P(target)
    int consec = 0;         // 门控连续计数
    bool interrupt = false; // 本帧是否触发 INTERRUPT
    double t = 0.0;         // 虚拟时间戳（秒）
};

class DemoCore {
public:
    DemoCore();
    ~DemoCore();
    bool init(const std::string& spk_model, const std::string& pvad_model, std::string& err);
    // 对若干 16k 单声道 wav 做 enrollment（L2 归一化后累加进注册集合，质心 = 均值归一化）
    bool enroll(const std::vector<std::string>& wavs, std::string& err);
    // 追加一段裸 16k 音频到注册集合（录音注册用）
    bool enroll_samples(const float* pcm, size_t n, std::string& err);
    void clear_enroll();
    bool enrolled() const { return has_tpl_; }
    int enroll_count() const { return n_emb_; }

    FrameEvent feed_frame(const float* frame160);
    // 文件/注入模式：整段音频一次性预计算 P(target) 序列（整段均值归一化 + 单次 GRU 前向，
    // 与训练/python 评估一致）。之后 feed_frame 按帧查表。
    // 流式(mic)模式不可用整段统计，只能前缀归一化——前 ~0.5s 统计不稳会虚高 P(target)，
    // 故流式模式做 0.5s warm-up（期间报数但不更新门控）。
    bool precompute_file(const float* pcm, size_t n, std::string& err);
    void reset_stream();   // 清空流与门控（换注入源/重新监听时调用）
    double now() const { return frame_idx_ * 0.01; }

private:
    std::unique_ptr<SpeakerEmbedder> spk_;
    std::unique_ptr<Pvad> pvad_;
    std::vector<float> centroid_;
    std::vector<float> emb_sum_;   // 注册集合：各段 embedding（L2 归一化）的累加
    int n_emb_ = 0;
    bool has_tpl_ = false;
    PvadGate gate_{0.5f, 0.2f, 2};
    std::deque<float> seg_;
    std::vector<float> p2_pre_;      // 文件模式预计算序列（空 = 流式模式）
    size_t frame_idx_ = 0;
    static constexpr size_t kSegCap = 16000 * 8;
    static constexpr size_t kWarmupFrames = 50;  // 流式模式 warm-up（0.5s）
};
