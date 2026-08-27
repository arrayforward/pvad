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

// 一段注册的明细（持久化到 enrollment/segments.json）
struct SegRecord {
    std::string wav;        // 来源 wav 路径（录音为 recordings/rec_*.wav）
    double duration_s = 0;
    std::string time;       // 录入时间 "yyyy-MM-dd HH:mm:ss"
    std::vector<float> emb; // 该段 L2 归一化 embedding（192 维）
};

class DemoCore {
public:
    DemoCore();
    ~DemoCore();
    bool init(const std::string& spk_model, const std::string& pvad_model, std::string& err);
    // 对若干 16k 单声道 wav 做 enrollment（L2 归一化后累加进注册集合，质心 = 均值归一化）
    bool enroll(const std::vector<std::string>& wavs, std::string& err);
    // 追加一段裸 16k 音频到注册集合（录音注册用；可带来源元数据用于落盘）
    bool enroll_samples(const float* pcm, size_t n, std::string& err,
                        const std::string& wav = "", double duration_s = 0,
                        const std::string& time = "");
    void clear_enroll();
    // enrollment 的 fbank 均值（流式 CMVN 先验用；空 = 不可用）
    std::vector<float> fbank_mean() const;
    void get_fbank_state(std::vector<double>& sum, size_t& frames) const {
        sum = fbank_sum_;
        frames = fbank_frames_;
    }
    void set_fbank_state(const std::vector<double>& sum, size_t frames) {
        fbank_sum_ = sum;
        fbank_frames_ = frames;
    }
    // 注册状态备份/恢复（引导注册"取消则恢复旧质心"语义用）
    void get_enroll_state(std::vector<float>& emb_sum, int& n,
                          std::vector<SegRecord>& segs) const {
        emb_sum = emb_sum_;
        n = n_emb_;
        segs = segs_;
    }
    void set_enroll_state(const std::vector<float>& emb_sum, int n,
                          const std::vector<SegRecord>& segs);
    // 从逐段记录精确重建（emb_sum 按序累加，与增量注册逐位一致）
    void set_segments(const std::vector<SegRecord>& segs);
    const std::vector<SegRecord>& segments() const { return segs_; }
    const std::vector<float>& centroid() const { return centroid_; }
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
    void append_segment(const std::vector<float>& emb, const std::string& wav,
                        double duration_s, const std::string& time);

    std::unique_ptr<SpeakerEmbedder> spk_;
    std::unique_ptr<Pvad> pvad_;
    std::vector<float> centroid_;
    std::vector<float> emb_sum_;   // 注册集合：各段 embedding（L2 归一化）的累加
    std::vector<SegRecord> segs_;  // 逐段明细（持久化用）
    std::vector<double> fbank_sum_;  // 注册音频 fbank 累加（80 维，流式 CMVN 先验用）
    size_t fbank_frames_ = 0;
    int n_emb_ = 0;
    bool has_tpl_ = false;
    PvadGate gate_{0.5f, 0.2f, 2};
    std::deque<float> seg_;
    std::vector<float> p2_pre_;      // 文件模式预计算序列（空 = 流式模式）
    size_t frame_idx_ = 0;
    static constexpr size_t kSegCap = 16000 * 8;
    static constexpr size_t kWarmupFrames = 50;  // 流式模式 warm-up（0.5s）
};
