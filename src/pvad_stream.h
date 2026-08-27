// pvad_stream.h - 流式 PVAD（chunked GRU state 复用 + EMA CMVN）
// 模型 pvad_v4_stream.onnx：FiLM 条件，state 外置
//   输入 feats_chunk[B,t,80] + emb[B,192] + h0[2,B,128] → logits[B,t,3] + hN[2,B,128]
// 调用约定（models/pvad/pvad_v4_stream.md）：
//   - CMVN 由本类负责：EMA α=0.02，per-bin 指数滑动均值（double 累积，与 python 一致）
//   - chunk=5 帧（50ms）推理一次；会话起始 h=0、EMA=0
//   - 前 50 帧（0.5s）为 warm-up：照样返回 P(target) 但 gated=false（不应门控）
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

class PvadStream {
public:
    struct Out {
        float p = 0.f;      // 本帧 P(target)（warm-up 期也可能非 0，仅供参考）
        bool valid = false; // 首个 chunk 凑满前为 false（无分数）
        bool gated = false; // true = 已过 warm-up，可过门控
        size_t frame = 0;   // 该分数对应的流内绝对帧号（推理有 0-4 帧滞后）
    };

    explicit PvadStream(const std::string& model_path, int chunk = 5);
    ~PvadStream();
    PvadStream(const PvadStream&) = delete;
    PvadStream& operator=(const PvadStream&) = delete;

    // 设置 warm-up 帧数（默认 50 = 0.5s，期间分数不门控）
    void set_warmup(size_t frames) { warmup_ = frames; }

    // 设置 enrollment embedding（192 维，L2 归一化）；注册变化后须重新设置
    void set_emb(const float* emb192);
    // 设置 CMVN 先验均值（80 维 fbank 均值，如 enrollment 音频的均值）。
    // 会话 reset 时 EMA 从该值起步而非 0——显著抑制冷启动假阳（python 验证，见
    // pvad_v4_stream.md 的 ema_prior 变体）；未设置时按训练约定从 0 起步。
    void set_cmvn_prior(const float* m80);

    // 送入一帧 80 维原始 fbank（未归一化），返回本帧结果（内部 EMA CMVN + 凑 chunk 推理，
    // 逐帧对齐：首个 chunk 凑满前的帧返回 p=0, gated=false）。
    Out push_frame(const float* fbank80);
    void reset();  // 新会话：GRU state / EMA / warm-up 计数全清零
    size_t frames() const { return out_frame_; }
    static constexpr size_t kWarmupFrames = 50;  // 0.5s warm-up 不门控

private:
    void run_chunk();

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_feats_, in_emb_, in_h0_, out_logits_, out_hn_;
    int chunk_;
    size_t warmup_ = kWarmupFrames;
    std::vector<float> h_;       // [2*1*128] GRU state
    std::vector<float> emb_;     // enrollment embedding（192）
    std::vector<double> m_;      // EMA 均值（80 维）
    std::vector<double> prior_;  // CMVN 先验（空 = 按训练约定从 0 起步）
    std::vector<float> buf_;     // 待凑 chunk 的 CMVN 后特征（chunk_*80）
    std::vector<float> probs_;   // 最近 chunk 的逐帧 P(target)，逐帧取出
    size_t prob_next_ = 0;
    size_t out_frame_ = 0;   // 已发出分数的绝对帧号
};
