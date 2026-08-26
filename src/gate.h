// gate.h - 门控：AS-norm z 分数 + 负模板 margin 双判据（迟滞防抖）
#pragma once

struct GateOptions {
    float threshold = 0.55f;    // --no-norm 模式: sA_raw 触发阈值
    float margin = 0.15f;       // sA_raw - sNeg 最小间隔（两种模式都生效）
    float z_threshold = 3.0f;   // 归一化模式: sA_norm = (sA_raw-mu)/sigma 的触发阈值
    float hysteresis = 0.1f;    // sA_raw 低于 threshold - hysteresis 时计数清零（no-norm 模式）
    float z_hysteresis = 0.5f;  // 归一化模式: z 低于 z_threshold - z_hysteresis 时计数清零
    int confirm = 2;            // 连续 N 次满足条件才触发
    bool use_norm = true;       // false 退回纯余弦模式（--no-norm）
};

class Gate {
public:
    explicit Gate(const GateOptions& opt = GateOptions()) : opt_(opt) {}

    // 每次得到 (sA_raw, sNeg, sA_norm) 时调用；返回 true 表示本次触发 INTERRUPT（沿触发）
    bool update(float sA_raw, float sNeg, float sA_norm) {
        float primary = opt_.use_norm ? sA_norm : sA_raw;
        float thr = opt_.use_norm ? opt_.z_threshold : opt_.threshold;
        float hyst = opt_.use_norm ? opt_.z_hysteresis : opt_.hysteresis;
        if (primary > thr && (sA_raw - sNeg) > opt_.margin) {
            consec_++;
            if (consec_ >= opt_.confirm && !triggered_) {
                triggered_ = true;
                return true;
            }
        } else if (primary < thr - hyst) {
            consec_ = 0;
            triggered_ = false;
        }
        return false;
    }

    void reset() { consec_ = 0; triggered_ = false; }
    int consec() const { return consec_; }
    bool triggered() const { return triggered_; }

private:
    GateOptions opt_;
    int consec_ = 0;
    bool triggered_ = false;
};

// PVAD 模式门控：作用于帧级 P(target) 序列（与 scripts/eval_pvad.py 的
// pvad_gate_trigger 相同：thr=0.5, hyst=0.2, confirm=2）
class PvadGate {
public:
    explicit PvadGate(float thr = 0.5f, float hyst = 0.2f, int confirm = 2)
        : thr_(thr), hyst_(hyst), confirm_(confirm) {}

    bool update(float p) {
        if (p > thr_) {
            consec_++;
            if (consec_ >= confirm_ && !triggered_) {
                triggered_ = true;
                return true;
            }
        } else if (p < thr_ - hyst_) {
            consec_ = 0;
            triggered_ = false;
        }
        return false;
    }

    void reset() { consec_ = 0; triggered_ = false; }
    int consec() const { return consec_; }

private:
    float thr_, hyst_;
    int confirm_;
    int consec_ = 0;
    bool triggered_ = false;
};
