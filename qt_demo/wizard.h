// wizard.h - 引导式注册状态机（Qt 无关，可无头测试）
// idle → (start: 备份旧注册并清空) step0 → step1 → step2 → 完成(active_=false)
//   任意时刻 cancel → 恢复备份的旧注册，active_=false
// 单段 <2s 拒收（返回 false，不计入进度，可重录本段）。
#pragma once
#include <cstddef>
#include <vector>
#include "demo_core.h"

struct WizardStep {
    const char* hint;  // 场景提示
    const char* text;  // 念词文本（第 3 段为自由发挥示例）
};

class WizardController {
public:
    static constexpr int kTotal = 3;
    static constexpr size_t kMinSamples = 16000 * 2;  // 单段 <2s 拒收
    static const WizardStep kSteps[kTotal];

    bool active() const { return active_; }
    int step() const { return step_; }  // 0..kTotal；==kTotal 表示已完成

    // 开始向导：备份当前注册状态并清空（开始全新注册）
    void start(DemoCore& core);
    // 录入一段：<2s 返回 false（重录本段）；成功则 enroll_samples 且 step_++，
    // 最后一段完成后 active_ 自动变 false。wav/time 为可选落盘元数据。
    bool accept_segment(DemoCore& core, const float* pcm, size_t n,
                        const std::string& wav = "", double duration_s = 0,
                        const std::string& time = "");
    // 取消向导：恢复 start() 时备份的旧注册状态
    void cancel(DemoCore& core);

private:
    std::vector<float> backup_sum_;
    int backup_n_ = 0;
    std::vector<SegRecord> backup_segs_;  // 逐段明细也须备份，恢复才完整
    int step_ = 0;
    bool active_ = false;
};
