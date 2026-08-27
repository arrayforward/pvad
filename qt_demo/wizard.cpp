// wizard.cpp
#include "wizard.h"

const WizardStep WizardController::kSteps[WizardController::kTotal] = {
    {"正常距离（约30cm），用平时说话的音量，照着念：",
     "你好，今天天气不错。我想测试一下，我的声音能不能随时打断它说话。"},
    {"离麦克风远一点（约1米），稍微提高音量，照着念：",
     "一二三四五六七八九十。离得远一点，声音大一点，它也应该能听出是我。"},
    {"回到正常距离，像平时聊天一样自由发挥，随便说几句（≥5秒）：",
     "比如：介绍一下你今天吃了什么、现在几点、接下来要做什么……"},
};

void WizardController::start(DemoCore& core) {
    core.get_enroll_state(backup_sum_, backup_n_, backup_segs_);
    core.get_fbank_state(backup_fsum_, backup_fframes_);
    core.clear_enroll();
    step_ = 0;
    active_ = true;
}

bool WizardController::accept_segment(DemoCore& core, const float* pcm, size_t n,
                                      const std::string& wav, double duration_s,
                                      const std::string& time) {
    if (!active_ || step_ >= kTotal) return false;
    if (n < kMinSamples) return false;  // 太短，重录本段
    std::string err;
    if (!core.enroll_samples(pcm, n, err, wav, duration_s, time)) return false;
    step_++;
    if (step_ >= kTotal) active_ = false;
    return true;
}

void WizardController::cancel(DemoCore& core) {
    if (!active_) return;
    core.set_enroll_state(backup_sum_, backup_n_, backup_segs_);
    core.set_fbank_state(backup_fsum_, backup_fframes_);
    step_ = 0;
    active_ = false;
}
