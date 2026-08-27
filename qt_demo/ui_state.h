// ui_state.h - 按钮使能/状态转移 + 背压决策（Qt 无关，可无头测试）
#pragma once
#include <cstddef>

// GUI 三态（listening/recording/wizard）与各动作的使能规则。
// MainWindow::updateButtons() 的规则全部来自这里，保证 UI 行为可无头验证。
struct UiState {
    bool listening = false;
    bool recording = false;
    bool wizard = false;

    bool busy() const { return listening || recording || wizard; }
    bool canSpeak() const { return !wizard; }           // 朗读：仅向导态禁用
    bool canStartListen() const { return !busy(); }
    bool canStopListen() const { return listening; }
    bool canEnroll() const { return !busy(); }          // WAV注册/清空注册
    bool canRecord() const { return !listening && !wizard; }  // 手动录音注册
    bool canStartWizard() const { return !busy(); }
    bool canPickWav() const { return !busy(); }

    void onListenChanged(bool on) { listening = on; }
    void onRecordChanged(bool on) { recording = on; }
    void onWizardChanged(bool on) { wizard = on; }
};

// 音频队列背压决策：处理跟不上时保实时性弃完整性（丢最旧帧）。
// PVAD 流式打分成本随流长增长（demo 原型形态），没有背压时采集队列会持续积压、
// tick() 全量消费不返回、引擎事件循环饿死（表现为"停止/朗读按钮都没反应"）。
struct Backpressure {
    static constexpr size_t kMaxPerTick = 5;        // 每 tick 最多处理帧数（10ms/帧）
    static constexpr size_t kDropAfterSamples = 16000;   // 积压 >1s 开始丢
    static constexpr size_t kKeepSamples = 16000 / 2;    // 丢时保留最新 0.5s

    // 需要从队列头部丢弃的采样数（0 = 不丢）
    static size_t drop_count(size_t qsize_samples) {
        return qsize_samples > kDropAfterSamples ? qsize_samples - kKeepSamples : 0;
    }
};
