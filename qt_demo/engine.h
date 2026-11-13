// engine.h - Qt 线程模型：Engine 运行在独立 QThread，
// 拥有 miniaudio 采集/播放设备、DemoCore、Tts；GUI 通过信号槽通信。
#pragma once
#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
#include "agc.h"
#include "demo_core.h"
#include "denoise.h"
#include "direct_gate.h"
#include "enroll_store.h"
#include "fbank.h"
#include "tts.h"
#include "vad.h"
#include "wizard.h"
#include "tse_chain.h"
#include "asr.h"
#include <miniaudio.h>

class Engine : public QObject {
    Q_OBJECT
public:
    explicit Engine(QObject* parent = nullptr);
    ~Engine() override;

public slots:
    void init();                          // 在线程内初始化模型（路径用 qgetenv/默认）
    void enrollFiles(QStringList files);  // 注册 A
    void speakText(QString text);         // TTS 合成（worker 内，防卡 UI）并播放
    void stopPlayback();
    void startListenMic();
    void startListenWav(QString path);    // WAV 注入模拟麦克风
    void stopListen();
    void startRecord();                   // 录音注册：开始（与监听互斥，15s 上限自动停止）
    void stopRecord();                    // 录音注册：手动停止
    void clearEnroll();                   // 清空注册集合
    void startWizard();                   // 引导注册：备份旧注册并开始 3 段向导
    void cancelWizard();                  // 引导注册：取消并恢复旧注册
    void setDenoiseEnabled(bool on);      // 监听降噪（RNNoise），默认关
    void setPvadThreshold(double threshold);  // 实时打断复核阈值（direct_v4 identity）
    void setAsrEnabled(bool on);          // [仅保留注册用户声音]：VAD 自动分段转写

signals:
    void logLine(QString);
    void enrollStatus(QString);
    void probUpdate(float p, int consec);
    void interruptFired(double t);
    void playbackChanged(bool playing);
    void listenChanged(bool listening);
    void recordStateChanged(bool recording);
    void recordProgress(double seconds);
    void ttsStatus(QString);
    void wizardStateChanged(bool active);
    void wizardStepChanged(int step);              // 0..2 进行中
    void wizardSegmentAccepted(int stepIndex, double seconds);
    void wizardSegmentRejected(double seconds);    // 太短，重录本段
    void wizardFinished(int segments);
    void wizardCancelled();
    void asrText(QString text);                    // TSE+ASR 后的用户语音文本
    void keepRatio(double ratio);                  // 过滤模式：保留比例

private slots:
    void tick();

private:
    static void capture_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames);
    static void playback_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames);
    bool openCapture();
    bool openPlayback(int sample_rate);
    void closeDevices();
    void handleEvent(const FrameEvent& ev);
    bool ensureTseEmb();            // 给缺 tse_emb 的注册段从 wav 补算，并更新 tse_emb_ 均值；返回是否有补算
    bool ensureDirectRef();         // 二级复核用原始电平 deep-192（懒计算，不改持久化声纹）
    void finalizeCollect();         // 收集结束：direct_v4 复核 → TSE 提取 → ASR → asrText
    void finishRecord();
    void persistEnrollment();       // 注册状态变化后落盘 enrollment/
    QString enrollmentDir() const;  // <root>/qt_demo/enrollment

    DemoCore core_;
    Tts tts_;
    bool models_ok_ = false;
    std::unique_ptr<Denoise> denoise_;    // 勾选[启用降噪]时创建

    ma_device cap_{};
    ma_device pb_{};
    bool cap_open_ = false;
    bool pb_open_ = false;
    int pb_rate_ = 16000;

    std::mutex mu_;
    std::deque<float> capq_;              // 采集回调 -> tick

    std::vector<float> playbuf_;          // 待播放 16k PCM
    std::atomic<size_t> playpos_{0};
    std::atomic<bool> playing_{false};

    std::vector<float> inject_;           // WAV 注入音频
    size_t inject_pos_ = 0;
    bool inject_mode_ = false;

    bool listening_ = false;              // 监听中（与录音互斥）
    bool recording_ = false;              // 录音注册中（与监听互斥）
    WizardController wizard_;             // 引导注册状态机（active() 时为向导态）
    std::vector<float> recbuf_;           // 录音缓冲（16k float）
    static constexpr size_t kMaxRecordSamples = 16000 * 15;  // 15s 上限
    static constexpr size_t kMinRecordSamples = 16000 * 2;   // <2s 不入注册

    QTimer* timer_ = nullptr;
    bool interrupt_latched_ = false;      // 本次监听是否已触发过（日志只记首次，之后仍上报数值）
    // v7b 流式 PVAD 已移除（长流自适应失效 + 真人 B 90% 误触发，见 REGRESSION）；
    // 打断与 P(target) 显示统一由 direct_v4 周期复核提供（1s 滑窗）。
    std::unique_ptr<Vad> vad_;            // silero VAD（连续转写的收集/结束判定）
    long last_speech_frame_ = 0;          // 最近 VAD 语音帧（打断复核的静音保护）

    // ---- VAD 自动分段转写链（无需 INTERRUPT；打断仅停止 TTS） ----
    std::unique_ptr<TseSpeakerEncoder> tse_enc_;   // 多尺度 4288 编码器
    std::vector<float> tse_emb_;                   // 注册声纹（4288 维均值）
    std::unique_ptr<DirectGate> direct_gate_;      // v7b 触发后的 direct_v4 句级复核
    std::vector<float> direct_ref_;                // 原始电平注册 deep-192 均值
    // ---- 打断快速复核（TTS 播放期间 direct_v4 周期复核，确认注册用户才停）----
    // 依据（docs/REGRESSION.md barge-in 对照 + confirm-test）：
    // v7b 对真人 B 90% 误触发、且长流自适应后触发不及时（S2 实测 4.7s 才 fire）；
    // TTS 回声在真人注册下 identity≈0，v4 不会自我打断（旧 SAPI 注册资产的 0.42~0.61 是假象）。
    // v4-3s滑窗 @TH=0.35/UNC=0：A 召回 83%、纯 B 误停 30%、延迟 p50 1.83s。
    std::unique_ptr<DirectGate> confirm_gate_;     // direct_v4 TH=0.35/UNC=0
    std::deque<float> confirm_buf_;                // 最近 4s post-denoise 音频（复核滑窗来源）
    long confirm_frame_ = 0;                       // 本次监听已处理帧数（复核调度用）
    long confirm_next_check_ = 0;                  // 下次复核帧号（播放中每 1s 一次）
    void runConfirmCheck();                        // 执行一次复核，放行则停 TTS
    std::deque<float> hist_;                       // 监听中 post-denoise 音频环形缓冲（封顶 8s）
    size_t hist_base_frame_ = 0;                   // hist_.front() 对应的绝对帧号
    size_t mic_frame_ = 0;                         // 本次监听已入 hist_ 的绝对帧计数
    bool collecting_ = false;                      // 打断后收集中
    int silence_run_ = 0;                          // 连续非语音帧（收集 hangover）
    size_t collect_from_ = 0;                      // 收集起点（触发帧 - 0.5s 回滚）
    size_t interrupt_frame_ = 0;
    bool asr_enabled_ = false;                     // [仅保留注册用户声音] 勾选
    std::unique_ptr<Asr> asr_;
    // 背压/节流状态
    size_t drop_acc_ = 0;                 // 自上次日志以来丢弃的采样数
    qint64 last_drop_log_ms_ = 0;
    double last_prog_sec_ = -1.0;         // recordProgress 上次上报值（0.1s 节流）
    qint64 listen_start_ms_ = 0;
    qint64 stats_frames_ = 0;             // 本次监听已处理帧数
    qint64 stats_window_start_ms_ = 0;    // 处理率统计窗口起点
    qint64 stats_window_frames_ = 0;
    LevelMeter level_meter_;              // 1s 麦克风电平窗口（post-RNNoise）
    qint64 last_level_log_ms_ = 0;
};
