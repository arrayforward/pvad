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
#include "demo_core.h"
#include "tts.h"
#include "wizard.h"
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

private slots:
    void tick();

private:
    static void capture_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames);
    static void playback_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames);
    bool openCapture();
    bool openPlayback(int sample_rate);
    void closeDevices();
    void handleEvent(const FrameEvent& ev);
    void finishRecord();

    DemoCore core_;
    Tts tts_;
    bool models_ok_ = false;

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
};
