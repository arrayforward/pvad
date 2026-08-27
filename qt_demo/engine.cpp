// engine.cpp
#include "engine.h"
#include "wav_io.h"
#include <QDateTime>
#include <QDir>
#include <QFileInfo>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

// 采集回调：miniaudio 内部线程 -> 队列（tick 在 engine 线程消费）
void Engine::capture_cb(ma_device* dev, void*, const void* in, ma_uint32 frames) {
    Engine* e = static_cast<Engine*>(dev->pUserData);
    if (!in) return;
    std::lock_guard<std::mutex> lk(e->mu_);
    const float* p = static_cast<const float*>(in);
    e->capq_.insert(e->capq_.end(), p, p + frames);
    if (e->capq_.size() > 16000 * 10) e->capq_.erase(e->capq_.begin(), e->capq_.end() - 16000 * 10);
}

// 播放回调：从 playbuf 顺序读出，播完输出静音
void Engine::playback_cb(ma_device* dev, void* out, const void*, ma_uint32 frames) {
    Engine* e = static_cast<Engine*>(dev->pUserData);
    std::lock_guard<std::mutex> lk(e->mu_);
    float* o = static_cast<float*>(out);
    size_t pos = e->playpos_.load();
    size_t n = e->playbuf_.size();
    for (ma_uint32 i = 0; i < frames; i++) {
        o[i] = (pos + i < n) ? e->playbuf_[pos + i] : 0.f;
    }
    pos += frames;
    e->playpos_.store(pos);
    if (pos >= n) e->playing_.store(false);
}

Engine::Engine(QObject* parent) : QObject(parent) {}

Engine::~Engine() { closeDevices(); }

void Engine::closeDevices() {
    if (cap_open_) { ma_device_uninit(&cap_); cap_open_ = false; }
    if (pb_open_) { ma_device_uninit(&pb_); pb_open_ = false; }
}

void Engine::init() {
    QString root = qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT);
    QString err;
    std::string e;
    models_ok_ = core_.init((root + "/models/campplus.onnx").toStdString(),
                            (root + "/models/pvad/pvad_v4.onnx").toStdString(), e);
    if (!models_ok_) {
        emit logLine("模型加载失败: " + QString::fromStdString(e));
        return;
    }
    emit logLine("模型已加载 (campplus + pvad_v3)");
    QString tts_dir = qEnvironmentVariable("TTS_MODEL_DIR", TTS_MODEL_DIR);
    if (tts_.init(tts_dir.toStdString(), e)) emit ttsStatus("TTS 就绪 (" + tts_dir + ")");
    else emit ttsStatus("TTS 初始化失败: " + QString::fromStdString(e));
    timer_ = new QTimer(this);
    timer_->setInterval(5);
    connect(timer_, &QTimer::timeout, this, &Engine::tick);
    timer_->start();
}

void Engine::enrollFiles(QStringList files) {
    if (!models_ok_) { emit enrollStatus("模型未就绪"); return; }
    std::vector<std::string> wavs;
    for (auto& f : files) wavs.push_back(f.toStdString());
    std::string e;
    if (core_.enroll(wavs, e)) {
        emit enrollStatus(QString("已注册 %1 段，质心就绪").arg(core_.enroll_count()));
        emit logLine(QString("注册成功: %1 个文件").arg(files.size()));
    } else {
        emit enrollStatus("注册失败");
        emit logLine("注册失败: " + QString::fromStdString(e));
    }
}

bool Engine::openPlayback(int sample_rate) {
    if (pb_open_ && pb_rate_ == sample_rate) return true;
    if (pb_open_) { ma_device_uninit(&pb_); pb_open_ = false; }
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format = ma_format_f32;
    cfg.playback.channels = 1;
    cfg.sampleRate = sample_rate;
    cfg.dataCallback = playback_cb;
    cfg.pUserData = this;
    if (ma_device_init(nullptr, &cfg, &pb_) != MA_SUCCESS) {
        emit logLine("播放设备打开失败（无声卡?），仅记录状态");
        return false;
    }
    pb_open_ = true;
    pb_rate_ = sample_rate;
    return true;
}

void Engine::speakText(QString text) {
    if (wizard_.active()) { emit logLine("引导注册中，请先完成或取消向导"); return; }
    if (text.trimmed().isEmpty()) return;
    emit ttsStatus("合成中...");
    std::vector<float> pcm;
    int sr = 0;
    std::string e;
    if (!tts_.speak16k(text.toStdString(), pcm, sr, e)) {
        emit ttsStatus("合成失败: " + QString::fromStdString(e));
        return;
    }
    emit ttsStatus(QString("合成完成 %.1fs").arg(pcm.size() / 16000.0));
    {
        std::lock_guard<std::mutex> lk(mu_);
        playbuf_ = std::move(pcm);
    }
    playpos_.store(0);
    if (openPlayback(16000)) {
        playing_.store(true);
        emit playbackChanged(true);
        if (ma_device_start(&pb_) != MA_SUCCESS) {
            emit logLine("播放启动失败");
            playing_.store(false);
            emit playbackChanged(false);
        } else {
            emit logLine(QString("TTS 播放中 (%.1fs)").arg(playbuf_.size() / 16000.0));
        }
    } else {
        // 无播放设备：虚拟播放（状态机一致，便于无头环境）
        playing_.store(true);
        emit playbackChanged(true);
        emit logLine("（虚拟）TTS 播放中");
    }
}

void Engine::stopPlayback() {
    if (pb_open_) ma_device_stop(&pb_);
    if (playing_.exchange(false)) {
        emit playbackChanged(false);
        emit logLine(QString("TTS 播放已停止 @ %1s").arg(core_.now(), 0, 'f', 2));
    }
}

bool Engine::openCapture() {
    if (cap_open_) return true;
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = 16000;
    cfg.dataCallback = capture_cb;
    cfg.pUserData = this;
    if (ma_device_init(nullptr, &cfg, &cap_) != MA_SUCCESS) {
        emit logLine("麦克风打开失败");
        return false;
    }
    cap_open_ = true;
    return true;
}

void Engine::startListenMic() {
    if (recording_ || wizard_.active()) { emit logLine("录音/向导中，请先完成或停止"); return; }
    if (!core_.enrolled()) { emit logLine("请先注册 A"); return; }
    inject_mode_ = false;
    if (!openCapture()) return;
    core_.reset_stream();
    interrupt_latched_ = false;
    if (ma_device_start(&cap_) != MA_SUCCESS) { emit logLine("采集启动失败"); return; }
    listening_ = true;
    emit listenChanged(true);
    emit logLine("开始监听（麦克风）");
}

void Engine::startListenWav(QString path) {
    if (recording_ || wizard_.active()) { emit logLine("录音/向导中，请先完成或停止"); return; }
    if (!core_.enrolled()) { emit logLine("请先注册 A"); return; }
    try {
        WavData wd = read_wav(path.toStdString());
        inject_ = std::move(wd.samples);
    } catch (const std::exception& ex) {
        emit logLine("WAV 读取失败: " + QString::fromStdString(ex.what()));
        return;
    }
    inject_pos_ = 0;
    inject_mode_ = true;
    core_.reset_stream();
    std::string e;
    if (core_.precompute_file(inject_.data(), inject_.size(), e))
        emit logLine("已整段预计算 P(target) 序列（文件模式）");
    interrupt_latched_ = false;
    listening_ = true;
    emit listenChanged(true);
    emit logLine(QString("开始监听（WAV 注入 %1, %.1fs）")
                     .arg(QFileInfo(path).fileName())
                     .arg(inject_.size() / 16000.0));
}

void Engine::stopListen() {
    if (cap_open_ && !recording_) ma_device_stop(&cap_);
    inject_mode_ = false;
    listening_ = false;
    emit listenChanged(false);
    emit logLine("停止监听");
}

// ---------------- 录音注册 ----------------

void Engine::startRecord() {
    if (listening_) { emit logLine("监听中，请先停止监听"); return; }
    if (recording_) return;
    if (!models_ok_) { emit logLine("模型未就绪"); return; }
    if (!openCapture()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        capq_.clear();
        recbuf_.clear();
    }
    if (ma_device_start(&cap_) != MA_SUCCESS) { emit logLine("采集启动失败"); return; }
    recording_ = true;
    emit recordStateChanged(true);
    emit recordProgress(0.0);
    emit logLine("录音注册中...（再次点击停止，15s 自动停止）");
}

void Engine::stopRecord() {
    if (recording_) finishRecord();
}

void Engine::finishRecord() {
    recording_ = false;
    if (cap_open_) ma_device_stop(&cap_);
    emit recordStateChanged(false);
    double dur = recbuf_.size() / 16000.0;
    if (recbuf_.size() < kMinRecordSamples) {
        if (wizard_.active()) {
            emit wizardSegmentRejected(dur);
            emit logLine(QString("本段太短（%1s，建议3-10秒），请重录本段").arg(dur, 0, 'f', 1));
        } else {
            emit logLine(QString("录音太短（%.1fs，建议3-10秒），未加入注册").arg(dur));
        }
        return;
    }
    // 保存 wav 到 qt_demo/recordings/（文件名带时间戳）
    QString dir = qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT) + "/qt_demo/recordings";
    QDir().mkpath(dir);
    QString fname = dir + "/rec_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".wav";
    std::string e;
    try {
        write_wav16(fname.toStdString(), recbuf_, 16000);
    } catch (const std::exception& ex) {
        emit logLine("录音保存失败: " + QString::fromStdString(ex.what()));
    }
    // 向导态：段入向导状态机（取消向导时整体回滚，不会污染旧质心）
    if (wizard_.active()) {
        if (wizard_.accept_segment(core_, recbuf_.data(), recbuf_.size())) {
            int done_idx = wizard_.step() - 1;
            emit wizardSegmentAccepted(done_idx, dur);
            emit logLine(QString("第 %1/3 段已录入（%2s）").arg(done_idx + 1).arg(dur, 0, 'f', 1));
            if (wizard_.step() >= WizardController::kTotal) {
                emit wizardFinished(core_.enroll_count());
                emit wizardStateChanged(false);
                emit enrollStatus(QString("注册完成（%1 段）").arg(core_.enroll_count()));
                emit logLine(QString("引导注册完成，共 %1 段").arg(core_.enroll_count()));
            } else {
                emit wizardStepChanged(wizard_.step());
            }
        } else {
            emit wizardSegmentRejected(dur);
        }
        return;
    }
    if (core_.enroll_samples(recbuf_.data(), recbuf_.size(), e)) {
        emit enrollStatus(QString("已注册 %1 段").arg(core_.enroll_count()));
        emit logLine(QString("录音 %1s 已加入注册（共 %2 段），已存 %3")
                         .arg(dur, 0, 'f', 1)
                         .arg(core_.enroll_count())
                         .arg(fname));
    } else {
        emit logLine("注册失败: " + QString::fromStdString(e));
    }
}

void Engine::clearEnroll() {
    core_.clear_enroll();
    emit enrollStatus("未注册");
    emit logLine("已清空注册集合");
}

void Engine::setDenoiseEnabled(bool on) {
    if (on && !denoise_) {
        try {
            denoise_ = std::make_unique<Denoise>();
            emit logLine("降噪已启用（RNNoise，新增延迟约 11ms）");
        } catch (const std::exception& e) {
            emit logLine("降噪初始化失败: " + QString::fromStdString(e.what()));
        }
    } else if (!on && denoise_) {
        denoise_.reset();
        emit logLine("降噪已关闭");
    }
}

// ---------------- 引导注册 ----------------

void Engine::startWizard() {
    if (listening_ || recording_ || wizard_.active()) {
        emit logLine("请先停止当前监听/录音");
        return;
    }
    if (!models_ok_) { emit logLine("模型未就绪"); return; }
    wizard_.start(core_);
    emit wizardStateChanged(true);
    emit wizardStepChanged(0);
    emit enrollStatus("引导注册中（旧注册已备份，取消可恢复）");
    emit logLine("开始引导注册：3 段，按提示照念");
}

void Engine::cancelWizard() {
    if (!wizard_.active()) return;
    if (recording_) { emit logLine("请先停止录音"); return; }
    wizard_.cancel(core_);
    emit wizardStateChanged(false);
    emit wizardCancelled();
    emit enrollStatus(core_.enrolled() ? QString("已注册 %1 段").arg(core_.enroll_count())
                                       : QString("未注册"));
    emit logLine("已取消向导，恢复之前的注册");
}

void Engine::tick() {
    // 录音注册：采集帧全部进录音缓冲（不做 PVAD 打分），15s 自动停止
    if (recording_) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            recbuf_.insert(recbuf_.end(), capq_.begin(), capq_.end());
            capq_.clear();
        }
        emit recordProgress(recbuf_.size() / 16000.0);
        if (recbuf_.size() >= kMaxRecordSamples) finishRecord();
        return;
    }
    // 优先消费注入音频（每 tick 处理 20ms，略快于实时，demo 用）
    if (inject_mode_) {
        for (int k = 0; k < 2 && inject_pos_ + 160 <= inject_.size(); k++) {
            float frame[160];
            const float* src = inject_.data() + inject_pos_;
            if (denoise_) { denoise_->process(src, frame); src = frame; }
            FrameEvent ev = core_.feed_frame(src);
            inject_pos_ += 160;
            handleEvent(ev);
        }
        if (inject_pos_ + 160 > inject_.size()) {
            inject_mode_ = false;
            listening_ = false;
            emit listenChanged(false);
            emit logLine("注入音频播放完");
        }
        return;
    }
    // 麦克风：按 10ms 帧消费
    for (;;) {
        float frame[160];
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (capq_.size() < 160) break;
            for (int i = 0; i < 160; i++) { frame[i] = capq_.front(); capq_.pop_front(); }
        }
        if (denoise_) {
            float dn[160];
            denoise_->process(frame, dn);
            FrameEvent ev = core_.feed_frame(dn);
            handleEvent(ev);
        } else {
            FrameEvent ev = core_.feed_frame(frame);
            handleEvent(ev);
        }
    }
}

void Engine::handleEvent(const FrameEvent& ev) {
    emit probUpdate(ev.p, ev.consec);
    if (ev.interrupt) {
        emit interruptFired(ev.t);
        emit logLine(QString(">>> INTERRUPT @ %1s (P=%2)").arg(ev.t, 0, 'f', 2).arg(ev.p, 0, 'f', 3));
        if (playing_.load()) stopPlayback();
    }
}
