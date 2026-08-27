// engine.cpp
#include "engine.h"
#include "ui_state.h"
#include "wav_io.h"
#include <QDateTime>
#include <QDir>
#include <QFile>
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
    emit logLine("模型已加载 (campplus + pvad_v4)");
    // 启动自动加载持久化注册（损坏则降级空注册，不崩溃）
    {
        std::vector<SegRecord> segs;
        std::vector<float> fm;
        bool loaded = false;
        std::string le;
        QString dir = enrollmentDir();
        if (EnrollStore::load(dir.toStdString(), segs, fm, loaded, le)) {
            if (loaded) {
                core_.set_segments(segs);
                if (!fm.empty()) {
                    std::vector<double> s(80);
                    for (int b = 0; b < 80; b++) s[b] = (double)fm[b] * 1000.0;
                    core_.set_fbank_state(s, 1000);
                }
                emit enrollStatus(QString("已从磁盘加载（%1 段）").arg(core_.enroll_count()));
                emit logLine(QString("已从 %1 加载注册（%2 段）").arg(dir).arg(core_.enroll_count()));
            }
        } else {
            emit logLine("注册文件损坏，已降级为空注册: " + QString::fromStdString(le));
        }
    }
    // 实时流式 PVAD（chunked GRU state 复用）
    try {
        stream_ = std::make_unique<PvadStream>(
            (root + "/models/pvad/pvad_v4_stream.onnx").toStdString());
        refreshStreamEnroll();
        emit logLine("流式 PVAD 已加载 (pvad_v4_stream)");
    } catch (const std::exception& e) {
        emit logLine("流式 PVAD 加载失败（麦克风监听不可用）: " + QString::fromStdString(e.what()));
    }
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
        persistEnrollment();
        emit enrollStatus(QString("已注册 %1 段，质心就绪").arg(core_.enroll_count()));
        emit logLine(QString("注册成功: %1 个文件").arg(files.size()));
    } else {
        emit enrollStatus("注册失败");
        emit logLine("注册失败: " + QString::fromStdString(e));
    }
}

QString Engine::enrollmentDir() const {
    return qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT) + "/qt_demo/enrollment";
}

void Engine::persistEnrollment() {
    std::string e;
    if (EnrollStore::save(enrollmentDir().toStdString(), core_.segments(), core_.centroid(),
                          core_.fbank_mean(), e)) {
        emit logLine(QString("注册已落盘（%1 段 -> enrollment/）").arg(core_.enroll_count()));
        refreshStreamEnroll();
    } else {
        emit logLine("注册保存失败: " + QString::fromStdString(e));
    }
}

// 注册状态变化后同步流式 PVAD 的 emb 与 CMVN 先验
void Engine::refreshStreamEnroll() {
    if (!stream_) return;
    if (core_.enrolled()) {
        stream_->set_emb(core_.centroid().data());
        auto m = core_.fbank_mean();
        if (!m.empty()) stream_->set_cmvn_prior(m.data());
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
    if (!stream_) { emit logLine("流式 PVAD 未加载，无法监听"); return; }
    inject_mode_ = false;
    if (!openCapture()) return;
    core_.reset_stream();
    stream_->reset();
    sgate_.reset();
    swin_.clear();
    refreshStreamEnroll();
    interrupt_latched_ = false;
    if (ma_device_start(&cap_) != MA_SUCCESS) { emit logLine("采集启动失败"); return; }
    listening_ = true;
    emit listenChanged(true);
    emit logLine("开始监听（麦克风，流式 PVAD）");
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
        QString rel = "recordings/" + QFileInfo(fname).fileName();
        QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
        if (wizard_.accept_segment(core_, recbuf_.data(), recbuf_.size(),
                                   rel.toStdString(), dur, ts.toStdString())) {
            int done_idx = wizard_.step() - 1;
            emit wizardSegmentAccepted(done_idx, dur);
            emit logLine(QString("第 %1/3 段已录入（%2s）").arg(done_idx + 1).arg(dur, 0, 'f', 1));
            persistEnrollment();
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
    QString rel = "recordings/" + QFileInfo(fname).fileName();
    QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    if (core_.enroll_samples(recbuf_.data(), recbuf_.size(), e,
                             rel.toStdString(), dur, ts.toStdString())) {
        persistEnrollment();
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
    // 删除持久化文件（recordings/ 的 wav 保留）
    QString dir = enrollmentDir();
    QFile::remove(dir + "/tpl.bin");
    QFile::remove(dir + "/segments.json");
    emit enrollStatus("未注册");
    emit logLine("已清空注册（enrollment/ 文件已删除，recordings/ 录音保留）");
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
    persistEnrollment();  // 恢复后的状态落盘（向导期间的段被回滚）
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
        double sec = recbuf_.size() / 16000.0;
        if (sec - last_prog_sec_ >= 0.1 || sec < last_prog_sec_) {  // 0.1s 节流，避免洪泛 GUI
            last_prog_sec_ = sec;
            emit recordProgress(sec);
        }
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
    // 麦克风：背压——积压 >1s 丢弃最旧帧（保留最新 0.5s），每 tick 最多处理 5 帧。
    // PVAD 流式打分成本随流长增长（demo 原型形态），不限量消费会让 tick() 不返回、
    // 事件循环饿死（"停止/朗读没反应"的根因）。
    size_t dropped;
    {
        std::lock_guard<std::mutex> lk(mu_);
        dropped = Backpressure::drop_count(capq_.size());
        if (dropped) capq_.erase(capq_.begin(), capq_.begin() + (ptrdiff_t)dropped);
    }
    if (dropped) {
        drop_acc_ += dropped;
        qint64 now = QDateTime::currentMSecsSinceEpoch();
        if (now - last_drop_log_ms_ > 3000) {  // 3s 节流日志
            emit logLine(QString("处理跟不上采集，已自动降载（丢弃 %1 采样旧音频；"
                                 "判定作用在处理后的帧序列上，门控语义不变）")
                             .arg(drop_acc_));
            drop_acc_ = 0;
            last_drop_log_ms_ = now;
        }
    }
    for (size_t processed = 0; processed < Backpressure::kMaxPerTick; processed++) {
        float frame[160];
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (capq_.size() < 160) break;
            for (int i = 0; i < 160; i++) { frame[i] = capq_.front(); capq_.pop_front(); }
        }
        stats_frames_++;
        stats_window_frames_++;
        const float* src = frame;
        float dn[160];
        if (denoise_) { denoise_->process(frame, dn); src = dn; }
        // 流式 PVAD：单帧 fbank（对齐 480 窗）+ chunk GRU 增量推理，每帧 O(1)
        for (int i = 0; i < 160; i++) swin_.push_back(src[i]);
        if (swin_.size() < 480) continue;
        float w[400], f80[80];
        std::copy(swin_.begin(), swin_.begin() + 400, w);
        sfbank_.compute_one(w, f80);
        swin_.erase(swin_.begin(), swin_.begin() + 160);
        auto o = stream_->push_frame(f80);
        if (!o.valid || !o.gated) continue;
        bool fire = sgate_.update(o.p);
        FrameEvent ev;
        ev.p = o.p;
        ev.consec = sgate_.consec();
        ev.interrupt = fire;
        ev.t = o.frame * 0.01;
        handleEvent(ev);
    }
    // 处理率看门狗：监听中每 10s 报一次实际处理帧率（采集标称 100 帧/s）
    if (listening_ && stats_window_frames_ > 0) {
        qint64 now = QDateTime::currentMSecsSinceEpoch();
        if (stats_window_start_ms_ == 0) stats_window_start_ms_ = now;
        if (now - stats_window_start_ms_ >= 10000) {
            double rate = stats_window_frames_ * 1000.0 / (now - stats_window_start_ms_);
            if (rate < 95.0)
                emit logLine(QString("看门狗：处理率 %1 帧/s（标称 100），管线有积压风险")
                                 .arg(rate, 0, 'f', 1));
            stats_window_start_ms_ = now;
            stats_window_frames_ = 0;
        }
    } else if (!listening_) {
        stats_window_start_ms_ = 0;
        stats_window_frames_ = 0;
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
