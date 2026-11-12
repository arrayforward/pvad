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
    if (pos >= n && e->playing_.exchange(false)) {
        // 音频线程 -> GUI 队列信号，避免播完后内部已空闲但 UI 仍显示播放中。
        emit e->playbackChanged(false);
    }
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
                            (root + "/models/pvad/pvad_v5.onnx").toStdString(), e);
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
                // 旧格式（无 tokens）从 wav 重算升级；wav 被删则降级告警（v5 回退单 token）
                int tok_failed = core_.rebuild_missing_tokens(
                    qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT).toStdString() + "/qt_demo");
                if (tok_failed > 0)
                    emit logLine(QString("警告：%1 段注册缺少 tokens 且无法从 wav 重算"
                                         "（录音已删？），v5 将回退单 token 质心").arg(tok_failed));
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
            (root + "/models/pvad/pvad_v7b_stream.onnx").toStdString());
        vad_ = std::make_unique<Vad>((root + "/models/silero_vad.onnx").toStdString());
        refreshStreamEnroll();
        emit logLine("流式 PVAD 已加载 (pvad_v7b_stream + silero VAD)");
    } catch (const std::exception& e) {
        emit logLine("流式 PVAD 加载失败（麦克风监听不可用）: " + QString::fromStdString(e.what()));
    }
    // 打断后 TSE+ASR 链（听故事打断演示）：加载失败仅降级，不影响监听主功能
    try {
        tse_enc_ = std::make_unique<TseSpeakerEncoder>(
            (root + "/models/tse_e16/speaker_encoder.onnx").toStdString());
        emit logLine("TSE 声纹编码器已加载 (tse_e16/speaker_encoder)");
        if (core_.enrolled() && ensureTseEmb()) persistEnrollment();
    } catch (const std::exception& e) {
        tse_enc_.reset();
        emit logLine("TSE 声纹编码器加载失败（连续转写不可用）: " + QString::fromStdString(e.what()));
    }
    try {
        direct_gate_ = std::make_unique<DirectGate>(
            (root + "/models/tse_e16/direct_v3.onnx").toStdString(), 0.0f, 0.3f);
        emit logLine("句级二级复核已加载（direct_v3 TH=0.0，不确定区放行）");
    } catch (const std::exception& e) {
        direct_gate_.reset();
        emit logLine("二级复核加载失败（仅跳过复核，不影响连续转写）: " + QString::fromStdString(e.what()));
    }
    {
        asr_ = std::make_unique<Asr>();
        QString asr_dir = qEnvironmentVariable("ASR_MODEL_DIR",
                                               root + "/third_party/models/paraformer-zh-int8");
        if (asr_->init(asr_dir.toStdString(), e))
            emit logLine("ASR 已加载 (" + asr_dir + ")");
        else {
            asr_.reset();
            emit logLine("ASR 加载失败（连续转写不可用）: " + QString::fromStdString(e));
        }
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
    direct_ref_.clear();  // 注册集合变化后，二级复核引用下次懒重算
    if (tse_enc_ && core_.enrolled()) ensureTseEmb();  // 新段顺带补算 TSE 声纹
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

// 给缺 tse_emb 的注册段从来源 wav 补算（与 rebuild_missing_tokens 同一路径解析），
// 并刷新 tse_emb_（各段均值，TSE 提取的目标声纹）。返回是否有段被补算。
bool Engine::ensureTseEmb() {
    if (!tse_enc_) return false;
    std::string qtd = (qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT) + "/qt_demo").toStdString();
    auto& segs = core_.segments_mut();
    bool changed = false;
    int failed = 0;
    for (auto& s : segs) {
        if (!s.tse_emb.empty() && s.tse_emb_agc) continue;
        if (s.wav.empty() || s.wav == "(imported tpl.bin)") continue;
        std::string path = s.wav;
        FILE* probe = fopen(path.c_str(), "rb");
        if (!probe) {
            std::string alt = qtd + "/" + s.wav;
            probe = fopen(alt.c_str(), "rb");
            if (probe) path = alt;
        }
        if (!probe) { failed++; continue; }
        fclose(probe);
        try {
            WavData wd = read_wav(path);
            s.tse_emb = tse_enc_->encode({wd.samples});
            s.tse_emb_agc = true;
            changed = true;
        } catch (...) {
            failed++;
        }
    }
    if (failed > 0)
        emit logLine(QString("警告：%1 段注册无法补算 TSE 声纹（录音已删？），连续转写将降级").arg(failed));
    // 均值（所有有 tse_emb 的段）
    tse_emb_.assign(TseSpeakerEncoder::EMB_DIM, 0.f);
    int n = 0;
    for (const auto& s : segs) {
        if (s.tse_emb.empty()) continue;
        for (size_t i = 0; i < tse_emb_.size(); i++) tse_emb_[i] += s.tse_emb[i];
        n++;
    }
    if (n > 0) {
        for (auto& v : tse_emb_) v /= (float)n;
    } else {
        tse_emb_.clear();
    }
    return changed;
}

bool Engine::ensureDirectRef() {
    if (direct_ref_.size() == 192) return true;
    if (!tse_enc_) return false;
    std::string qtd = (qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT) + "/qt_demo").toStdString();
    direct_ref_.assign(192, 0.f);
    int count = 0;
    for (const auto& s : core_.segments()) {
        if (s.wav.empty() || s.wav == "(imported tpl.bin)") continue;
        std::string path = s.wav;
        FILE* probe = fopen(path.c_str(), "rb");
        if (!probe) {
            std::string alt = qtd + "/" + s.wav;
            probe = fopen(alt.c_str(), "rb");
            if (probe) path = alt;
        }
        if (!probe) continue;
        fclose(probe);
        try {
            WavData wd = read_wav(path);
            auto e = tse_enc_->encode({wd.samples}, false);  // direct_v3 对照：原始电平
            for (int i = 0; i < 192; i++) direct_ref_[i] += e[i];
            count++;
        } catch (...) {
        }
    }
    if (count > 0) {
        for (float& v : direct_ref_) v /= (float)count;
        return true;
    }
    // 录音源缺失时退回复用 TSE 注册前 192 维，功能可用但域一致性稍弱。
    if (tse_emb_.size() == TseSpeakerEncoder::EMB_DIM) {
        direct_ref_.assign(tse_emb_.begin(), tse_emb_.begin() + 192);
        emit logLine("二级复核：原注册 wav 不可用，复用持久化 deep-192");
        return true;
    }
    direct_ref_.clear();
    return false;
}

void Engine::setAsrEnabled(bool on) {
    asr_enabled_ = on;
    if (on && (!tse_enc_ || !asr_ || tse_emb_.empty()))
        emit logLine("注意：TSE/ASR 未就绪或未注册，连续转写不会生效");
    emit logLine(on ? "连续转写已开启（TSE 提取 → ASR，仅注册用户声音）" : "连续转写已关闭");
}

// 收集结束：hist_ 切片（触发帧-0.5s 起）→ TSE 提取 → ASR → asrText
void Engine::finalizeCollect() {
    collecting_ = false;
    // finalize 期间采集仍在积压，清掉避免结束后处理陈旧帧
    {
        std::lock_guard<std::mutex> lk(mu_);
        capq_.clear();
    }
    if (!tse_enc_ || !asr_ || tse_emb_.empty()) return;
    size_t start_frame = collect_from_ > hist_base_frame_ ? collect_from_ : hist_base_frame_;
    size_t off = (start_frame - hist_base_frame_) * 160;
    if (off >= hist_.size()) return;
    std::vector<float> seg(hist_.begin() + (ptrdiff_t)off, hist_.end());
    double sec = seg.size() / 16000.0;
    if (seg.size() < 16000 / 2) {
        emit logLine(QString("转写片段太短（%1s），跳过转写").arg(sec, 0, 'f', 2));
        return;
    }
    // v7b 已负责快速打断；direct_v3 只复核是否把本段送入 TSE/ASR，绝不撤销用户打断。
    // TH=0.0 是高召回工作点，置信度不足时默认放行，优先避免误杀用户。
    if (direct_gate_ && ensureDirectRef()) {
        try {
            auto windows = tse_enc_->encode_windows(seg.data(), (int)seg.size());
            auto review = direct_gate_->decide(windows, direct_ref_);
            emit logLine(QString("二级复核：identity=%1 confidence=%2，%3")
                         .arg(review.identity, 0, 'f', 3).arg(review.confidence, 0, 'f', 3)
                         .arg(review.closed ? "关闭本段转写" : "放行"));
            if (review.closed) {
                emit keepRatio(0.0);
                // 不用空文本覆盖上一句有效识别。
                emit logLine("二级复核关闭：本段不进入 TSE/ASR（已发生的用户打断不撤销）");
                return;
            }
        } catch (const std::exception& e) {
            emit logLine("二级复核异常，按放行降级: " + QString::fromStdString(e.what()));
        }
    }
    emit logLine(QString("连续转写：截取 %1s，TSE 提取中（单麦模拟 4ch）...").arg(sec, 0, 'f', 2));
    QString root = qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT);
    std::vector<float> ext;
    try {
        TseEngine eng((root + "/models/tse_e16/erb_df_step.onnx").toStdString(), tse_emb_);
        AgcReport agc;
        ext = tse_extract(eng, seg.data(), (int)seg.size(), &agc);
        emit logLine(QString("TSE AGC：RMS %1→%2（%3→%4 dBFS），增益 ×%5，峰值 %6")
                     .arg(agc.before.rms, 0, 'f', 5).arg(agc.after.rms, 0, 'f', 5)
                     .arg(agc.before.rms_dbfs(), 0, 'f', 1).arg(agc.after.rms_dbfs(), 0, 'f', 1)
                     .arg(agc.gain, 0, 'f', 2).arg(agc.after.peak, 0, 'f', 3));
    } catch (const std::exception& e) {
        emit logLine("TSE 提取失败: " + QString::fromStdString(e.what()));
        return;
    }
    emit keepRatio(seg.empty() ? 0.0 : (double)ext.size() / (double)seg.size());
    if (ext.empty()) { emit logLine("TSE 提取为空"); return; }
    // 不做 PVAD 帧掩码：ASR 直接识别 TSE 输出，避免低分目标帧被整段清零。
    std::string text = asr_->decode(ext.data(), (int)ext.size());
    emit logLine(QString("ASR 完成：%1").arg(QString::fromStdString(text)));
    if (!text.empty()) {
        emit asrText(QString::fromStdString(text));
    } else {
        // 连续监听中的空结果（如纯 TTS/噪声段）不清空上一句有效识别。
        emit logLine("本段 ASR 为空，保留上一句有效结果");
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
    // 新一次播放允许再次打断；否则 PvGate 上次 triggered 必须等 P<0.3 才复位，
    // 用户重播后会看到门控仍在动却不再触发。
    sgate_.reset();
    interrupt_latched_ = false;
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
    vad_->reset();
    sgate_.reset();
    swin_.clear();
    svad_speech_ = false;
    warmup_left_ = 20;
    refreshStreamEnroll();
    interrupt_latched_ = false;
    hist_.clear();
    hist_base_frame_ = 0;
    mic_frame_ = 0;
    collecting_ = false;
    silence_run_ = 0;
    level_meter_.reset();
    last_level_log_ms_ = 0;
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
    tse_emb_.clear();
    direct_ref_.clear();
    collecting_ = false;
    // 删除持久化文件（recordings/ 的 wav 保留）
    QString dir = enrollmentDir();
    QFile::remove(dir + "/tpl.bin");
    QFile::remove(dir + "/segments.json");
    emit enrollStatus("未注册");
    emit logLine("已清空注册（enrollment/ 文件已删除，recordings/ 录音保留）");
}

void Engine::setPvadThreshold(double threshold) {
    float thr = (float)threshold;
    core_.set_gate_threshold(thr);
    sgate_ = PvadGate{thr, 0.2f, 2};
    emit logLine(QString("PVAD 门控阈值已实时更新为 %1（连续 2 帧，计数已复位）")
                 .arg(thr, 0, 'f', 2));
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
        // 电平量化只观测，不改变 PVAD 输入；TSE 在缓冲切片上另做恒定增益 AGC。
        level_meter_.add(src, 160);
        qint64 level_now = QDateTime::currentMSecsSinceEpoch();
        if (listening_ && level_now - last_level_log_ms_ >= 1000) {
            AudioLevel level = level_meter_.measure();
            emit logLine(QString("电平（1s）：RMS %1（%2 dBFS），有效语音 %3，峰值 %4；TSE AGC 目标 0.045")
                         .arg(level.rms, 0, 'f', 5).arg(level.rms_dbfs(), 0, 'f', 1)
                         .arg(level.active_rms, 0, 'f', 5).arg(level.peak, 0, 'f', 3));
            level_meter_.reset();
            last_level_log_ms_ = level_now;
        }
        // 连续转写用回放缓冲：post-denoise 采样入 hist_（封顶 8s，8s=800 帧）
        if (asr_enabled_) {
            for (int i = 0; i < 160; i++) hist_.push_back(src[i]);
            mic_frame_++;
            while (hist_.size() > 16000 * 8) {
                for (int i = 0; i < 160; i++) hist_.pop_front();
                hist_base_frame_++;
            }
        }
        // 长流会话策略（与 CLI StreamRunner 一致）：
        // 1) silero VAD 判语音的帧才入门控；2) VAD speech-end 会话分段复位
        // （只清 GRU state，保留 EMA/先验）；3) 复位后 warm-up 20 VAD 帧，confirm=4
        float vp = vad_->process(src, 160);
        bool sp = vp >= 0.f && vp > 0.5f;
        if (vp >= 0.f && sp != svad_speech_) {
            if (svad_speech_ && !sp) {
                stream_->reset_gru();
                sgate_.reset();
                warmup_left_ = 20;
            }
            svad_speech_ = sp;
        }
        // 连续转写：VAD 检测到语音即开始收集，不等待 INTERRUPT。
        // PVAD 打断只负责停止 TTS，与本转写状态机相互独立。
        if (asr_enabled_ && !collecting_ && sp && tse_enc_ && asr_ && !tse_emb_.empty()) {
            collecting_ = true;
            silence_run_ = 0;
            collect_from_ = mic_frame_ >= 50 ? mic_frame_ - 50 : hist_base_frame_;
            if (collect_from_ < hist_base_frame_) collect_from_ = hist_base_frame_;
            emit logLine("连续转写：检测到语音，开始收集（无需打断，VAD 静音或 6s 截止）");
        }
        // 收集结束判定：连续 0.4s 非语音（hangover 跨句间停顿）或满 6s
        // （须在 !sp continue 之前检查）
        if (collecting_) {
            if (!sp) silence_run_++; else silence_run_ = 0;
            bool end = silence_run_ >= 40 || (mic_frame_ - collect_from_ >= 600);
            if (end) {
                if (mic_frame_ - collect_from_ >= 30) finalizeCollect();
                else { collecting_ = false; emit logLine("转写片段太短，取消转写"); }
            }
        }
        // 流式 PVAD：单帧 fbank（对齐 480 窗）+ chunk GRU 增量推理，每帧 O(1)
        for (int i = 0; i < 160; i++) swin_.push_back(src[i]);
        if (swin_.size() < 480) continue;
        float w[400], f80[80];
        std::copy(swin_.begin(), swin_.begin() + 400, w);
        sfbank_.compute_one(w, f80);
        swin_.erase(swin_.begin(), swin_.begin() + 160);
        auto o = stream_->push_frame(f80);
        // !o.gated = 会话起始 0.5s warm-up（PvadStream 自带）：压会话冷启动 blip
        // （EMA/GRU 暂态会让首个语音段前 ~0.4s 的 P 虚高过阈，story-test 复现）
        if (!o.valid || !o.gated || !sp) continue;
        if (warmup_left_ > 0) { warmup_left_--; continue; }
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
        // 兜底：正常已由 VAD 自动收集；若尚未开始，则在触发帧-0.5s 补启动（注入模式跳过）
        if (asr_enabled_ && !collecting_ && !inject_mode_ &&
            tse_enc_ && asr_ && !tse_emb_.empty()) {
            collecting_ = true;
            silence_run_ = 0;
            interrupt_frame_ = mic_frame_;
            collect_from_ = mic_frame_ >= 50 ? mic_frame_ - 50 : hist_base_frame_;
            if (collect_from_ < hist_base_frame_) collect_from_ = hist_base_frame_;
            emit logLine("连续转写：开始收集语音（VAD 静音或 6s 截止）");
        }
    }
}
