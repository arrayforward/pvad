// demo_core.cpp
#include "demo_core.h"
#include "fbank.h"
#include "pvad.h"
#include "speaker.h"
#include "wav_io.h"
#include <ctime>

DemoCore::DemoCore() = default;
DemoCore::~DemoCore() = default;

bool DemoCore::init(const std::string& spk_model, const std::string& pvad_model, std::string& err) {
    try {
        spk_ = std::make_unique<SpeakerEmbedder>(spk_model);
        pvad_ = std::make_unique<Pvad>(pvad_model);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

static std::string now_str() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

void DemoCore::append_segment(const std::vector<float>& emb, const std::string& wav,
                              double duration_s, const std::string& time) {
    if (emb_sum_.empty()) emb_sum_.assign(emb.size(), 0.f);
    for (size_t i = 0; i < emb.size(); i++) emb_sum_[i] += emb[i];
    segs_.push_back(SegRecord{wav, duration_s, time, emb});
    n_emb_ += 1;
    centroid_ = emb_sum_;
    l2_normalize(centroid_);
    has_tpl_ = true;
}

bool DemoCore::enroll(const std::vector<std::string>& wavs, std::string& err) {
    if (!spk_) { err = "core not initialized"; return false; }
    try {
        int added = 0;
        for (auto& w : wavs) {
            WavData wd = read_wav(w);
            auto emb = spk_->embed(wd.samples.data(), (int)wd.samples.size());
            append_segment(emb, w, wd.samples.size() / 16000.0, now_str());
            added++;
        }
        if (added == 0) { err = "no wav"; return false; }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool DemoCore::enroll_samples(const float* pcm, size_t n, std::string& err,
                              const std::string& wav, double duration_s,
                              const std::string& time) {
    if (!spk_) { err = "core not initialized"; return false; }
    try {
        auto emb = spk_->embed(pcm, (int)n);
        append_segment(emb, wav, duration_s > 0 ? duration_s : n / 16000.0,
                       time.empty() ? now_str() : time);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

void DemoCore::clear_enroll() {
    emb_sum_.clear();
    segs_.clear();
    centroid_.clear();
    n_emb_ = 0;
    has_tpl_ = false;
}

void DemoCore::set_enroll_state(const std::vector<float>& emb_sum, int n,
                                const std::vector<SegRecord>& segs) {
    emb_sum_ = emb_sum;
    n_emb_ = n;
    segs_ = segs;
    has_tpl_ = n > 0 && !emb_sum_.empty();
    if (has_tpl_) {
        centroid_ = emb_sum_;
        l2_normalize(centroid_);
    } else {
        centroid_.clear();
    }
}

void DemoCore::set_segments(const std::vector<SegRecord>& segs) {
    segs_ = segs;
    // 按原顺序重算累加和（浮点加法顺序一致 -> 与增量注册逐位一致）
    emb_sum_.clear();
    for (auto& s : segs_) {
        if (s.emb.empty()) continue;
        if (emb_sum_.empty()) emb_sum_.assign(s.emb.size(), 0.f);
        for (size_t i = 0; i < s.emb.size(); i++) emb_sum_[i] += s.emb[i];
    }
    n_emb_ = (int)segs_.size();
    has_tpl_ = n_emb_ > 0 && !emb_sum_.empty();
    if (has_tpl_) {
        centroid_ = emb_sum_;
        l2_normalize(centroid_);
    } else {
        centroid_.clear();
    }
}

void DemoCore::reset_stream() {
    seg_.clear();
    p2_pre_.clear();
    frame_idx_ = 0;
    gate_.reset();
}

bool DemoCore::precompute_file(const float* pcm, size_t n, std::string& err) {
    if (!has_tpl_) { err = "not enrolled"; return false; }
    Fbank fbank;
    std::vector<float> feats;
    int T = fbank.compute(pcm, (int)n, feats);
    if (T < 4) { err = "audio too short"; return false; }
    mean_normalize(feats, T, 80);
    try {
        p2_pre_ = pvad_->target_probs(feats.data(), T, centroid_.data());
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

FrameEvent DemoCore::feed_frame(const float* frame160) {
    for (int i = 0; i < 160; i++) {
        seg_.push_back(frame160[i]);
        if (seg_.size() > kSegCap) seg_.pop_front();
    }
    FrameEvent ev;
    ev.t = frame_idx_ * 0.01;
    if (frame_idx_ >= 4 && has_tpl_) {
        if (!p2_pre_.empty()) {
            // 文件模式：查预计算序列
            if (frame_idx_ < p2_pre_.size()) {
                ev.p = p2_pre_[frame_idx_];
                ev.interrupt = gate_.update(ev.p);
                ev.consec = gate_.consec();
            }
        } else {
            // 流式模式：前缀归一化整段 GRU（0.5s warm-up 后才更新门控）
            std::vector<float> w(seg_.begin(), seg_.end());
            Fbank fbank;
            std::vector<float> feats;
            int T = fbank.compute(w.data(), (int)w.size(), feats);
            if (T >= 4) {
                mean_normalize(feats, T, 80);
                try {
                    auto p2 = pvad_->target_probs(feats.data(), T, centroid_.data());
                    ev.p = p2.back();
                    if (frame_idx_ >= kWarmupFrames) {
                        ev.interrupt = gate_.update(ev.p);
                        ev.consec = gate_.consec();
                    }
                } catch (...) {
                    // 推理异常本帧跳过
                }
            }
        }
    }
    frame_idx_++;
    return ev;
}
