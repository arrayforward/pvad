// demo_core.cpp
#include "demo_core.h"
#include "fbank.h"
#include "pvad.h"
#include "speaker.h"
#include "wav_io.h"

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

bool DemoCore::enroll(const std::vector<std::string>& wavs, std::string& err) {
    if (!spk_) { err = "core not initialized"; return false; }
    try {
        int added = 0;
        for (auto& w : wavs) {
            WavData wd = read_wav(w);
            auto emb = spk_->embed(wd.samples.data(), (int)wd.samples.size());
            if (emb_sum_.empty()) emb_sum_.assign(emb.size(), 0.f);
            for (size_t i = 0; i < emb.size(); i++) emb_sum_[i] += emb[i];
            added++;
        }
        if (added == 0) { err = "no wav"; return false; }
        n_emb_ += added;
        centroid_ = emb_sum_;
        l2_normalize(centroid_);
        has_tpl_ = true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool DemoCore::enroll_samples(const float* pcm, size_t n, std::string& err) {
    if (!spk_) { err = "core not initialized"; return false; }
    try {
        auto emb = spk_->embed(pcm, (int)n);
        if (emb_sum_.empty()) emb_sum_.assign(emb.size(), 0.f);
        for (size_t i = 0; i < emb.size(); i++) emb_sum_[i] += emb[i];
        n_emb_ += 1;
        centroid_ = emb_sum_;
        l2_normalize(centroid_);
        has_tpl_ = true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

void DemoCore::clear_enroll() {
    emb_sum_.clear();
    centroid_.clear();
    n_emb_ = 0;
    has_tpl_ = false;
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
