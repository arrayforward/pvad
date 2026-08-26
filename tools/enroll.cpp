// enroll.cpp - 声纹注册：
//   enroll --out tpl.bin [--neg tts1.wav ...] [--cohort c1.wav ...] a1.wav a2.wav ...
// --neg 后的所有 wav 合并为一个负质心（标签 "neg"，如 TTS 音色）
// --cohort 后的所有 wav 合并为一个负质心（标签 "cohort"，通用无关人声）
#include "speaker.h"
#include "wav_io.h"
#include <cstdio>
#include <string>
#include <vector>

static std::vector<float> centroid(SpeakerEmbedder& spk, const std::vector<std::string>& wavs) {
    std::vector<float> mean_emb;
    for (auto& w : wavs) {
        WavData wd = read_wav(w);
        auto emb = spk.embed(wd.samples.data(), (int)wd.samples.size());
        printf("  %s -> embedding dim=%zu, %.2fs\n", w.c_str(), emb.size(), wd.samples.size() / 16000.0);
        if (mean_emb.empty()) mean_emb = emb;
        else for (size_t i = 0; i < emb.size(); i++) mean_emb[i] += emb[i];
    }
    l2_normalize(mean_emb);
    return mean_emb;
}

int main(int argc, char** argv) {
    std::string out = "tpl.bin";
    std::string model = "models/campplus.onnx";
    std::vector<std::string> pos, neg, cohort;
    std::vector<std::string>* cur = &pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) { out = argv[++i]; cur = &pos; }
        else if (a == "--model" && i + 1 < argc) { model = argv[++i]; cur = &pos; }
        else if (a == "--neg") cur = &neg;
        else if (a == "--cohort") cur = &cohort;
        else if (a == "--pos") cur = &pos;
        else cur->push_back(a);
    }
    if (pos.empty()) {
        fprintf(stderr, "usage: enroll --out tpl.bin [--neg tts1.wav ...] [--cohort c1.wav ...] [--pos] a1.wav a2.wav ...\n");
        fprintf(stderr, "  bare wavs go to the most recent group; use --pos to switch back to positives\n");
        return 1;
    }
    try {
        SpeakerEmbedder spk(model);
        Template tpl;
        printf("positive (speaker A):\n");
        tpl.pos = centroid(spk, pos);
        if (!neg.empty()) {
            printf("negative (e.g. TTS voice):\n");
            tpl.neg.push_back({"neg", centroid(spk, neg)});
        }
        // cohort：每个 wav 存一个独立的 L2 归一化 embedding（打分归一化用，不合并）
        for (auto& w : cohort) {
            WavData wd = read_wav(w);
            auto emb = spk.embed(wd.samples.data(), (int)wd.samples.size());
            printf("  cohort %s -> dim=%zu, %.2fs\n", w.c_str(), emb.size(), wd.samples.size() / 16000.0);
            tpl.cohort.push_back(std::move(emb));  // embed() 已 L2 归一化
        }
        save_template(out, tpl);
        printf("template saved: %s (v3, dim=%d, neg=%zu, cohort=%zu)\n",
               out.c_str(), tpl.dim(), tpl.neg.size(), tpl.cohort.size());
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
