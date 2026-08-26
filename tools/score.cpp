// score.cpp - 整段打分：score [选项] tpl.bin test.wav
// 输出 sA_raw、sA_norm（t-norm）、sNeg 及判决结果
#include "speaker.h"
#include "wav_io.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model = "models/campplus.onnx";
    float threshold = 0.55f, margin = 0.15f, z_threshold = 3.0f;
    int topk = 50;
    bool use_norm = true;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (a == "--margin" && i + 1 < argc) margin = std::stof(argv[++i]);
        else if (a == "--z-threshold" && i + 1 < argc) z_threshold = std::stof(argv[++i]);
        else if (a == "--norm-topk" && i + 1 < argc) topk = std::stoi(argv[++i]);
        else if (a == "--no-norm") use_norm = false;
        else args.push_back(a);
    }
    if (args.size() != 2) {
        fprintf(stderr, "usage: score [--threshold 0.55] [--margin 0.15] [--z-threshold 3.0] [--norm-topk 50] [--no-norm] tpl.bin test.wav\n");
        return 1;
    }
    try {
        auto tpl = load_template(args[0]);
        WavData wd = read_wav(args[1]);
        SpeakerEmbedder spk(model);
        auto emb = spk.embed(wd.samples.data(), (int)wd.samples.size());
        float sA = cosine_sim(tpl.pos, emb);
        float sNeg = max_neg_sim(tpl, emb);
        float mu, sigma;
        tnorm_stats(tpl, emb, topk, mu, sigma);
        float z = (sA - mu) / (sigma > 1e-4f ? sigma : 1e-4f);
        bool margin_ok = (sA - sNeg) > margin;
        bool pass = use_norm ? (z > z_threshold && margin_ok) : (sA > threshold && margin_ok);
        printf("sA_raw  = %.4f\n", sA);
        printf("sA_norm = %.3f  (cohort=%zu, topk=%d, mu=%.4f, sigma=%.4f, z_threshold=%.2f)\n",
               z, tpl.cohort.size(), topk, mu, sigma, z_threshold);
        printf("sNeg    = %.4f  (sA-sNeg=%.4f, margin=%.2f)\n", sNeg, sA - sNeg, margin);
        printf("decision: %s  (%s mode)\n", pass ? "PASS (would interrupt)" : "REJECT",
               use_norm ? "norm" : "no-norm");
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
