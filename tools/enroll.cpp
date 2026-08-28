// enroll.cpp - 声纹注册：
//   enroll --out tpl.bin [--neg tts1.wav ...] [--cohort c1.wav ...] a1.wav a2.wav ...
// --neg 后的所有 wav 合并为一个负质心（标签 "neg"，如 TTS 音色）
// --cohort 后的所有 wav 各存一个独立 embedding（打分归一化用）
// 批量模式（回归网格用）：enroll --batch-jsonl labels.jsonl --out-dir DIR [--max-n 500] [--root .]
//   取 overlap_frames 非空的前 N 条，对每条 enrollment wav 现场提 embedding 存
//   DIR/<id>.bin（v3 模板，仅正质心），与历史 eval_cpp_pvad.py "每条现场 enroll"
//   语义一致，但 CAM++ 只加载一次。
#include "speaker.h"
#include "wav_io.h"
#include <cstdio>
#include <cstdlib>
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

// 从单行 JSON 提取 "key": "value"（labels.jsonl 为单行 JSON，字符串扫描即可）
static std::string json_str(const std::string& line, const char* key) {
    std::string pat = std::string("\"") + key + "\": \"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    size_t e = line.find('"', p);
    if (e == std::string::npos) return "";
    return line.substr(p, e - p);
}

static int run_batch(const std::string& jsonl, const std::string& out_dir, int max_n,
                     const std::string& model, const std::string& root) {
    FILE* f = fopen(jsonl.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", jsonl.c_str()); return 1; }
    std::string cmd = "mkdir \"" + out_dir + "\" 2>NUL";
    system(cmd.c_str());  // 已存在则忽略错误

    SpeakerEmbedder spk(model);
    char* line = new char[1 << 21];
    int done = 0;
    while (fgets(line, 1 << 21, f)) {
        std::string s(line);
        if (s.find("\"overlap_frames\"") == std::string::npos) continue;
        if (s.find("\"overlap_frames\": []") != std::string::npos) continue;  // 无重叠
        std::string id = json_str(s, "id");
        std::string enroll_wav = json_str(s, "enrollment");
        if (id.empty() || enroll_wav.empty()) continue;
        try {
            WavData wd = read_wav(root + "/" + enroll_wav);
            auto emb = spk.embed(wd.samples.data(), (int)wd.samples.size());
            Template tpl;
            tpl.pos = std::move(emb);
            tpl.tokens = spk.embed_tokens(wd.samples.data(), (int)wd.samples.size());
            save_template(out_dir + "/" + id + ".bin", tpl);
            done++;
        } catch (const std::exception& e) {
            fprintf(stderr, "[%s] enroll failed: %s\n", id.c_str(), e.what());
        }
        if (done % 100 == 0) printf("[enroll-batch] %d\n", done);
        if (done >= max_n) break;
    }
    delete[] line;
    fclose(f);
    printf("[enroll-batch] done: %d templates -> %s\n", done, out_dir.c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string out = "tpl.bin";
    std::string model = "models/campplus.onnx";
    std::string batch_jsonl, out_dir, root = ".";
    int max_n = 500;
    std::vector<std::string> pos, neg, cohort;
    std::vector<std::string>* cur = &pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) { out = argv[++i]; cur = &pos; }
        else if (a == "--model" && i + 1 < argc) { model = argv[++i]; cur = &pos; }
        else if (a == "--batch-jsonl" && i + 1 < argc) batch_jsonl = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--max-n" && i + 1 < argc) max_n = std::stoi(argv[++i]);
        else if (a == "--neg") cur = &neg;
        else if (a == "--cohort") cur = &cohort;
        else if (a == "--pos") cur = &pos;
        else cur->push_back(a);
    }
    if (!batch_jsonl.empty()) {
        if (out_dir.empty()) {
            fprintf(stderr, "--batch-jsonl requires --out-dir\n");
            return 1;
        }
        try {
            return run_batch(batch_jsonl, out_dir, max_n, model, root);
        } catch (const std::exception& e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    if (pos.empty()) {
        fprintf(stderr, "usage: enroll --out tpl.bin [--neg tts1.wav ...] [--cohort c1.wav ...] [--pos] a1.wav a2.wav ...\n");
        fprintf(stderr, "  bare wavs go to the most recent group; use --pos to switch back to positives\n");
        fprintf(stderr, "  batch: enroll --batch-jsonl labels.jsonl --out-dir DIR [--max-n 500] [--root .]\n");
        return 1;
    }
    try {
        SpeakerEmbedder spk(model);
        Template tpl;
        printf("positive (speaker A):\n");
        tpl.pos = centroid(spk, pos);
        // v4：多帧 enrollment tokens（每个正样本 wav 按 1s 切分，尾段丢弃）
        for (auto& w : pos) {
            WavData wd = read_wav(w);
            auto toks = spk.embed_tokens(wd.samples.data(), (int)wd.samples.size());
            printf("  tokens %s -> N=%zu\n", w.c_str(), toks.size());
            for (auto& t : toks) tpl.tokens.push_back(std::move(t));
        }
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
        printf("template saved: %s (v4, dim=%d, neg=%zu, cohort=%zu, tokens=%zu)\n",
               out.c_str(), tpl.dim(), tpl.neg.size(), tpl.cohort.size(), tpl.tokens.size());
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
