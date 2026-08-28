// enroll_store.cpp
#include "enroll_store.h"
#include "speaker.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace {

std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += c; }
        else if ((unsigned char)c >= 0x20) r += c;
    }
    return r;
}

// 从 json 文本中提取 "key": "value"（字符串值）
std::string jstr(const std::string& s, const char* key, size_t from = 0) {
    std::string pat = std::string("\"") + key + "\": \"";
    size_t p = s.find(pat, from);
    if (p == std::string::npos) return "";
    p += pat.size();
    std::string r;
    for (size_t i = p; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) { r += s[i + 1]; i++; continue; }
        if (s[i] == '"') return r;
        r += s[i];
    }
    return "";
}

// 提取 "key": <number>
double jnum(const std::string& s, const char* key, bool& ok) {
    std::string pat = std::string("\"") + key + "\": ";
    size_t p = s.find(pat);
    if (p == std::string::npos) { ok = false; return 0; }
    ok = true;
    return strtod(s.c_str() + p + pat.size(), nullptr);
}

// 提取 "embedding": [ ... ] 的 float 数组（%.9g 写盘，strtof 解析可无损往返）
bool jfloats(const std::string& s, const char* key, std::vector<float>& out) {
    std::string pat = std::string("\"") + key + "\": [";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    out.clear();
    const char* cur = s.c_str() + p;
    while (*cur && *cur != ']') {
        if (*cur == ',' || *cur == ' ' || *cur == '\n' || *cur == '\t') { cur++; continue; }
        char* end = nullptr;
        float v = strtof(cur, &end);
        if (end == cur) return false;
        out.push_back(v);
        cur = end;
    }
    return *cur == ']';
}

// 从 cur 处解析一个 '[' ... ']' 的 float 数组；cur 推进到 ']' 之后
static const char* parse_farr(const char* cur, std::vector<float>& out) {
    out.clear();
    if (*cur != '[') return nullptr;
    cur++;
    while (*cur && *cur != ']') {
        if (*cur == ',' || *cur == ' ' || *cur == '\n' || *cur == '\t') { cur++; continue; }
        char* end = nullptr;
        float v = strtof(cur, &end);
        if (end == cur) return nullptr;
        out.push_back(v);
        cur = end;
    }
    if (*cur != ']') return nullptr;
    return cur + 1;
}

// 提取 "tokens": [[...],[...],...]（嵌套数组）
bool jtokens(const std::string& s, std::vector<std::vector<float>>& out) {
    std::string pat = "\"tokens\": [";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    const char* cur = s.c_str() + p + pat.size();
    out.clear();
    while (true) {
        while (*cur == ' ' || *cur == '\n' || *cur == '\t' || *cur == ',') cur++;
        if (*cur == ']') return true;  // tokens 数组结束
        std::vector<float> one;
        cur = parse_farr(cur, one);
        if (!cur) return false;
        out.push_back(std::move(one));
    }
}

}  // namespace

bool EnrollStore::save(const std::string& dir, const std::vector<SegRecord>& segs,
                       const std::vector<float>& centroid, const std::vector<float>& fbank_mean,
                       std::string& err) {
    try {
        std::filesystem::create_directories(dir);
        // tpl.bin（CLI v3 格式：仅正质心）
        Template tpl;
        tpl.pos = centroid;
        save_template(dir + "/tpl.bin", tpl);
        // segments.json
        std::string path = dir + "/segments.json";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { err = "cannot write " + path; return false; }
        fprintf(f, "{");
        if (!fbank_mean.empty()) {
            fprintf(f, "\"fbank_mean\": [");
            for (size_t j = 0; j < fbank_mean.size(); j++)
                fprintf(f, "%s%.9g", j ? ", " : "", (double)fbank_mean[j]);
            fprintf(f, "],\n");
        }
        fprintf(f, "\"segments\":[\n");
        for (size_t i = 0; i < segs.size(); i++) {
            const SegRecord& s = segs[i];
            fprintf(f, " {\"wav\": \"%s\", \"duration_s\": %.3f, \"time\": \"%s\", \"embedding\": [",
                    json_escape(s.wav).c_str(), s.duration_s, json_escape(s.time).c_str());
            for (size_t j = 0; j < s.emb.size(); j++)
                fprintf(f, "%s%.9g", j ? ", " : "", (double)s.emb[j]);
            fprintf(f, "]");
            if (!s.tokens.empty()) {
                fprintf(f, ", \"tokens\": [");
                for (size_t k = 0; k < s.tokens.size(); k++) {
                    fprintf(f, "%s[", k ? ", " : "");
                    for (size_t j = 0; j < s.tokens[k].size(); j++)
                        fprintf(f, "%s%.9g", j ? ", " : "", (double)s.tokens[k][j]);
                    fprintf(f, "]");
                }
                fprintf(f, "]");
            }
            fprintf(f, "}%s\n", i + 1 < segs.size() ? "," : "");
        }
        fprintf(f, "]}\n");
        fclose(f);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool EnrollStore::load(const std::string& dir, std::vector<SegRecord>& segs,
                       std::vector<float>& fbank_mean, bool& loaded, std::string& err) {
    loaded = false;
    segs.clear();
    fbank_mean.clear();
    std::string tpl_path = dir + "/tpl.bin";
    std::string seg_path = dir + "/segments.json";
    bool has_tpl = std::filesystem::exists(tpl_path);
    bool has_seg = std::filesystem::exists(seg_path);
    if (!has_tpl && !has_seg) return true;  // 空目录，正常初始状态

    try {
        if (has_seg) {
            FILE* f = fopen(seg_path.c_str(), "rb");
            if (!f) { err = "cannot read " + seg_path; return false; }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::string text(sz > 0 ? sz : 1, '\0');
            if (sz > 0 && fread(&text[0], 1, sz, f) != (size_t)sz) {
                fclose(f);
                err = "cannot read " + seg_path;
                return false;
            }
            fclose(f);
            // 可选根字段 fbank_mean
            jfloats(text, "fbank_mean", fbank_mean);
            // 逐段解析：找 '"wav":' 出现的位置作为段起点
            size_t pos = 0;
            while (true) {
                size_t p = text.find("\"wav\":", pos);
                if (p == std::string::npos) break;
                // 段范围到下一个 '"wav":' 或结尾
                size_t next = text.find("\"wav\":", p + 1);
                std::string rec = text.substr(p, next == std::string::npos ? next : next - p);
                SegRecord s;
                s.wav = jstr(rec, "wav");
                bool ok = false;
                s.duration_s = jnum(rec, "duration_s", ok);
                s.time = jstr(rec, "time");
                if (!jfloats(rec, "embedding", s.emb) || s.emb.empty()) {
                    err = "bad embedding in " + seg_path;
                    return false;
                }
                jtokens(rec, s.tokens);  // 可选字段（旧格式无 tokens -> 空，走重算升级路径）
                segs.push_back(std::move(s));
                pos = p + 1;
            }
            if (segs.empty()) { err = "no segments in " + seg_path; return false; }
            loaded = true;
            return true;
        }
        // 仅 tpl.bin（CLI enroll 导入）：正质心作单段
        Template tpl = load_template(tpl_path);
        SegRecord s;
        s.wav = "(imported tpl.bin)";
        s.time = "";
        s.emb = std::move(tpl.pos);
        segs.push_back(std::move(s));
        loaded = true;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        segs.clear();
        loaded = false;
        return false;
    }
}
