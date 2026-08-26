// main.cpp - double_voice: 说话人门控打断 (barge-in) 原型
// 离线: double_voice --wav in.wav --template tpl.bin [--aec --far ref.wav] [选项]（默认无 AEC）
// 实时: double_voice --mic --template tpl.bin [--play-tone] [--aec] [--seconds 30] [选项]
#include "aec.h"
#include "denoise.h"
#include "fbank.h"
#include "gate.h"
#include "pvad.h"
#include "speaker.h"
#include "vad.h"
#include "wav_io.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

namespace {

struct Args {
    bool mic = false;
    bool play_tone = false;
    bool aec = false;            // 默认不做 AEC，显式 --aec 才启用（离线还需 --far）
    std::string wav, far_wav, tpl = "tpl.bin";
    std::string batch_list, batch_out;   // 批量模式：--batch-list 文件列表 -> --batch-out 结果
    std::string vad_model = "models/silero_vad.onnx";
    std::string spk_model = "models/campplus.onnx";
    std::string pvad_model = "models/pvad/pvad.onnx";
    std::string gate_mode = "pvad";   // pvad (默认) | asnorm
    std::string denoise = "rnnoise";  // rnnoise (默认) | off
    bool bench_denoise = false;
    float pvad_threshold = 0.5f;
    float pvad_hyst = 0.2f;
    float threshold = 0.55f;
    float margin = 0.15f;
    float z_threshold = 3.0f;
    int norm_topk = 50;
    bool use_norm = true;        // --no-norm 退回纯余弦
    float vad_threshold = 0.5f;
    int confirm = 2;
    int window_ms = 500;
    int seconds = 30;
};

void usage() {
    printf(R"(usage:
  double_voice --wav in.wav --template tpl.bin [--aec --far tts_ref.wav] [options]
  double_voice --mic --template tpl.bin [--play-tone] [--aec] [--seconds 30] [options]
options:
  --gate pvad         门控模式: pvad (默认, Personal VAD) | asnorm (AS-norm+margin)
  --pvad-threshold 0.5  pvad 模式: P(target) 触发阈值
  --pvad-hyst 0.2     pvad 模式: 低于 threshold-hyst 计数清零
  --pvad-model PATH   pvad.onnx 路径
  --denoise rnnoise   降噪: rnnoise (默认, RNNoise) | off (回滚/对比用)
  --bench-denoise     测降噪单帧耗时后退出
  --threshold 0.55    asnorm no-norm 模式: sA_raw 触发阈值
  --z-threshold 3.0   norm 模式: sA_norm=(sA_raw-mu)/sigma 触发阈值
  --norm-topk 50      t-norm 取 cohort 相似度 top-K 估计 mu/sigma
  --no-norm           退回纯余弦模式（对比实验用）
  --margin 0.15       sA_raw - sNeg 最小间隔（负模板压制回声/无关人声）
  --vad-threshold 0.5 VAD 语音概率阈值
  --confirm 2         连续 N 次满足条件才触发
  --window-ms 500     声纹分析窗长度（毫秒）
  --aec               启用 SpeexDSP AEC（默认关闭；离线模式需同时给 --far 参考）
  --vad-model PATH    silero_vad.onnx 路径
  --spk-model PATH    campplus.onnx 路径
)");
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](const char* def) -> std::string { return i + 1 < argc ? argv[++i] : def; };
        if (s == "--mic") a.mic = true;
        else if (s == "--play-tone") a.play_tone = true;
        else if (s == "--aec") a.aec = true;
        else if (s == "--wav") a.wav = next("");
        else if (s == "--batch-list") a.batch_list = next("");
        else if (s == "--batch-out") a.batch_out = next("batch_out.tsv");
        else if (s == "--far") a.far_wav = next("");
        else if (s == "--template") a.tpl = next("tpl.bin");
        else if (s == "--gate") a.gate_mode = next("pvad");
        else if (s == "--pvad-threshold") a.pvad_threshold = std::stof(next("0.5"));
        else if (s == "--pvad-hyst") a.pvad_hyst = std::stof(next("0.2"));
        else if (s == "--pvad-model") a.pvad_model = next("");
        else if (s == "--denoise") a.denoise = next("off");
        else if (s == "--bench-denoise") a.bench_denoise = true;
        else if (s == "--threshold") a.threshold = std::stof(next("0.55"));
        else if (s == "--margin") a.margin = std::stof(next("0.15"));
        else if (s == "--z-threshold") a.z_threshold = std::stof(next("3.0"));
        else if (s == "--norm-topk") a.norm_topk = std::stoi(next("50"));
        else if (s == "--no-norm") a.use_norm = false;
        else if (s == "--vad-threshold") a.vad_threshold = std::stof(next("0.5"));
        else if (s == "--confirm") a.confirm = std::stoi(next("2"));
        else if (s == "--window-ms") a.window_ms = std::stoi(next("500"));
        else if (s == "--seconds") a.seconds = std::stoi(next("30"));
        else if (s == "--vad-model") a.vad_model = next("");
        else if (s == "--spk-model") a.spk_model = next("");
        else { usage(); exit(1); }
    }
    return a;
}

// 共享管线：(可选 AEC) -> VAD -> 滑窗 -> PVAD 门控 或 sA/sNeg margin 门控
struct Pipeline {
    Vad vad;
    SpeakerEmbedder spk;
    Template tpl;
    Gate gate;
    PvadGate pgate;
    std::unique_ptr<Pvad> pvad;
    std::unique_ptr<Denoise> denoise;
    bool skip_frame_denoise = false;  // 离线预降噪后跳过逐帧降噪
    bool pvad_mode;
    std::unique_ptr<Aec> aec;
    std::deque<float> win;      // 滑窗采样环形缓冲（asnorm 用）
    std::deque<float> seg;      // 当前语音段采样（pvad 用，GRU 整段输入，与训练一致；封顶 8s）
    static constexpr size_t kSegCap = 16000 * 8;
    size_t win_len;
    float vad_threshold;
    int norm_topk;
    bool vad_speech = false;
    std::vector<float> p2_pre;   // 离线模式：整段预计算的 P(target) 序列（与 python 评估一致）
    size_t frame_idx = 0;

    Pipeline(const Args& a)
        : vad(a.vad_model), spk(a.spk_model), tpl(load_template(a.tpl)),
          gate(GateOptions{a.threshold, a.margin, a.z_threshold, 0.1f, 0.5f, a.confirm, a.use_norm}),
          pgate(PvadGate{a.pvad_threshold, a.pvad_hyst, a.confirm}),
          pvad_mode(a.gate_mode == "pvad"),
          win_len((size_t)a.window_ms * 16), vad_threshold(a.vad_threshold), norm_topk(a.norm_topk) {
        if (pvad_mode) pvad = std::make_unique<Pvad>(a.pvad_model);
        if (a.denoise == "rnnoise") denoise = std::make_unique<Denoise>();
        if (a.aec) aec = std::make_unique<Aec>(160, 2048, 16000);
        gate.reset();
        pgate.reset();
    }

    // 处理一个 10ms 帧；far_frame 可为 nullptr（不做 AEC）。t_sec 用于打印时间戳。
    void process_frame(const float* frame, const float* far_frame, double t_sec) {
        // 采集 -> (可选降噪) -> (可选 AEC) -> VAD/PVAD 门控
        float dn[160];
        const float* src = frame;
        if (denoise && !skip_frame_denoise) {
            denoise->process(frame, dn);
            src = dn;
        }
        float clean[160];
        if (aec && far_frame) aec->process(src, far_frame, clean);
        else memcpy(clean, src, sizeof(clean));

        float vad_prob = vad.process(clean, 160);
        for (int i = 0; i < 160; i++) {
            win.push_back(clean[i]);
            if (win.size() > win_len) win.pop_front();
            seg.push_back(clean[i]);
            if (seg.size() > kSegCap) seg.pop_front();
        }
        if (vad_prob >= 0.f) {
            bool speech = vad_prob > vad_threshold;
            if (speech != vad_speech) {
                printf("[t=%6.2f] VAD %s (p=%.3f)\n", t_sec, speech ? "speech start" : "speech end  ", vad_prob);
                vad_speech = speech;
                if (!speech) { gate.reset(); pgate.reset(); }
            }
        }
        // 打分: pvad 模式每 10ms 帧都打（PVAD 自带静音类，无需 silero 门控）;
        // asnorm 模式按 VAD + 0.5s 滑窗
        if (pvad_mode) {
            if (frame_idx >= 4) score_pvad(t_sec);
        } else if (vad_speech && win.size() >= win_len) {
            std::vector<float> w(win.begin(), win.end());
            score_asnorm(w, t_sec);
        }
        frame_idx++;
    }

    // 离线: 整段音频一次性算 P(target) 序列（整段均值归一化 + 单次 GRU 前向，
    // 与训练/scripts/eval_pvad.py 完全一致）
    void precompute_pvad(const float* pcm, int n) {
        Fbank fbank;
        std::vector<float> feats;
        int T = fbank.compute(pcm, n, feats);
        if (T < 4) return;
        mean_normalize(feats, T, 80);
        p2_pre = pvad->target_probs(feats.data(), T, tpl.pos.data());
    }

    // PVAD 门控: 取当前帧的 P(target) 过迟滞门控。
    // 离线用预计算序列; 实时用"流起始至今(封顶 8s)整段 GRU + 前缀均值归一化"近似。
    // 注意: 不能用 0.5s 短窗 + 每窗重置 GRU——实测该用法对非目标说话人 FAR 很高
    // (voice2 短窗 P(target) 爬到 0.99), 整段上下文下同一模型 P(target)≈0.003。
    void score_pvad(double t_sec) {
        float p;
        if (!p2_pre.empty()) {
            if (frame_idx >= p2_pre.size()) return;
            p = p2_pre[frame_idx];
        } else {
            std::vector<float> w(seg.begin(), seg.end());
            Fbank fbank;
            std::vector<float> feats;
            int T = fbank.compute(w.data(), (int)w.size(), feats);
            if (T < 4) return;
            mean_normalize(feats, T, 80);
            try {
                auto p2 = pvad->target_probs(feats.data(), T, tpl.pos.data());
                p = p2.back();
            } catch (...) { return; }
        }
        bool fire = pgate.update(p);
        printf("[t=%6.2f] p_target=%.4f consec=%d%s\n", t_sec, p, pgate.consec(),
               fire ? "  >>> INTERRUPT <<<" : "");
    }

    void score_asnorm(const std::vector<float>& w, double t_sec) {
        float sA, sNeg, z, mu, sigma;
        try {
            auto emb = spk.embed(w.data(), (int)w.size());
            sA = cosine_sim(tpl.pos, emb);
            sNeg = max_neg_sim(tpl, emb);
            tnorm_stats(tpl, emb, norm_topk, mu, sigma);
            z = (sA - mu) / (sigma > 1e-4f ? sigma : 1e-4f);
        } catch (...) { return; }
        bool fire = gate.update(sA, sNeg, z);
        printf("[t=%6.2f] sA=%.4f z=%6.2f sNeg=%.4f margin=%.4f consec=%d%s\n", t_sec, sA, z,
               sNeg, sA - sNeg, gate.consec(), fire ? "  >>> INTERRUPT <<<" : "");
    }
};

int run_offline(const Args& a) {
    WavData wd = read_wav(a.wav);
    std::vector<float> far_samples;
    if (a.aec && !a.far_wav.empty()) {
        WavData fw = read_wav(a.far_wav);
        far_samples = std::move(fw.samples);
    }
    Pipeline pipe(a);
    std::vector<float> samples = wd.samples;
    // 开降噪时先整段预降噪（流式过一遍），保证 PVAD 整段预计算也作用在干净信号上
    if (pipe.denoise) {
        std::vector<float> out(samples.size());
        size_t nf = samples.size() / 160;
        for (size_t i = 0; i < nf; i++) pipe.denoise->process(&samples[i * 160], &out[i * 160]);
        samples = std::move(out);
        pipe.skip_frame_denoise = true;
    }
    if (pipe.pvad_mode) pipe.precompute_pvad(samples.data(), (int)samples.size());
    printf("offline: %s (%.2fs), AEC=%s, denoise=%s, gate=%s, confirm=%d window=%dms\n",
           a.wav.c_str(), samples.size() / 16000.0,
           (a.aec && !far_samples.empty()) ? a.far_wav.c_str() : "off",
           a.denoise.c_str(), a.gate_mode.c_str(), a.confirm, a.window_ms);
    size_t n = samples.size() / 160;
    for (size_t i = 0; i < n; i++) {
        const float* far_frame = nullptr;
        if (!far_samples.empty() && (i + 1) * 160 <= far_samples.size())
            far_frame = &far_samples[i * 160];
        pipe.process_frame(&samples[i * 160], far_frame, i * 0.01);
    }
    printf("done.\n");
    return 0;
}

// ---------------- 实时模式 ----------------
struct Ring {
    std::mutex mu;
    std::deque<float> q;
    void push(const float* p, size_t n) {
        std::lock_guard<std::mutex> lk(mu);
        q.insert(q.end(), p, p + n);
        if (q.size() > 16000 * 5) q.erase(q.begin(), q.end() - 16000 * 5);  // 兜底 5s
    }
    bool pop(float* out, size_t n) {
        std::lock_guard<std::mutex> lk(mu);
        if (q.size() < n) return false;
        for (size_t i = 0; i < n; i++) { out[i] = q.front(); q.pop_front(); }
        return true;
    }
};

Ring g_mic_ring, g_far_ring;
ma_uint32 g_tone_phase = 0;

void capture_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames) {
    (void)dev; (void)out;
    if (in) g_mic_ring.push((const float*)in, frames);
}

void playback_cb(ma_device* dev, void* out, const void* in, ma_uint32 frames) {
    (void)dev; (void)in;
    float* o = (float*)out;
    for (ma_uint32 i = 0; i < frames; i++) {
        // 440Hz 测试音，幅度 0.3
        o[i] = 0.3f * sinf(2.f * 3.14159265f * 440.f * (g_tone_phase + i) / 16000.f);
    }
    g_tone_phase += frames;
    g_far_ring.push(o, frames);
}

int run_realtime(const Args& a) {
    Pipeline pipe(a);
    printf("realtime: mic @16k mono, gate=%s, AEC=%s, confirm=%d window=%dms, run %ds%s\n",
           a.gate_mode.c_str(), a.aec ? "on" : "off", a.confirm, a.window_ms, a.seconds,
           a.play_tone ? ", play-tone on" : "");

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = 16000;
    cfg.dataCallback = capture_cb;
    ma_device cap;
    if (ma_device_init(nullptr, &cfg, &cap) != MA_SUCCESS) {
        fprintf(stderr, "failed to init capture device\n");
        return 1;
    }

    ma_device pb;
    bool pb_ok = false;
    if (a.play_tone) {
        ma_device_config pcfg = ma_device_config_init(ma_device_type_playback);
        pcfg.playback.format = ma_format_f32;
        pcfg.playback.channels = 1;
        pcfg.sampleRate = 16000;
        pcfg.dataCallback = playback_cb;
        if (ma_device_init(nullptr, &pcfg, &pb) == MA_SUCCESS) pb_ok = true;
        else fprintf(stderr, "playback init failed, continue without tone\n");
    }

    if (ma_device_start(&cap) != MA_SUCCESS) {
        fprintf(stderr, "failed to start capture\n");
        ma_device_uninit(&cap);
        return 1;
    }
    if (pb_ok) ma_device_start(&pb);

    double t = 0;
    int total_frames = a.seconds * 100;
    for (int f = 0; f < total_frames;) {
        float frame[160], far_wav[160];
        if (!g_mic_ring.pop(frame, 160)) { ma_sleep(2); continue; }
        const float* far_ptr = nullptr;
        if (pb_ok && g_far_ring.pop(far_wav, 160)) far_ptr = far_wav;
        pipe.process_frame(frame, far_ptr, t);
        t += 0.01;
        f++;
    }

    ma_device_uninit(&cap);
    if (pb_ok) ma_device_uninit(&pb);
    printf("done.\n");
    return 0;
}

}  // namespace

// ---------------- 批量模式（回归网格用） ----------------
// 输入列表每行: <wav路径>\t<tpl路径>；输出每行: <wav路径>\t<首个INTERRUPT帧|-1>\t<max_p>
// 模型只加载一次；每个文件的处理与 run_offline 的 pvad 路径语义一致：
// （可选整段预降噪）→ 整段 fbank+PVAD 预计算 → 逐帧 silero VAD（speech-end 复位门控）
// → PvadGate 扫描。enrollment 用预算好的 tpl（与历史 eval_cpp_pvad.py 的现场 enroll 等价）。
int run_batch(const Args& a) {
    FILE* fin = fopen(a.batch_list.c_str(), "r");
    if (!fin) throw std::runtime_error("cannot open batch list: " + a.batch_list);
    FILE* fout = fopen(a.batch_out.c_str(), "w");
    if (!fout) throw std::runtime_error("cannot write batch out: " + a.batch_out);

    Vad vad(a.vad_model);
    Pvad pvad(a.pvad_model);
    Fbank fbank;

    char line[8192];
    int idx = 0;
    while (fgets(line, sizeof(line), fin)) {
        std::string s(line);
        size_t tab = s.find('\t');
        if (tab == std::string::npos) continue;
        std::string wav_path = s.substr(0, tab);
        std::string tpl_path = s.substr(tab + 1);
        while (!tpl_path.empty() && (tpl_path.back() == '\n' || tpl_path.back() == '\r' || tpl_path.back() == ' ' || tpl_path.back() == '\t'))
            tpl_path.pop_back();
        if (wav_path.empty() || tpl_path.empty()) continue;

        WavData wd = read_wav(wav_path);
        std::vector<float> samples = wd.samples;
        if (a.denoise == "rnnoise") {
            Denoise d;
            std::vector<float> out(samples.size());
            size_t nf = samples.size() / 160;
            for (size_t i = 0; i < nf; i++) d.process(&samples[i * 160], &out[i * 160]);
            samples = std::move(out);
        }
        Template tpl = load_template(tpl_path);

        std::vector<float> feats;
        int T = fbank.compute(samples.data(), (int)samples.size(), feats);
        std::vector<float> p2;
        if (T >= 4) {
            mean_normalize(feats, T, 80);
            p2 = pvad.target_probs(feats.data(), T, tpl.pos.data());
        }

        // 逐帧门控（与 run_offline pvad 路径一致：frame>=4 起评分，VAD speech-end 复位）
        PvadGate gate{a.pvad_threshold, a.pvad_hyst, a.confirm};
        vad.reset();
        bool prev_speech = false;
        int first = -1;
        float maxp = 0.f;
        size_t nf = samples.size() / 160;
        for (size_t t = 0; t < nf; t++) {
            float vp = vad.process(&samples[t * 160], 160);
            if (vp >= 0.f) {
                bool speech = vp > a.vad_threshold;
                if (prev_speech && !speech) gate.reset();
                prev_speech = speech;
            }
            if (t >= 4 && t < p2.size()) {
                float p = p2[t];
                if (p > maxp) maxp = p;
                if (gate.update(p) && first < 0) first = (int)t;
            }
        }
        fprintf(fout, "%s\t%d\t%.4f\n", wav_path.c_str(), first, maxp);
        fflush(fout);
        if (++idx % 50 == 0) printf("[batch] %d files done\n", idx);
    }
    fclose(fin);
    fclose(fout);
    printf("[batch] done, %d files -> %s\n", idx, a.batch_out.c_str());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    Args a = parse_args(argc, argv);
    try {
        if (a.bench_denoise) {
            // 降噪单帧耗时实测：1000 帧预热 + 10000 帧计时
            Denoise d;
            float in[160], out[160];
            for (int i = 0; i < 160; i++) in[i] = 0.05f * ((i * 37 % 100) / 100.f - 0.5f);
            for (int i = 0; i < 1000; i++) d.process(in, out);
            auto t0 = std::chrono::steady_clock::now();
            const int N = 10000;
            for (int i = 0; i < N; i++) d.process(in, out);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
            printf("denoise(rnnoise): %.4f ms/frame (10ms frame), ~%.1f%% of realtime budget (1 thread)\n",
                   ms, ms / 10.0 * 100.0);
            printf("added pipeline latency: %.1f ms (RNNoise 10ms frame + resampler group delay)\n",
                   Denoise::extra_latency_ms());
            return 0;
        }
        if (a.mic) return run_realtime(a);
        if (!a.batch_list.empty()) return run_batch(a);
        if (!a.wav.empty()) return run_offline(a);
        usage();
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
