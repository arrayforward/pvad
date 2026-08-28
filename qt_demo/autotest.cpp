// autotest.cpp - 场景化无头验证
// 场景1「TTS 不打断自己」: 注册 A=voice1b，注入合成的 TTS 音频本身(0.6 增益, 模拟回声)，
//         预期无 INTERRUPT
// 场景2「真人录音打断 TTS」: TTS 虚拟播放中注入 voice1b(注册用户)，预期触发 INTERRUPT 且
//         虚拟播放在触发时刻被停止; 再注入 voice2(非注册)，预期不触发
#include "autotest.h"
#include "demo_core.h"
#include "denoise.h"
#include "enroll_store.h"
#include "speaker.h"
#include "tts.h"
#include "ui_state.h"
#include "wav_io.h"
#include "wizard.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <miniaudio.h>

namespace {

struct FeedResult {
    int interrupts = 0;
    float max_p = 0.f;
    double first_interrupt_t = -1.0;
};

FeedResult feed(DemoCore& core, const std::vector<float>& pcm, float gain = 1.0f,
                Denoise* denoise = nullptr) {
    FeedResult r;
    size_t n = pcm.size() / 160;
    for (size_t i = 0; i < n; i++) {
        float frame[160];
        for (int j = 0; j < 160; j++) frame[j] = pcm[i * 160 + j] * gain;
        const float* src = frame;
        float dn[160];
        if (denoise) { denoise->process(frame, dn); src = dn; }
        FrameEvent ev = core.feed_frame(src);
        if (ev.p > r.max_p) r.max_p = ev.p;
        if (ev.interrupt) {
            r.interrupts++;
            if (r.first_interrupt_t < 0) r.first_interrupt_t = ev.t;
        }
    }
    return r;
}

// 整段预降噪（与 CLI run_offline 一致：每个文件一个全新 Denoise——重采样器/RNNoise
// 是有状态的，跨文件复用会把上一文件的残态带进本文件特征，污染判定）
std::vector<float> pre_denoise(const std::vector<float>& pcm, Denoise*) {
    Denoise d;
    std::vector<float> out(pcm.size());
    size_t n = pcm.size() / 160;
    for (size_t i = 0; i < n; i++) d.process(&pcm[i * 160], &out[i * 160]);
    return out;
}

}  // namespace

int run_auto_test(const std::string& root, const std::string& tts_model_dir, bool use_denoise) {
    int fails = 0;
    std::string err;

    // 降噪默认开（与 CLI/GUI 默认值一致）；--denoise off 可关（回滚/对比用）
    std::unique_ptr<Denoise> denoise;
    if (use_denoise) {
        try {
            denoise = std::make_unique<Denoise>();
        } catch (const std::exception& e) {
            printf("[auto-test] denoise init failed (%s), continue without denoise\n", e.what());
        }
    }
    printf("[auto-test] denoise=%s\n", denoise ? "rnnoise" : "off");

    DemoCore core;
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err)) {
        printf("[auto-test] core init failed: %s\n", err.c_str());
        return 1;
    }
    if (!core.enroll({root + "/test_audio/voice1b.wav"}, err)) {
        printf("[auto-test] enroll failed: %s\n", err.c_str());
        return 1;
    }
    printf("[auto-test] enrolled A = test_audio/voice1b.wav\n");

    Tts tts;
    if (!tts.init(tts_model_dir, err)) {
        printf("[auto-test] TTS init failed: %s\n", err.c_str());
        return 1;
    }
    printf("[auto-test] TTS ready (native_sr=%d)\n", tts.native_sample_rate());

    // 场景用 TTS 音频：优先读缓存 test_audio/auto_tts.wav（保证回归可复现），
    // 不存在则现场合成并写入缓存（VITS 合成带随机性，不同 utterance 的 P(target)
    // 余量有波动，缓存后判定稳定；删除该文件可重新生成）
    std::vector<float> tts_pcm;
    int native_sr = tts.native_sample_rate();
    std::string cache = root + "/test_audio/auto_tts.wav";
    bool from_cache = false;
    try {
        WavData cw = read_wav(cache);
        tts_pcm = std::move(cw.samples);
        from_cache = true;
    } catch (...) {
    }
    if (from_cache) {
        printf("[auto-test] TTS audio from cache %s (%.2fs)\n", cache.c_str(), tts_pcm.size() / 16000.0);
    } else {
        if (!tts.speak16k("你好，现在开始为你播报一段较长的消息，请耐心听完，中途不要打断我，谢谢配合。",
                          tts_pcm, native_sr, err)) {
            printf("[auto-test] TTS synth failed: %s\n", err.c_str());
            return 1;
        }
        write_wav16(cache, tts_pcm, 16000);
        printf("[auto-test] TTS synthesized %.2fs (native_sr=%d -> 16k), cached to %s\n",
               tts_pcm.size() / 16000.0, native_sr, cache.c_str());
    }
    double tts_dur = tts_pcm.size() / 16000.0;

    // ---- 场景1: TTS 不打断自己 ----
    {
        core.reset_stream();
        std::vector<float> gained(tts_pcm.size());
        for (size_t i = 0; i < tts_pcm.size(); i++) gained[i] = tts_pcm[i] * 0.6f;
        if (use_denoise) gained = pre_denoise(gained, nullptr);
        if (!core.precompute_file(gained.data(), gained.size(), err)) {
            printf("[auto-test] precompute failed: %s\n", err.c_str());
            return 1;
        }
        FeedResult r = feed(core, gained);
        bool pass = (r.interrupts == 0);
        printf("SCENARIO1 tts-self-echo: interrupts=%d max_p=%.4f -> %s\n",
               r.interrupts, r.max_p, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // ---- 场景2: 真人录音打断 TTS ----
    // 虚拟播放：虚拟时间随注入帧推进；触发时若 t < tts_dur 则记为"播放被停止"
    for (const char* name : {"voice1b", "voice2"}) {
        core.reset_stream();
        WavData wd = read_wav(root + "/test_audio/" + name + ".wav");
        std::vector<float> samples = use_denoise ? pre_denoise(wd.samples, nullptr)
                                             : std::move(wd.samples);
        if (!core.precompute_file(samples.data(), samples.size(), err)) {
            printf("[auto-test] precompute failed: %s\n", err.c_str());
            return 1;
        }
        bool playing = true;
        double stop_t = -1.0;
        FeedResult r;
        size_t n = samples.size() / 160;
        for (size_t i = 0; i < n; i++) {
            FrameEvent ev = core.feed_frame(&samples[i * 160]);
            if (ev.p > r.max_p) r.max_p = ev.p;
            if (ev.interrupt) {
                r.interrupts++;
                if (r.first_interrupt_t < 0) r.first_interrupt_t = ev.t;
                if (playing && ev.t < tts_dur) { playing = false; stop_t = ev.t; }
            }
        }
        bool expect_interrupt = (std::string(name) == "voice1b");
        bool pass;
        if (expect_interrupt) {
            pass = (r.interrupts > 0 && stop_t >= 0);
            printf("SCENARIO2 %s: interrupts=%d first_t=%.2f max_p=%.4f tts_stopped_at=%.2f (dur=%.2f) -> %s\n",
                   name, r.interrupts, r.first_interrupt_t, r.max_p, stop_t, tts_dur,
                   pass ? "PASS" : "FAIL");
        } else {
            pass = (r.interrupts == 0);
            printf("SCENARIO2 %s(non-enrolled): interrupts=%d max_p=%.4f -> %s\n",
                   name, r.interrupts, r.max_p, pass ? "PASS" : "FAIL");
        }
        if (!pass) fails++;
    }

    printf("[auto-test] %s\n", fails == 0 ? "ALL PASS" : "HAS FAILURES");
    return fails == 0 ? 0 : 1;
}

// ---------------- 录音注册冒烟（真实麦克风） ----------------

int run_record_test(const std::string& root, int seconds) {
    printf("[record-test] recording %ds from default mic...\n", seconds);
    std::vector<float> buf;
    std::mutex mu;
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = 16000;
    struct Ctx { std::vector<float>* buf; std::mutex* mu; } ctx{&buf, &mu};
    cfg.dataCallback = [](ma_device* dev, void*, const void* in, ma_uint32 frames) {
        auto* c = static_cast<Ctx*>(dev->pUserData);
        if (!in) return;
        std::lock_guard<std::mutex> lk(*c->mu);
        const float* p = static_cast<const float*>(in);
        c->buf->insert(c->buf->end(), p, p + frames);
    };
    cfg.pUserData = &ctx;
    ma_device dev;
    if (ma_device_init(nullptr, &cfg, &dev) != MA_SUCCESS) {
        printf("[record-test] capture open FAILED (no mic?)\n");
        return 1;
    }
    if (ma_device_start(&dev) != MA_SUCCESS) {
        printf("[record-test] capture start FAILED\n");
        ma_device_uninit(&dev);
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    ma_device_uninit(&dev);

    double dur = buf.size() / 16000.0;
    printf("[record-test] captured %.2fs (%zu samples)\n", dur, buf.size());
    if (buf.size() < 16000 * 2) {
        printf("[record-test] too short (<2s), would be rejected by UI logic\n");
        return 1;
    }
    // 保存 wav（文件名带时间戳）
    std::string dir = root + "/qt_demo/recordings";
    std::filesystem::create_directories(dir);
    char ts[32];
    std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string fname = dir + "/rec_" + ts + ".wav";
    write_wav16(fname, buf, 16000);
    printf("[record-test] saved %s\n", fname.c_str());

    // CAM++ 注册
    DemoCore core;
    std::string err;
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err)) {
        printf("[record-test] core init failed: %s\n", err.c_str());
        return 1;
    }
    if (!core.enroll_samples(buf.data(), buf.size(), err)) {
        printf("[record-test] enroll_samples failed: %s\n", err.c_str());
        return 1;
    }
    printf("[record-test] enrolled OK, segments=%d -> PASS\n", core.enroll_count());
    return 0;
}

// ---------------- 引导注册状态机无头验证 ----------------

int run_wizard_test(const std::string& root) {
    int fails = 0;
    std::string err;
    DemoCore core;
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err)) {
        printf("[wizard-test] core init failed: %s\n", err.c_str());
        return 1;
    }
    // 预注册旧质心（模拟用户已有注册）
    if (!core.enroll({root + "/test_audio/voice1.wav"}, err)) {
        printf("[wizard-test] pre-enroll failed: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> old_sum;
    int old_n = 0;
    { std::vector<SegRecord> s_; core.get_enroll_state(old_sum, old_n, s_); }

    WavData wd = read_wav(root + "/test_audio/voice1b.wav");  // ~9.8s
    const float* pcm = wd.samples.data();

    // 路径1「单段太短重录」：1s 拒收，不计入进度；3s 通过
    {
        WizardController w;
        w.start(core);
        bool rejected = !w.accept_segment(core, pcm, 16000 * 1) && w.step() == 0;
        bool accepted = w.accept_segment(core, pcm, 16000 * 3) && w.step() == 1;
        bool pass = rejected && accepted;
        printf("WIZARD path1 short-reject-retry: reject_1s=%d accept_3s=%d -> %s\n",
               (int)rejected, (int)accepted, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
        // 继续走完剩余 2 段（供路径2 的"完成 3 段"复用同一向导）
        bool ok2 = w.accept_segment(core, pcm + 16000 * 3, 16000 * 3) && w.step() == 2;
        bool ok3 = w.accept_segment(core, pcm + 16000 * 4, 16000 * 5) &&
                   w.step() == WizardController::kTotal && !w.active();
        bool done = ok2 && ok3 && core.enroll_count() == 3;
        printf("WIZARD path2 complete-3-segments: step=%d active=%d enrolled=%d -> %s\n",
               w.step(), (int)w.active(), core.enroll_count(), done ? "PASS" : "FAIL");
        if (!done) fails++;
    }

    // 路径3「中途取消恢复旧质心」：向导开始后备份当前状态，录 1 段后取消，状态须逐位还原
    {
        std::vector<float> pre_sum;
        int pre_n = 0;
        { std::vector<SegRecord> s_; core.get_enroll_state(pre_sum, pre_n, s_); }  // 当前为路径1/2 后的 3 段状态
        WizardController w;
        w.start(core);
        if (!w.accept_segment(core, pcm, 16000 * 3)) {
            printf("WIZARD path3 cancel-restore: accept failed -> FAIL\n");
            fails++;
        } else {
            w.cancel(core);
            std::vector<float> post_sum;
            int post_n = 0;
            { std::vector<SegRecord> s_; core.get_enroll_state(post_sum, post_n, s_); }
            bool pass = (post_n == pre_n) && (post_sum == pre_sum) && !w.active();
            printf("WIZARD path3 cancel-restore: n %d->%d sum_equal=%d -> %s\n",
                   pre_n, post_n, (int)(post_sum == pre_sum), pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    // 路径4「取消发生在 0 段时」：直接取消也须还原（边界）
    {
        std::vector<float> pre_sum;
        int pre_n = 0;
        { std::vector<SegRecord> s_; core.get_enroll_state(pre_sum, pre_n, s_); }
        WizardController w;
        w.start(core);  // 清空注册
        if (core.enrolled()) {
            printf("WIZARD path4 cancel-at-step0: enroll not cleared on start -> FAIL\n");
            fails++;
        } else {
            w.cancel(core);
            std::vector<float> post_sum;
            int post_n = 0;
            { std::vector<SegRecord> s_; core.get_enroll_state(post_sum, post_n, s_); }
            bool pass = (post_n == pre_n) && (post_sum == pre_sum);
            printf("WIZARD path4 cancel-at-step0: restored=%d -> %s\n", (int)pass,
                   pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    printf("[wizard-test] %s\n", fails == 0 ? "ALL PASS" : "HAS FAILURES");
    return fails == 0 ? 0 : 1;
}

// ---------------- 注册持久化无头验证 ----------------

int run_persist_test(const std::string& root) {
    int fails = 0;
    std::string err;
    std::string dir = root + "/build/persist_test/enrollment";
    std::filesystem::remove_all(root + "/build/persist_test");

    // 1) 注册 2 段（带元数据）-> 落盘
    DemoCore core;
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err)) {
        printf("[persist-test] core init failed: %s\n", err.c_str());
        return 1;
    }
    WavData wd = read_wav(root + "/test_audio/voice1b.wav");
    if (!core.enroll_samples(wd.samples.data(), 16000 * 3, err,
                             "recordings/rec_a.wav", 3.0, "2026-08-27 10:00:00") ||
        !core.enroll_samples(wd.samples.data() + 16000 * 3, 16000 * 3, err,
                             "recordings/rec_b.wav", 3.0, "2026-08-27 10:01:00")) {
        printf("[persist-test] enroll failed: %s\n", err.c_str());
        return 1;
    }
    if (!EnrollStore::save(dir, core.segments(), core.centroid(), core.fbank_mean(), err)) {
        printf("[persist-test] save failed: %s\n", err.c_str());
        return 1;
    }
    printf("[persist-test] saved %d segments -> %s\n", core.enroll_count(), dir.c_str());

    // 2) 模拟新进程：新 DemoCore 加载 -> 逐位比对
    std::vector<SegRecord> segs;
    std::vector<float> fm_, fm2_, fm3_;
    bool loaded = false;
    if (!EnrollStore::load(dir, segs, fm_, loaded, err) || !loaded) {
        printf("[persist-test] load failed: %s\n", err.c_str());
        return 1;
    }
    DemoCore core2;
    core2.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err);
    core2.set_segments(segs);
    double max_diff = 0;
    for (size_t i = 0; i < core.centroid().size(); i++)
        max_diff = std::max(max_diff, (double)std::fabs(core.centroid()[i] - core2.centroid()[i]));
    {
        std::vector<float> s1, s2;
        int n1, n2;
        std::vector<SegRecord> g1, g2;
        core.get_enroll_state(s1, n1, g1);
        core2.get_enroll_state(s2, n2, g2);
        bool sum_equal = (n1 == n2) && (s1 == s2);
        bool seg_equal = (g1.size() == g2.size());
        printf("PERSIST reload-bitexact: centroid_max_diff=%.3g sum_equal=%d seg_count=%d -> %s\n",
               max_diff, (int)sum_equal, (int)g2.size(),
               (max_diff == 0 && sum_equal && seg_equal) ? "PASS" : "FAIL");
        if (!(max_diff == 0 && sum_equal && seg_equal)) fails++;
    }

    // 3) CLI 互操作：qt_demo 存的 tpl.bin 用 CLI 模板加载，与质心余弦应为 1
    {
        Template tpl = load_template(dir + "/tpl.bin");
        double dot = 0;
        for (size_t i = 0; i < tpl.pos.size(); i++) dot += (double)tpl.pos[i] * core.centroid()[i];
        bool pass = dot > 1.0 - 1e-6;
        printf("PERSIST tpl-interop: cos(tpl.pos, centroid)=%.8f -> %s\n", dot,
               pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // 4) CLI 导入路径：仅 tpl.bin（无 segments.json）-> 单段恢复且可继续增量注册
    {
        std::string dir2 = root + "/build/persist_test/enrollment_cli";
        std::filesystem::create_directories(dir2);
        save_template(dir2 + "/tpl.bin", Template{core.centroid(), {}, {}});
        std::vector<SegRecord> s2;
        bool l2 = false;
        std::string e2;
        bool ok = EnrollStore::load(dir2, s2, fm2_, l2, e2);
        DemoCore core3;
        core3.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err);
        core3.set_segments(s2);
        double dot = 0;
        for (size_t i = 0; i < core.centroid().size(); i++)
            dot += (double)core3.centroid()[i] * core.centroid()[i];
        bool pass = ok && l2 && s2.size() == 1 && dot > 1.0 - 1e-6;
        printf("PERSIST cli-tpl-import: segments=%zu cos=%.8f -> %s\n", s2.size(), dot,
               pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // 5) 损坏文件：不崩溃，降级空注册
    {
        std::string dir3 = root + "/build/persist_test/enrollment_bad";
        std::filesystem::create_directories(dir3);
        FILE* f = fopen((dir3 + "/segments.json").c_str(), "w");
        fprintf(f, "{\"segments\":[{garbage!!!");
        fclose(f);
        std::vector<SegRecord> s3;
        bool l3 = false;
        std::string e3;
        bool ok = EnrollStore::load(dir3, s3, fm3_, l3, e3);
        bool pass = !ok && !l3;  // 报错但不崩溃、loaded=false
        printf("PERSIST corrupt-file-degrade: ok=%d loaded=%d err=%.40s -> %s\n", (int)ok,
               (int)l3, e3.c_str(), pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // 5) v5 tokens：落盘往返逐位一致 + 旧格式升级（重算/降级）
    {
        // 5a) tokens 往返
        std::vector<SegRecord> segs5;
        std::vector<float> fm5;
        bool l5 = false;
        std::string e5;
        EnrollStore::load(dir, segs5, fm5, l5, e5);
        bool tok_ok = l5 && segs5.size() == core.segments().size();
        if (tok_ok) {
            for (size_t i = 0; i < segs5.size(); i++) {
                if (segs5[i].tokens.size() != core.segments()[i].tokens.size()) { tok_ok = false; break; }
                for (size_t k = 0; k < segs5[i].tokens.size(); k++)
                    if (segs5[i].tokens[k] != core.segments()[i].tokens[k]) { tok_ok = false; break; }
            }
        }
        printf("PERSIST v5-tokens-roundtrip: ok=%d -> %s\n", (int)tok_ok, tok_ok ? "PASS" : "FAIL");
        if (!tok_ok) fails++;

        // 5b) 旧格式（无 tokens 字段）+ wav 在（绝对路径真实文件） -> 重算升级
        std::string dir_old = root + "/build/persist_test/enrollment_oldfmt";
        std::filesystem::create_directories(dir_old);
        {
            const SegRecord& s0 = core.segments()[0];
            std::string real_wav = root + "/test_audio/voice1b.wav";  // 绝对路径
            std::string jf = dir_old + "/segments.json";
            FILE* f = fopen(jf.c_str(), "wb");
            fprintf(f, "{\"segments\":[\n {\"wav\": \"%s\", \"duration_s\": %.3f, \"time\": \"%s\", \"embedding\": [",
                    real_wav.c_str(), s0.duration_s, s0.time.c_str());
            for (size_t j = 0; j < s0.emb.size(); j++)
                fprintf(f, "%s%.9g", j ? ", " : "", (double)s0.emb[j]);
            fprintf(f, "]}]}\n");
            fclose(f);
        }
        std::vector<SegRecord> so;
        std::vector<float> fmo;
        bool lo = false;
        std::string eo;
        EnrollStore::load(dir_old, so, fmo, lo, eo);
        DemoCore core_old;
        core_old.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v5.onnx", err);
        core_old.set_segments(so);
        int fr = core_old.rebuild_missing_tokens(root + "/qt_demo");
        bool upg = (fr == 0) && !core_old.all_tokens().empty();
        printf("PERSIST v5-oldfmt-upgrade: failed=%d tokens=%zu -> %s\n", fr,
               core_old.all_tokens().size(), upg ? "PASS" : "FAIL");
        if (!upg) fails++;

        // 5c) 旧格式 + wav 被删 -> 降级（failed>0, tokens 空, 不崩溃）
        std::vector<SegRecord> sbad(1);
        sbad[0].wav = "/nonexistent/rec_gone.wav";
        sbad[0].duration_s = 3.0;
        sbad[0].time = "2026-08-27 12:00:00";
        sbad[0].emb = core.segments()[0].emb;
        core_old.set_segments(sbad);
        int fr2 = core_old.rebuild_missing_tokens(root + "/qt_demo");
        bool deg = (fr2 == 1) && core_old.all_tokens().empty();
        printf("PERSIST v5-upgrade-degrade: failed=%d tokens_empty=%d -> %s\n", fr2,
               (int)core_old.all_tokens().empty(), deg ? "PASS" : "FAIL");
        if (!deg) fails++;
    }

    printf("[persist-test] %s\n", fails == 0 ? "ALL PASS" : "HAS FAILURES");
    return fails == 0 ? 0 : 1;
}

// ---------------- UI 状态机 / 背压决策无头验证 ----------------

int run_ui_state_test() {
    int fails = 0;
    auto check = [&](const char* name, bool cond) {
        printf("UI-STATE %-44s %s\n", name, cond ? "PASS" : "FAIL");
        if (!cond) fails++;
    };

    // 1) idle 初始：全部可用（停止监听除外）
    {
        UiState ui;
        check("idle: speak/listen/enroll/wizard enabled",
              ui.canSpeak() && ui.canStartListen() && ui.canEnroll() &&
              ui.canStartWizard() && ui.canRecord() && !ui.canStopListen());
    }
    // 2) 监听中：停止可用、朗读可用、监听/注册/向导禁用
    {
        UiState ui;
        ui.onListenChanged(true);
        check("listening: stop enabled, speak enabled",
              ui.canStopListen() && ui.canSpeak());
        check("listening: listen/enroll/wizard/record disabled",
              !ui.canStartListen() && !ui.canEnroll() && !ui.canStartWizard() &&
              !ui.canRecord());
        // 3) 监听中点停止 -> 回 idle -> 全部恢复
        ui.onListenChanged(false);
        check("stop-listen -> idle: speak/listen enabled again",
              ui.canSpeak() && ui.canStartListen() && ui.canEnroll() &&
              ui.canStartWizard() && !ui.canStopListen());
    }
    // 4) 录音中：监听禁用、朗读可用；录音结束恢复
    {
        UiState ui;
        ui.onRecordChanged(true);
        check("recording: listen disabled, speak enabled",
              !ui.canStartListen() && ui.canSpeak());
        ui.onRecordChanged(false);
        check("record done -> idle restored", !ui.busy() && ui.canStartListen());
    }
    // 5) 向导中：朗读禁用（关键互斥）；取消向导恢复
    {
        UiState ui;
        ui.onWizardChanged(true);
        check("wizard: speak DISABLED", !ui.canSpeak());
        check("wizard: listen/enroll/record disabled",
              !ui.canStartListen() && !ui.canEnroll() && !ui.canRecord());
        ui.onWizardChanged(false);
        check("wizard cancel -> speak enabled again", ui.canSpeak() && !ui.busy());
    }
    // 6) 背压语义：积压 <=1s 不丢；>1s 丢弃到保留最新 0.5s；有界单调
    {
        check("backpressure: no drop below threshold",
              Backpressure::drop_count(16000) == 0 &&
              Backpressure::drop_count(0) == 0 &&
              Backpressure::drop_count(15999) == 0);
        size_t d = Backpressure::drop_count(16000 * 5);
        check("backpressure: 5s backlog -> keep newest 0.5s",
              d == 16000 * 5 - Backpressure::kKeepSamples &&
              (16000 * 5 - d) == Backpressure::kKeepSamples);
        check("backpressure: monotonic + bounded",
              Backpressure::drop_count(16000 * 3) <= Backpressure::drop_count(16000 * 5) &&
              Backpressure::kMaxPerTick > 0 && Backpressure::kMaxPerTick <= 20);
    }

    printf("[ui-state-test] %s\n", fails == 0 ? "ALL PASS" : "HAS FAILURES");
    return fails == 0 ? 0 : 1;
}
