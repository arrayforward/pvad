// autotest.cpp - 场景化无头验证
// 场景1「TTS 不打断自己」: 注册 A=voice1b，注入合成的 TTS 音频本身(0.6 增益, 模拟回声)，
//         预期无 INTERRUPT
// 场景2「真人录音打断 TTS」: TTS 虚拟播放中注入 voice1b(注册用户)，预期触发 INTERRUPT 且
//         虚拟播放在触发时刻被停止; 再注入 voice2(非注册)，预期不触发
#include "autotest.h"
#include "demo_core.h"
#include "tts.h"
#include "wav_io.h"
#include "wizard.h"
#include <cstdio>
#include <chrono>
#include <ctime>
#include <filesystem>
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

FeedResult feed(DemoCore& core, const std::vector<float>& pcm, float gain = 1.0f) {
    FeedResult r;
    size_t n = pcm.size() / 160;
    for (size_t i = 0; i < n; i++) {
        float frame[160];
        for (int j = 0; j < 160; j++) frame[j] = pcm[i * 160 + j] * gain;
        FrameEvent ev = core.feed_frame(frame);
        if (ev.p > r.max_p) r.max_p = ev.p;
        if (ev.interrupt) {
            r.interrupts++;
            if (r.first_interrupt_t < 0) r.first_interrupt_t = ev.t;
        }
    }
    return r;
}

}  // namespace

int run_auto_test(const std::string& root, const std::string& tts_model_dir) {
    int fails = 0;
    std::string err;

    DemoCore core;
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v3.onnx", err)) {
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
        if (!core.precompute_file(wd.samples.data(), wd.samples.size(), err)) {
            printf("[auto-test] precompute failed: %s\n", err.c_str());
            return 1;
        }
        bool playing = true;
        double stop_t = -1.0;
        FeedResult r;
        size_t n = wd.samples.size() / 160;
        for (size_t i = 0; i < n; i++) {
            FrameEvent ev = core.feed_frame(&wd.samples[i * 160]);
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
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v3.onnx", err)) {
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
    if (!core.init(root + "/models/campplus.onnx", root + "/models/pvad/pvad_v3.onnx", err)) {
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
    core.get_enroll_state(old_sum, old_n);

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
        core.get_enroll_state(pre_sum, pre_n);  // 当前为路径1/2 后的 3 段状态
        WizardController w;
        w.start(core);
        if (!w.accept_segment(core, pcm, 16000 * 3)) {
            printf("WIZARD path3 cancel-restore: accept failed -> FAIL\n");
            fails++;
        } else {
            w.cancel(core);
            std::vector<float> post_sum;
            int post_n = 0;
            core.get_enroll_state(post_sum, post_n);
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
        core.get_enroll_state(pre_sum, pre_n);
        WizardController w;
        w.start(core);  // 清空注册
        if (core.enrolled()) {
            printf("WIZARD path4 cancel-at-step0: enroll not cleared on start -> FAIL\n");
            fails++;
        } else {
            w.cancel(core);
            std::vector<float> post_sum;
            int post_n = 0;
            core.get_enroll_state(post_sum, post_n);
            bool pass = (post_n == pre_n) && (post_sum == pre_sum);
            printf("WIZARD path4 cancel-at-step0: restored=%d -> %s\n", (int)pass,
                   pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    printf("[wizard-test] %s\n", fails == 0 ? "ALL PASS" : "HAS FAILURES");
    return fails == 0 ? 0 : 1;
}
