// main.cpp - pvad_demo: GUI 或 --auto-test 无头验证
#include <QApplication>
#include <QStringList>
#include <cstdio>
#include "autotest.h"
#include "mainwindow.h"
#include <miniaudio.h>

// 音频链路冒烟：尝试打开默认采集/播放设备
static int probe_audio() {
    int rc = 0;
    {
        ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
        cfg.capture.format = ma_format_f32;
        cfg.capture.channels = 1;
        cfg.sampleRate = 16000;
        cfg.dataCallback = [](ma_device*, void*, const void*, ma_uint32) {};
        ma_device dev;
        if (ma_device_init(nullptr, &cfg, &dev) == MA_SUCCESS) {
            ma_result r = ma_device_start(&dev);
            printf("[probe-audio] capture: open+start %s\n", r == MA_SUCCESS ? "OK" : "FAILED");
            ma_device_uninit(&dev);
            if (r != MA_SUCCESS) rc = 1;
        } else {
            printf("[probe-audio] capture: open FAILED (no mic?)\n");
            rc = 1;
        }
    }
    {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_f32;
        cfg.playback.channels = 1;
        cfg.sampleRate = 16000;
        cfg.dataCallback = [](ma_device*, void* out, const void*, ma_uint32 frames) {
            float* o = static_cast<float*>(out);
            for (ma_uint32 i = 0; i < frames; i++) o[i] = 0.f;
        };
        ma_device dev;
        if (ma_device_init(nullptr, &cfg, &dev) == MA_SUCCESS) {
            ma_result r = ma_device_start(&dev);
            printf("[probe-audio] playback: open+start %s\n", r == MA_SUCCESS ? "OK" : "FAILED");
            ma_device_uninit(&dev);
            if (r != MA_SUCCESS) rc = 1;
        } else {
            printf("[probe-audio] playback: open FAILED (no speaker?)\n");
            rc = 1;
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // 无头回归异常退出时保留最后进度
    QStringList args;
    for (int i = 0; i < argc; i++) args << QString::fromLocal8Bit(argv[i]);
    // GUI 与无头测试统一走运行时路径；发布包由 run_demo.bat 设置为包目录。
    // 统一正斜杠，旧格式持久化测试会把绝对 wav 路径写入 JSON。
    QString root_q = qEnvironmentVariable("DEMO_ROOT", DEMO_ROOT);
    root_q.replace('\\', '/');
    const std::string runtime_root = root_q.toStdString();
    QString tts_q = qEnvironmentVariable("TTS_MODEL_DIR", TTS_MODEL_DIR);
    tts_q.replace('\\', '/');
    const std::string runtime_tts_dir = tts_q.toStdString();

    if (args.contains("--probe-audio")) return probe_audio();

    if (args.contains("--record-test")) {
        int idx = args.indexOf("--record-test");
        int secs = 3;
        if (idx + 1 < args.size()) {
            bool ok = false;
            int v = args[idx + 1].toInt(&ok);
            if (ok && v > 0) secs = v;
        }
        return run_record_test(runtime_root, secs);
    }

    if (args.contains("--wizard-test")) return run_wizard_test(runtime_root);

    if (args.contains("--persist-test")) return run_persist_test(runtime_root);

    if (args.contains("--ui-state-test")) return run_ui_state_test();

    if (args.contains("--agc-test")) return run_agc_test();

    if (args.contains("--tse-smoke")) return run_tse_smoke(runtime_root);

    if (args.contains("--confirm-test")) {
        std::string tts_dir = runtime_tts_dir;
        int idx = args.indexOf("--tts-model");
        if (idx >= 0 && idx + 1 < args.size()) tts_dir = args[idx + 1].toStdString();
        return run_confirm_test(runtime_root, tts_dir);
    }

    if (args.contains("--story-test") || args.contains("--direct-asr-test")) {
        std::string tts_dir = runtime_tts_dir;
        int idx = args.indexOf("--tts-model");
        if (idx >= 0 && idx + 1 < args.size()) tts_dir = args[idx + 1].toStdString();
        float sg = 0.45f;
        int gi = args.indexOf("--story-gain");
        if (gi >= 0 && gi + 1 < args.size()) sg = args[gi + 1].toFloat();
        float mt = 0.5f;
        int mi = args.indexOf("--mask-thr");
        if (mi >= 0 && mi + 1 < args.size()) mt = args[mi + 1].toFloat();
        return run_story_test(runtime_root, tts_dir, sg, mt, args.contains("--direct-asr-test"));
    }

    if (args.contains("--auto-test")) {
        std::string tts_dir = runtime_tts_dir;
        bool use_denoise = true;
        int idx = args.indexOf("--tts-model");
        if (idx >= 0 && idx + 1 < args.size()) tts_dir = args[idx + 1].toStdString();
        int di = args.indexOf("--denoise");
        if (di >= 0 && di + 1 < args.size()) use_denoise = (args[di + 1] != "off");
        return run_auto_test(runtime_root, tts_dir, use_denoise);
    }

    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
