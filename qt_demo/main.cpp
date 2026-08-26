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
    QStringList args;
    for (int i = 0; i < argc; i++) args << QString::fromLocal8Bit(argv[i]);

    if (args.contains("--probe-audio")) return probe_audio();

    if (args.contains("--record-test")) {
        int idx = args.indexOf("--record-test");
        int secs = 3;
        if (idx + 1 < args.size()) {
            bool ok = false;
            int v = args[idx + 1].toInt(&ok);
            if (ok && v > 0) secs = v;
        }
        return run_record_test(DEMO_ROOT, secs);
    }

    if (args.contains("--wizard-test")) return run_wizard_test(DEMO_ROOT);

    if (args.contains("--auto-test")) {
        std::string root = DEMO_ROOT;
        std::string tts_dir = TTS_MODEL_DIR;
        bool use_denoise = true;
        int idx = args.indexOf("--tts-model");
        if (idx >= 0 && idx + 1 < args.size()) tts_dir = args[idx + 1].toStdString();
        int di = args.indexOf("--denoise");
        if (di >= 0 && di + 1 < args.size()) use_denoise = (args[di + 1] != "off");
        return run_auto_test(root, tts_dir, use_denoise);
    }

    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
