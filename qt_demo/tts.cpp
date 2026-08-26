// tts.cpp
#include "tts.h"
#include <sherpa-onnx/c-api/c-api.h>
#include <cstring>

Tts::~Tts() {
    if (tts_) SherpaOnnxDestroyOfflineTts(static_cast<const SherpaOnnxOfflineTts*>(tts_));
}

bool Tts::init(const std::string& model_dir, std::string& err) {
    std::string model = model_dir + "/model.onnx";
    std::string tokens = model_dir + "/tokens.txt";
    std::string data_dir = model_dir + "/espeak-ng-data";

    SherpaOnnxOfflineTtsConfig config;
    std::memset(&config, 0, sizeof(config));
    config.model.vits.model = model.c_str();
    config.model.vits.lexicon = "";
    config.model.vits.tokens = tokens.c_str();
    config.model.vits.data_dir = data_dir.c_str();
    config.model.vits.noise_scale = 0.667f;
    config.model.vits.noise_scale_w = 0.8f;
    config.model.vits.length_scale = 1.0f;
    config.model.num_threads = 2;
    config.model.debug = 0;
    config.model.provider = "cpu";
    config.max_num_sentences = 2;

    tts_ = SherpaOnnxCreateOfflineTts(&config);
    if (!tts_) {
        err = "SherpaOnnxCreateOfflineTts failed (check model.onnx/tokens.txt/espeak-ng-data in " + model_dir + ")";
        return false;
    }
    sr_ = SherpaOnnxOfflineTtsSampleRate(static_cast<const SherpaOnnxOfflineTts*>(tts_));
    return true;
}

bool Tts::speak16k(const std::string& text, std::vector<float>& out, int& native_sr, std::string& err) {
    if (!tts_) { err = "tts not initialized"; return false; }
    SherpaOnnxGenerationConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.sid = 0;
    cfg.speed = 1.0f;
    cfg.silence_scale = 0.2f;
    const SherpaOnnxGeneratedAudio* audio = SherpaOnnxOfflineTtsGenerateWithConfig(
        static_cast<const SherpaOnnxOfflineTts*>(tts_), text.c_str(), &cfg, nullptr, nullptr);
    if (!audio || audio->n <= 0) {
        err = "TTS generate failed";
        if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        return false;
    }
    native_sr = audio->sample_rate;
    if (native_sr == 16000) {
        out.assign(audio->samples, audio->samples + audio->n);
    } else {
        // 线性插值重采样到 16k
        double ratio = 16000.0 / native_sr;
        size_t n16 = (size_t)(audio->n * ratio);
        out.resize(n16);
        for (size_t i = 0; i < n16; i++) {
            double src = i / ratio;
            size_t j = (size_t)src;
            double frac = src - j;
            float a = audio->samples[j];
            float b = (j + 1 < (size_t)audio->n) ? audio->samples[j + 1] : a;
            out[i] = (float)(a + frac * (b - a));
        }
    }
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    return true;
}
