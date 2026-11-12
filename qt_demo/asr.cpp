// asr.cpp
#include "asr.h"
#include <sherpa-onnx/c-api/c-api.h>
#include <cstring>

Asr::~Asr() {
    if (rec_) SherpaOnnxDestroyOfflineRecognizer(static_cast<const SherpaOnnxOfflineRecognizer*>(rec_));
}

bool Asr::init(const std::string& model_dir, std::string& err) {
    std::string model = model_dir + "/model.int8.onnx";
    std::string tokens = model_dir + "/tokens.txt";
    SherpaOnnxOfflineRecognizerConfig config;
    std::memset(&config, 0, sizeof(config));
    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;
    config.model_config.paraformer.model = model.c_str();
    config.model_config.tokens = tokens.c_str();
    config.model_config.provider = "cpu";
    config.model_config.num_threads = 4;
    config.model_config.debug = 0;
    config.decoding_method = "greedy_search";
    rec_ = SherpaOnnxCreateOfflineRecognizer(&config);
    if (!rec_) {
        err = "SherpaOnnxCreateOfflineRecognizer failed (check " + model_dir + ")";
        return false;
    }
    return true;
}

std::string Asr::decode(const float* pcm, int n) {
    if (!rec_) return "";
    const SherpaOnnxOfflineRecognizer* r = static_cast<const SherpaOnnxOfflineRecognizer*>(rec_);
    const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(r);
    SherpaOnnxAcceptWaveformOffline(stream, 16000, pcm, n);
    SherpaOnnxDecodeOfflineStream(r, stream);
    const SherpaOnnxOfflineRecognizerResult* res = SherpaOnnxGetOfflineStreamResult(stream);
    std::string out = res && res->text ? res->text : "";
    if (res) SherpaOnnxDestroyOfflineRecognizerResult(res);
    SherpaOnnxDestroyOfflineStream(stream);
    return out;
}
