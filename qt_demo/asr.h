// asr.h - sherpa-onnx 离线 Paraformer ASR 封装（整段识别，demo 用）
#pragma once
#include <string>
#include <vector>

class Asr {
public:
    ~Asr();
    // model_dir: 含 model.int8.onnx + tokens.txt
    bool init(const std::string& model_dir, std::string& err);
    bool ok() const { return rec_ != nullptr; }
    // 16k 单声道 float → 文本；失败返回空串
    std::string decode(const float* pcm, int n);
private:
    const void* rec_ = nullptr;  // SherpaOnnxOfflineRecognizer*
};
