// tts.h - sherpa-onnx OfflineTts (VITS) 封装，输出重采样到 16k 单声道 float
#pragma once
#include <string>
#include <vector>

class Tts {
public:
    ~Tts();
    // model_dir: 含 model.onnx / tokens.txt / espeak-ng-data 的目录
    bool init(const std::string& model_dir, std::string& err);
    bool ok() const { return tts_ != nullptr; }
    // 合成 text，线性重采样到 16k；native_sr 返回模型原生采样率
    bool speak16k(const std::string& text, std::vector<float>& out, int& native_sr, std::string& err);
    int native_sample_rate() const { return sr_; }

private:
    const void* tts_ = nullptr;  // SherpaOnnxOfflineTts*
    int sr_ = 0;
};
