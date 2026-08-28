// speaker.h - CAM++ 声纹 embedding (ONNX) 封装
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

class SpeakerEmbedder {
public:
    explicit SpeakerEmbedder(const std::string& model_path);
    ~SpeakerEmbedder();
    SpeakerEmbedder(const SpeakerEmbedder&) = delete;
    SpeakerEmbedder& operator=(const SpeakerEmbedder&) = delete;

    // 输入 16k 单声道 PCM，输出 L2 归一化后的 embedding（192 维）
    std::vector<float> embed(const float* pcm, int num_samples);
    // 多帧 enrollment tokens：按 1s 整段切分（尾段丢弃），每段过 CAM++ 各得一个
    // L2 归一化 embedding（[N][192]，N = num_samples/16000；与 scripts/precompute_features.py
    // --tokens 规则一致）
    std::vector<std::vector<float>> embed_tokens(const float* pcm, int num_samples);
    int dim() const { return dim_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    int dim_ = 0;
};

float cosine_sim(const std::vector<float>& a, const std::vector<float>& b);
void l2_normalize(std::vector<float>& v);

// 模板 v3/v4：正模板质心（注册用户 A）+ 负模板质心列表 + N 个独立 cohort embedding
// （打分归一化用）+ v4 追加多帧 enrollment tokens（交叉注意力模型用）
struct Template {
    std::vector<float> pos;                                  // A 的质心（L2 归一化）
    std::vector<std::pair<std::string, std::vector<float>>> neg;  // (标签, 质心)
    std::vector<std::vector<float>> cohort;                  // 独立 cohort embedding（L2 归一化）
    std::vector<std::vector<float>> tokens;                  // v4: enrollment tokens [N][192]
    int dim() const { return (int)pos.size(); }
};

// 与所有负质心相似度的最大值；无负模板时返回 0
float max_neg_sim(const Template& tpl, const std::vector<float>& emb);

// test-side t-norm：x 与全部 cohort 的余弦取 top-K 的均值 mu 和标准差 sigma（总体标准差）
// topk <= 0 时用全部 cohort；cohort 为空时 mu=0, sigma=1
void tnorm_stats(const Template& tpl, const std::vector<float>& emb, int topk, float& mu, float& sigma);

// 二进制格式 v4（向后兼容读 v3）：
//   int32 version=4, int32 dim, float pos[dim],
//   int32 n_neg, 每组负质心: int32 label_len, char label[label_len], float centroid[dim],
//   int32 n_cohort, float cohort[n_cohort * dim],
//   int32 n_tok, float tokens[n_tok * dim]   // v4 追加；v3 文件无此段，读时 tokens 为空
void save_template(const std::string& path, const Template& tpl);
Template load_template(const std::string& path);
