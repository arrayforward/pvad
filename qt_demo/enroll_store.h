// enroll_store.h - 注册持久化（Qt 无关）
// 目录 qt_demo/enrollment/ 下两个文件：
//   tpl.bin        CLI 模板 v3 格式（src/speaker.h），与 CLI enroll/score/double_voice 双向兼容
//   segments.json  逐段明细 {"segments":[{"wav","duration_s","time","embedding":[...]}]}
// embedding 用 %.9g 写盘（float32 十进制 9 位有效数字可无损往返），加载按序重算 emb_sum，
// 与增量注册逐位一致。
#pragma once
#include <string>
#include <vector>
#include "demo_core.h"

class EnrollStore {
public:
    // 全量重写 tpl.bin + segments.json（fbank_mean 可为空）
    static bool save(const std::string& dir, const std::vector<SegRecord>& segs,
                     const std::vector<float>& centroid, const std::vector<float>& fbank_mean,
                     std::string& err);
    // 加载。fbank_mean 为空表示未存（旧格式/CLI 导入）。
    static bool load(const std::string& dir, std::vector<SegRecord>& segs,
                     std::vector<float>& fbank_mean, bool& loaded, std::string& err);
};
