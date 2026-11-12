// autotest.h - 无头自动验证（不弹窗）：三个场景化判定
#pragma once
#include <string>

// 返回 0 = 全部 PASS，1 = 有 FAIL；use_denoise 默认 true（与 CLI/GUI 默认开一致）
int run_auto_test(const std::string& root, const std::string& tts_model_dir, bool use_denoise = true);

// 录音注册代码路径冒烟：默认麦克风录 seconds 秒 -> 存 wav -> CAM++ 注册
int run_record_test(const std::string& root, int seconds);

// 引导注册状态机无头验证：完成 3 段 / 单段太短重录 / 中途取消恢复旧质心
int run_wizard_test(const std::string& root);

// 注册持久化无头验证：保存 -> 新实例加载 -> 质心/emb_sum 逐位一致 + CLI tpl 互操作
int run_persist_test(const std::string& root);

// UI 状态机/背压决策无头验证（UiState 转移 + Backpressure 丢帧语义）
int run_ui_state_test();

// 电平计与 AGC 无头验证（--agc-test）：目标 RMS、静音、峰值与分块一致性
int run_agc_test();

// TSE 链路冒烟（--tse-smoke）：多尺度声纹注册 + e16 step 提取 + ASR 对照
int run_tse_smoke(const std::string& root);

// 听故事打断全链验证（--story-test）：PVAD 守门 → TSE 提取 → ASR，泄漏率对照断言
int run_story_test(const std::string& root, const std::string& tts_model_dir,
                   float story_gain = 0.45f, float mask_thr = 0.5f);
