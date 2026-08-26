// autotest.h - 无头自动验证（不弹窗）：三个场景化判定
#pragma once
#include <string>

// 返回 0 = 全部 PASS，1 = 有 FAIL
int run_auto_test(const std::string& root, const std::string& tts_model_dir);

// 录音注册代码路径冒烟：默认麦克风录 seconds 秒 -> 存 wav -> CAM++ 注册
int run_record_test(const std::string& root, int seconds);

// 引导注册状态机无头验证：完成 3 段 / 单段太短重录 / 中途取消恢复旧质心
int run_wizard_test(const std::string& root);
