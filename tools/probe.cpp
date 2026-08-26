// probe.cpp - 打印 ONNX 模型的输入/输出名和 shape
#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <string>
#include <vector>

static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }

static void dump(Ort::Env& env, const std::string& path) {
    printf("=== %s ===\n", path.c_str());
    Ort::Session session(env, widen(path).c_str(), Ort::SessionOptions{});
    Ort::AllocatorWithDefaultOptions alloc;
    auto print = [&](bool input) {
        size_t n = input ? session.GetInputCount() : session.GetOutputCount();
        printf("%s (%zu):\n", input ? "inputs" : "outputs", n);
        for (size_t i = 0; i < n; i++) {
            auto name = input ? session.GetInputNameAllocated(i, alloc) : session.GetOutputNameAllocated(i, alloc);
            auto info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
            auto t = info.GetTensorTypeAndShapeInfo();
            auto shape = t.GetShape();
            printf("  [%zu] %s type=%d shape=[", i, name.get(), (int)t.GetElementType());
            for (auto d : shape) printf("%lld ", (long long)d);
            printf("]\n");
        }
    };
    print(true);
    print(false);
}

int main(int argc, char** argv) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "probe");
    for (int i = 1; i < argc; i++) {
        try { dump(env, argv[i]); }
        catch (const std::exception& e) { printf("error: %s\n", e.what()); }
    }
    return 0;
}
