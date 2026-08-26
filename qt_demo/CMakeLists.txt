cmake_minimum_required(VERSION 3.20)
project(pvad_bargein_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
get_filename_component(PROJ_ROOT "${PROJ_ROOT}" ABSOLUTE)
set(ORT_DIR ${PROJ_ROOT}/third_party/onnxruntime)
set(SHERPA_DIR ${PROJ_ROOT}/third_party/sherpa-onnx)
set(TTS_MODEL_DIR "D:/vit/tts/models/vits-piper-zh_CN-huayan-medium" CACHE PATH "vits-piper TTS 模型目录")

# 管线核心（复用 ../src，不含 AEC/speexdsp、不含 silero vad——demo 走 pvad 门控）
add_library(pipe STATIC
    ${PROJ_ROOT}/src/fbank.cpp
    ${PROJ_ROOT}/src/speaker.cpp
    ${PROJ_ROOT}/src/pvad.cpp
)
target_include_directories(pipe PUBLIC ${PROJ_ROOT}/src ${ORT_DIR}/include)
target_link_libraries(pipe PUBLIC ${ORT_DIR}/lib/onnxruntime.lib)

add_executable(pvad_demo
    main.cpp
    mainwindow.h mainwindow.cpp
    engine.h engine.cpp
    demo_core.h demo_core.cpp
    wizard.h wizard.cpp
    tts.h tts.cpp
    autotest.h autotest.cpp
)
target_include_directories(pvad_demo PRIVATE
    ${PROJ_ROOT}/src
    ${PROJ_ROOT}/third_party
    ${SHERPA_DIR}/include
)
# 注意: onnxruntime.dll 用 sherpa 包自带版本（同名按名导入，ORT C API 向后兼容）
target_link_libraries(pvad_demo PRIVATE Qt6::Widgets pipe ${SHERPA_DIR}/lib/sherpa-onnx-c-api.lib)
target_compile_definitions(pvad_demo PRIVATE
    DEMO_ROOT="${PROJ_ROOT}"
    TTS_MODEL_DIR="${TTS_MODEL_DIR}"
)

# DLL 拷贝 + windeployqt
add_custom_command(TARGET pvad_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHERPA_DIR}/lib/sherpa-onnx-c-api.dll
        ${SHERPA_DIR}/lib/sherpa-onnx-cxx-api.dll
        ${SHERPA_DIR}/lib/onnxruntime.dll
        ${SHERPA_DIR}/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:pvad_demo>
    COMMAND $<TARGET_FILE:Qt6::windeployqt> --release --no-translations $<TARGET_FILE:pvad_demo>
)
