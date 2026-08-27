cmake_minimum_required(VERSION 3.20)
project(double_voice C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(ORT_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
set(SPEEX_DIR ${CMAKE_SOURCE_DIR}/third_party/speexdsp)

# --- SpeexDSP (AEC) built from source, no autotools ---
add_library(speexdsp STATIC
    ${SPEEX_DIR}/libspeexdsp/mdf.c
    ${SPEEX_DIR}/libspeexdsp/fftwrap.c
    ${SPEEX_DIR}/libspeexdsp/kiss_fft.c
    ${SPEEX_DIR}/libspeexdsp/kiss_fftr.c
    ${SPEEX_DIR}/libspeexdsp/smallft.c
    ${SPEEX_DIR}/libspeexdsp/filterbank.c
    ${SPEEX_DIR}/libspeexdsp/preprocess.c
    ${SPEEX_DIR}/libspeexdsp/buffer.c
    ${SPEEX_DIR}/libspeexdsp/scal.c
    ${SPEEX_DIR}/libspeexdsp/resample.c
)
target_include_directories(speexdsp PUBLIC ${SPEEX_DIR}/include PRIVATE ${SPEEX_DIR}/libspeexdsp)
target_compile_definitions(speexdsp PUBLIC USE_KISS_FFT FLOATING_POINT EXPORT=)

# --- RNNoise (vendor 源码, 不跑其构建系统; master 已自带 rnn_fft_* 符号前缀, 与 speexdsp 无冲突) ---
set(RNNOISE_DIR ${CMAKE_SOURCE_DIR}/third_party/rnnoise)
add_library(rnnoise STATIC
    ${RNNOISE_DIR}/src/denoise.c
    ${RNNOISE_DIR}/src/rnn.c
    ${RNNOISE_DIR}/src/rnn_data.c
    ${RNNOISE_DIR}/src/kiss_fft.c
    ${RNNOISE_DIR}/src/pitch.c
    ${RNNOISE_DIR}/src/celt_lpc.c
    ${RNNOISE_DIR}/src/rnn_reader.c
)
target_include_directories(rnnoise PUBLIC ${RNNOISE_DIR}/include PRIVATE ${RNNOISE_DIR}/src)
set_target_properties(rnnoise PROPERTIES C_STANDARD 99)
target_compile_definitions(rnnoise PRIVATE _USE_MATH_DEFINES)  # MSVC 的 math.h 需要才有 M_PI（MinGW 下无害）

# --- core library: AEC + VAD + fbank + speaker embedding ---
add_library(core STATIC
    src/aec.cpp
    src/vad.cpp
    src/fbank.cpp
    src/speaker.cpp
    src/pvad.cpp
    src/pvad_stream.cpp
    src/denoise.cpp
)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src ${ORT_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(core PUBLIC speexdsp rnnoise ${ORT_DIR}/lib/onnxruntime.dll)

add_executable(double_voice src/main.cpp)
target_link_libraries(double_voice PRIVATE core)

add_executable(enroll tools/enroll.cpp)
target_link_libraries(enroll PRIVATE core)

add_executable(score tools/score.cpp)
target_link_libraries(score PRIVATE core)

add_executable(probe tools/probe.cpp)
target_link_libraries(probe PRIVATE core)

# copy onnxruntime.dll next to the executables
foreach(t double_voice enroll score probe)
    add_custom_command(TARGET ${t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_DIR}/lib/onnxruntime.dll $<TARGET_FILE_DIR:${t}>)
endforeach()
