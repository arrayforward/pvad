// wav_io.h - 极简 WAV 读写（16bit PCM / 32bit float，单声道 16kHz）
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct WavData {
    int sample_rate = 0;
    std::vector<float> samples;  // [-1, 1]
};

inline uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

inline WavData read_wav(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open wav: " + path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    if (fread(buf.data(), 1, size, f) != (size_t)size) { fclose(f); throw std::runtime_error("read fail: " + path); }
    fclose(f);
    if (size < 44 || memcmp(buf.data(), "RIFF", 4) || memcmp(buf.data() + 8, "WAVE", 4))
        throw std::runtime_error("not a RIFF/WAVE file: " + path);

    uint16_t audio_fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    uint32_t data_size = 0;
    size_t pos = 12;
    while (pos + 8 <= (size_t)size) {
        uint32_t csize = rd_u32(&buf[pos + 4]);
        if (memcmp(&buf[pos], "fmt ", 4) == 0 && pos + 8 + csize <= (size_t)size) {
            audio_fmt = rd_u16(&buf[pos + 8]);
            channels = rd_u16(&buf[pos + 10]);
            rate = rd_u32(&buf[pos + 12]);
            bits = rd_u16(&buf[pos + 22]);
        } else if (memcmp(&buf[pos], "data", 4) == 0) {
            data = &buf[pos + 8];
            data_size = csize;
            if (pos + 8 + data_size > (size_t)size) data_size = (uint32_t)(size - pos - 8);
        }
        pos += 8 + csize + (csize & 1);
    }
    if (!data || audio_fmt == 0) throw std::runtime_error("no fmt/data chunk: " + path);
    if (rate != 16000) throw std::runtime_error("sample rate must be 16000, got " + std::to_string(rate) + ": " + path);

    WavData w;
    w.sample_rate = 16000;
    if (audio_fmt == 1 && bits == 16) {
        size_t n = data_size / 2 / channels;
        w.samples.resize(n);
        for (size_t i = 0; i < n; i++) {
            int16_t v = (int16_t)rd_u16(data + (i * channels) * 2);
            w.samples[i] = v / 32768.f;
        }
    } else if (audio_fmt == 3 && bits == 32) {
        size_t n = data_size / 4 / channels;
        w.samples.resize(n);
        for (size_t i = 0; i < n; i++) {
            float v;
            memcpy(&v, data + (i * channels) * 4, 4);
            w.samples[i] = v;
        }
    } else {
        throw std::runtime_error("only 16bit PCM or 32bit float wav supported: " + path);
    }
    return w;
}

inline void write_wav16(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write wav: " + path);
    uint32_t data_size = (uint32_t)samples.size() * 2;
    uint8_t hdr[44] = {0};
    memcpy(hdr, "RIFF", 4);
    uint32_t riff_size = 36 + data_size;
    hdr[4] = riff_size & 0xff; hdr[5] = (riff_size >> 8) & 0xff; hdr[6] = (riff_size >> 16) & 0xff; hdr[7] = (riff_size >> 24) & 0xff;
    memcpy(hdr + 8, "WAVEfmt ", 8);
    hdr[16] = 16; hdr[20] = 1; hdr[22] = 1;  // fmt size, PCM, mono
    hdr[24] = sample_rate & 0xff; hdr[25] = (sample_rate >> 8) & 0xff;
    uint32_t br = sample_rate * 2;
    hdr[28] = br & 0xff; hdr[29] = (br >> 8) & 0xff; hdr[30] = (br >> 16) & 0xff; hdr[31] = (br >> 24) & 0xff;
    hdr[32] = 2; hdr[34] = 16;
    memcpy(hdr + 36, "data", 4);
    hdr[40] = data_size & 0xff; hdr[41] = (data_size >> 8) & 0xff; hdr[42] = (data_size >> 16) & 0xff; hdr[43] = (data_size >> 24) & 0xff;
    fwrite(hdr, 1, 44, f);
    for (float s : samples) {
        if (s > 1.f) s = 1.f;
        if (s < -1.f) s = -1.f;
        int16_t v = (int16_t)(s * 32767.f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
}
