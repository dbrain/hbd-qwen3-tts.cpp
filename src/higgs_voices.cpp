#include "higgs_voices.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace higgs {

VoiceStore::VoiceStore(const std::string & dir) : dir_(dir) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
}

std::string VoiceStore::sanitize(const std::string & name) {
    std::string out;
    for (char c : name) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-') {
            out.push_back(c);
        } else if (!out.empty() && out.back() != '_') {
            out.push_back('_');
        }
    }
    while (!out.empty() && (out.back() == '_' || out.back() == '.')) out.pop_back();
    size_t s = 0; while (s < out.size() && (out[s] == '_' || out[s] == '.')) ++s;
    out = out.substr(s);
    if (out.empty()) out = "voice";
    return out;
}

std::string VoiceStore::npy_path(const std::string & id) const { return dir_ + "/" + id + ".npy"; }
std::string VoiceStore::txt_path(const std::string & id) const { return dir_ + "/" + id + ".reftext"; }
std::string VoiceStore::wav_path(const std::string & id) const { return dir_ + "/" + id + ".wav"; }

// ---- minimal float32 C-order .npy I/O (matches tools/npy.h save_f32) ----
static bool write_npy_f32(const std::string & path, const std::vector<float> & data,
                          int T, int N, std::string & err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { err = "open for write failed: " + path; return false; }
    std::string sh = "(" + std::to_string(T) + ", " + std::to_string(N) + ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    // total header (magic 8 + 2 len + hdr + pad) padded to 64.
    size_t base = 10 + hdr.size() + 1;            // +1 for trailing '\n'
    size_t pad = (64 - (base % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');
    uint8_t magic[8] = { 0x93, 'N','U','M','P','Y', 1, 0 };
    uint16_t hlen = (uint16_t)hdr.size();
    f.write((const char*)magic, 8);
    f.write((const char*)&hlen, 2);
    f.write(hdr.data(), hdr.size());
    f.write((const char*)data.data(), data.size() * sizeof(float));
    return (bool)f;
}

static bool read_npy_f32(const std::string & path, std::vector<float> & data,
                         int & T, int & N) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint8_t magic[8];
    f.read((char*)magic, 8);
    if (f.gcount() != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    size_t hlen = 0;
    if (magic[6] == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
    else { uint32_t h; f.read((char*)&h, 4); hlen = h; }
    std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);
    if (hdr.find("'<f4'") == std::string::npos) return false;
    // parse shape (a, b)
    size_t sp = hdr.find("'shape':");
    size_t lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    if (sp == std::string::npos || lp == std::string::npos || rp == std::string::npos) return false;
    std::string sh = hdr.substr(lp + 1, rp - lp - 1);
    int a = 0, b = 0; char comma = 0;
    if (sscanf(sh.c_str(), "%d %c %d", &a, &comma, &b) < 1) return false;
    if (b == 0) { b = 1; }   // 1-D fallback
    T = a; N = b;
    data.resize((size_t)T * N);
    f.read((char*)data.data(), data.size() * sizeof(float));
    return (size_t)f.gcount() == data.size() * sizeof(float);
}

bool VoiceStore::exists(const std::string & id) const {
    std::error_code ec; return fs::exists(npy_path(id), ec);
}

std::vector<VoiceInfo> VoiceStore::list() const {
    std::vector<VoiceInfo> out;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return out;
    for (const auto & e : fs::directory_iterator(dir_, ec)) {
        if (!e.is_regular_file()) continue;
        const auto p = e.path();
        if (p.extension() != ".npy") continue;
        VoiceInfo vi;
        vi.id = p.stem().string();
        std::vector<float> d;
        if (read_npy_f32(p.string(), d, vi.T, vi.N)) {
            vi.has_ref_text = fs::exists(txt_path(vi.id), ec);
            out.push_back(vi);
        }
    }
    std::sort(out.begin(), out.end(), [](const VoiceInfo & a, const VoiceInfo & b){ return a.id < b.id; });
    return out;
}

bool VoiceStore::save(const std::string & id, const int32_t * codes_TN, int T, int N,
                      const std::string & ref_text, std::string & err) {
    if (T <= 0 || N <= 0) { err = "empty codes"; return false; }
    std::vector<float> d((size_t)T * N);
    for (size_t i = 0; i < d.size(); ++i) {
        int v = codes_TN[i]; if (v < 0) v = 0; if (v > 1023) v = 1023;
        d[i] = (float)v;
    }
    if (!write_npy_f32(npy_path(id), d, T, N, err)) return false;
    if (!ref_text.empty()) { std::ofstream t(txt_path(id), std::ios::binary); t << ref_text; }
    return true;
}

bool VoiceStore::load(const std::string & id, std::vector<int32_t> & codes_TN,
                      int & T, int & N, std::string & ref_text) const {
    std::vector<float> d;
    if (!read_npy_f32(npy_path(id), d, T, N)) return false;
    codes_TN.resize(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        int v = (int)lrintf(d[i]); if (v < 0) v = 0; if (v > 1023) v = 1023;
        codes_TN[i] = v;
    }
    ref_text.clear();
    std::ifstream t(txt_path(id), std::ios::binary);
    if (t) { std::string s((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>()); ref_text = s; }
    return true;
}

bool VoiceStore::remove(const std::string & id, std::string & err) {
    std::error_code ec;
    bool any = fs::remove(npy_path(id), ec);
    fs::remove(txt_path(id), ec);
    fs::remove(wav_path(id), ec);
    if (!any) { err = "no such voice: " + id; return false; }
    return true;
}

bool VoiceStore::save_wav(const std::string & id, const std::string & wav_bytes, std::string & err) {
    std::ofstream f(wav_path(id), std::ios::binary);
    if (!f) { err = "open wav for write failed"; return false; }
    f.write(wav_bytes.data(), (std::streamsize)wav_bytes.size());
    if (!f) { err = "wav write failed"; return false; }
    return true;
}

bool VoiceStore::load_wav(const std::string & id, std::string & wav_bytes) const {
    std::ifstream f(wav_path(id), std::ios::binary);
    if (!f) return false;
    wav_bytes.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return !wav_bytes.empty();
}

bool VoiceStore::load_ref_text(const std::string & id, std::string & ref_text) const {
    std::ifstream t(txt_path(id), std::ios::binary);
    if (!t) return false;
    ref_text.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return true;
}

} // namespace higgs
