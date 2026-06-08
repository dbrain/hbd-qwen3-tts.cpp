#pragma once
// Minimal .npy reader/writer for port-validation harnesses. Supports v1.0/v2.0
// headers and the dtypes we actually emit/consume: <f4, <f8, <i4, <i8.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace npy {

struct Array {
    std::string descr;          // e.g. "<f4"
    std::vector<int64_t> shape;
    std::vector<uint8_t> raw;   // raw element bytes

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return shape.empty() ? 0 : n;
    }

    std::vector<float> as_f32() const {
        int64_t n = numel();
        std::vector<float> out(n);
        if (descr == "<f4") {
            std::memcpy(out.data(), raw.data(), n * sizeof(float));
        } else if (descr == "<f8") {
            const double * d = reinterpret_cast<const double *>(raw.data());
            for (int64_t i = 0; i < n; ++i) out[i] = (float)d[i];
        } else {
            throw std::runtime_error("as_f32: unsupported descr " + descr);
        }
        return out;
    }

    std::vector<int32_t> as_i32() const {
        int64_t n = numel();
        std::vector<int32_t> out(n);
        if (descr == "<i8") {
            const int64_t * d = reinterpret_cast<const int64_t *>(raw.data());
            for (int64_t i = 0; i < n; ++i) out[i] = (int32_t)d[i];
        } else if (descr == "<i4") {
            std::memcpy(out.data(), raw.data(), n * sizeof(int32_t));
        } else {
            throw std::runtime_error("as_i32: unsupported descr " + descr);
        }
        return out;
    }
};

inline Array load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    uint8_t magic[8];
    if (fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        fclose(f); throw std::runtime_error("npy: bad magic " + path);
    }
    uint32_t hlen = 0; size_t hstart = 0;
    if (magic[6] == 1) {
        uint16_t h; if (fread(&h, 1, 2, f) != 2) { fclose(f); throw std::runtime_error("npy hdr"); }
        hlen = h; hstart = 10;
    } else {
        uint32_t h; if (fread(&h, 1, 4, f) != 4) { fclose(f); throw std::runtime_error("npy hdr"); }
        hlen = h; hstart = 12;
    }
    std::string header(hlen, '\0');
    if (fread(&header[0], 1, hlen, f) != hlen) { fclose(f); throw std::runtime_error("npy hdr body"); }

    Array a;
    // descr
    {
        auto p = header.find("'descr'");
        auto c = header.find('\'', header.find(':', p) + 1);
        auto e = header.find('\'', c + 1);
        a.descr = header.substr(c + 1, e - c - 1);
    }
    // shape
    {
        auto p = header.find("'shape'");
        auto lp = header.find('(', p);
        auto rp = header.find(')', lp);
        std::string s = header.substr(lp + 1, rp - lp - 1);
        std::string num;
        for (char ch : s) {
            if (ch == ',' || ch == ' ') {
                if (!num.empty()) { a.shape.push_back(std::stoll(num)); num.clear(); }
            } else num.push_back(ch);
        }
        if (!num.empty()) a.shape.push_back(std::stoll(num));
    }
    // element size
    int esz = 4;
    if (a.descr.size() >= 3) esz = a.descr[2] - '0';
    int64_t n = a.numel();
    a.raw.resize((size_t)n * esz);
    if (fread(a.raw.data(), 1, a.raw.size(), f) != a.raw.size()) {
        fclose(f); throw std::runtime_error("npy: short read " + path);
    }
    (void)hstart;
    fclose(f);
    return a;
}

inline void save_f32(const std::string & path, const std::vector<float> & data,
                     const std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("npy: cannot write " + path);
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        sh += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) sh += ",";
        if (i + 1 < shape.size()) sh += " ";
    }
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');
    uint8_t magic[8] = { 0x93, 'N','U','M','P','Y', 1, 0 };
    uint16_t hlen = (uint16_t)hdr.size();
    fwrite(magic, 1, 8, f);
    fwrite(&hlen, 1, 2, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
}

} // namespace npy
