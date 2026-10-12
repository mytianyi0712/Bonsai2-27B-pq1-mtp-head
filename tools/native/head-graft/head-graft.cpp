// head-graft — replace the trained MTP head tensors (blk.64.*) in a qwen35 model GGUF.
//
// The head is trained in PyTorch (mtp-train/) and exported as one raw little-endian
// fp16 file per tensor, laid out exactly as ggml stores it (ne[0] fastest, i.e. the
// [out, in] matrix with `in` contiguous).  This tool streams a copy of the source
// model and patches each blk.64.* tensor in place with the trained weights, quantized
// to the type the original file already used (Q6_K / Q8_0 / F32), so every other byte
// of the 7 GB model is untouched and the tensor offsets stay valid.
//
// usage: head-graft --src model.gguf --out model-new.gguf --weights DIR
//
// For every tensor in DIR it also reports the quantization error (max |dequant(w) - w|)
// so a bad or mismatched export is visible immediately.

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    f.read((char *) out.data(), n);
    return (bool) f;
}

int main(int argc, char ** argv) {
    std::string src, dst, weights_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return std::string(argv[++i]); };
        if      (a == "--src")     src         = next();
        else if (a == "--out")     dst         = next();
        else if (a == "--weights") weights_dir = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (src.empty() || dst.empty() || weights_dir.empty()) {
        fprintf(stderr, "usage: head-graft --src model.gguf --out model-new.gguf --weights DIR\n");
        return 1;
    }

    // ---- read the source tensor table (metadata only, no data allocation) ----
    ggml_context * ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(src.c_str(), gp);
    if (!g) {
        fprintf(stderr, "failed to read %s\n", src.c_str());
        return 1;
    }
    const int64_t data_off = gguf_get_data_offset(g);
    const int64_t n_tensors = gguf_get_n_tensors(g);

    struct patch_t {
        std::string name;
        int64_t     file_off;      // absolute offset of the tensor data in the file
        size_t      size;          // bytes (unchanged: same quantization type)
        ggml_type   type;
        int64_t     n_per_row;
        int64_t     nrows;
        std::vector<uint8_t> data;
    };
    std::vector<patch_t> patches;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        if (strncmp(name, "blk.64.", 7) != 0) {
            continue;
        }
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            fprintf(stderr, "no tensor metadata for %s\n", name);
            return 1;
        }
        patch_t p;
        p.name      = name;
        p.file_off  = data_off + gguf_get_tensor_offset(g, i);
        p.size      = gguf_get_tensor_size(g, i);
        p.type      = t->type;
        p.n_per_row = t->ne[0];
        p.nrows     = ggml_nrows(t);
        patches.push_back(std::move(p));
    }
    if (patches.empty()) {
        fprintf(stderr, "no blk.64.* tensors in %s\n", src.c_str());
        return 1;
    }
    fprintf(stderr, "[graft] %zu MTP tensors, data offset %" PRId64 "\n", patches.size(), data_off);

    // ---- quantize the exported fp16 weights into each tensor's own type ----
    size_t total = 0;
    for (auto & p : patches) {
        const fs::path f16 = fs::path(weights_dir) / (p.name + ".f16");
        std::vector<uint8_t> raw;
        if (!read_file(f16.string(), raw)) {
            fprintf(stderr, "missing export: %s\n", f16.string().c_str());
            return 1;
        }
        const size_t n_elem = (size_t) p.n_per_row * (size_t) p.nrows;
        if (raw.size() != n_elem * 2) {
            fprintf(stderr, "%s: %zu bytes, expected %zu (fp16, %" PRId64 "x%" PRId64 ")\n",
                    f16.string().c_str(), raw.size(), n_elem * 2, p.n_per_row, p.nrows);
            return 1;
        }
        const ggml_fp16_t * h = (const ggml_fp16_t *) raw.data();
        std::vector<float> f32(n_elem);
        for (size_t k = 0; k < n_elem; ++k) {
            f32[k] = ggml_fp16_to_fp32(h[k]);
        }

        if (p.type == GGML_TYPE_F32) {
            p.data.assign((uint8_t *) f32.data(), (uint8_t *) f32.data() + n_elem * sizeof(float));
        } else {
            p.data.resize(p.size);
            const size_t written = ggml_quantize_chunk(p.type, f32.data(), p.data.data(),
                                                       0, p.nrows, p.n_per_row, nullptr);
            if (written != p.size) {
                fprintf(stderr, "%s: quantized %zu bytes, tensor is %zu (%s)\n",
                        p.name.c_str(), written, p.size, ggml_type_name(p.type));
                return 1;
            }
        }

        // quantization error report
        std::vector<float> back(n_elem);
        if (p.type == GGML_TYPE_F32) {
            memcpy(back.data(), p.data.data(), n_elem * sizeof(float));
        } else {
            ggml_get_type_traits(p.type)->to_float(p.data.data(), back.data(), (int64_t) n_elem);
        }
        float max_err = 0.0f;
        float max_abs = 0.0f;
        for (size_t k = 0; k < n_elem; ++k) {
            max_err = std::max(max_err, fabsf(back[k] - f32[k]));
            max_abs = std::max(max_abs, fabsf(f32[k]));
        }
        total += p.size;
        fprintf(stderr, "[graft] %-42s %-6s %8.2f MB  max|dW| %.5f (max|W| %.3f)\n",
                p.name.c_str(), ggml_type_name(p.type), p.size / 1e6, max_err, max_abs);
    }

    // ---- stream-copy the model, then patch the head tensors ----
    {
        std::ifstream in(src, std::ios::binary);
        std::ofstream out(dst, std::ios::binary | std::ios::trunc);
        if (!in || !out) {
            fprintf(stderr, "cannot copy %s -> %s\n", src.c_str(), dst.c_str());
            return 1;
        }
        std::vector<char> buf(1 << 22);
        while (in) {
            in.read(buf.data(), (std::streamsize) buf.size());
            out.write(buf.data(), in.gcount());
        }
        in.close();
        for (const auto & p : patches) {
            out.seekp((std::streamoff) p.file_off);
            out.write((const char *) p.data.data(), (std::streamsize) p.data.size());
            if (!out) {
                fprintf(stderr, "write failed at %s (offset %" PRId64 ")\n", p.name.c_str(), p.file_off);
                return 1;
            }
        }
        out.close();
    }

    fprintf(stderr, "[graft] wrote %s (%zu tensors patched, %.1f MB)\n", dst.c_str(), patches.size(), total / 1e6);
    gguf_free(g);
    ggml_free(ctx);
    return 0;
}
