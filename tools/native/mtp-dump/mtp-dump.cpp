// mtp-dump — dump per-token post-final-norm trunk hidden states (h_nextn) + token ids
// for MTP head training (Ternary-Bonsai-2-27B / qwen35, PTQ1_0).
//
// The MTP head consumes MTP rows built from this dump (see common/speculative.cpp
// draft-mtp): the row at position q pairs token x_q with the trunk hidden of position
// q-1 (h rows are shifted right by one position) and predicts x_{q+1}.  So per chunk,
// tok[q] + h[q-1] at position q is the row, and tok[q+1] is its target.
//
// Usage:
//   mtp-dump -m model.gguf --list files.txt --out DIR [--chunk 2048] [--ub 1024]
//            [--max-tokens 10000000] [--ngl 99] [--threads 8]
//
// Output per chunk (global index):
//   DIR/chunks/h_%07d.f16   n_tok × n_embd fp16 rows (row i = position i)
//   DIR/chunks/tok_%07d.i32 n_tok × int32 token ids
//   DIR/manifest.jsonl      one record per chunk

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool read_text(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, list_path, out_dir;
    int     chunk  = 2048;
    int     ub     = 1024;
    int     ngl    = 99;
    int     threads = 8;
    std::string ctk = "f16", ctv = "f16";
    int64_t max_tokens = 10000000;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return std::string(argv[++i]); };
        if      (a == "-m")           model_path = next();
        else if (a == "--list")       list_path  = next();
        else if (a == "--out")        out_dir    = next();
        else if (a == "--chunk")      chunk      = std::stoi(next());
        else if (a == "--ub")         ub         = std::stoi(next());
        else if (a == "--max-tokens") max_tokens = std::stoll(next());
        else if (a == "--ngl")        ngl        = std::stoi(next());
        else if (a == "--threads")    threads    = std::stoi(next());
        else if (a == "--ctk")        ctk        = next();
        else if (a == "--ctv")        ctv        = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || list_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: mtp-dump -m model.gguf --list files.txt --out DIR "
                        "[--chunk 2048] [--ub 1024] [--max-tokens N] [--ngl 99] [--threads 8] [--ctk f16] [--ctv f16]\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = chunk + 8;
    cp.n_batch         = chunk;
    cp.n_ubatch        = std::min(ub, chunk);
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto kv_type = [](const std::string & t) {
        if (t == "f16")  return GGML_TYPE_F16;
        if (t == "f32")  return GGML_TYPE_F32;
        if (t == "q8_0") return GGML_TYPE_Q8_0;
        if (t == "q5_0") return GGML_TYPE_Q5_0;
        if (t == "q4_0") return GGML_TYPE_Q4_0;
        fprintf(stderr, "unknown kv type %s\n", t.c_str()); exit(1);
    };
    cp.type_k = kv_type(ctk);
    cp.type_v = kv_type(ctv);

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    llama_memory_t mem = llama_get_memory(ctx);

    fs::create_directories(fs::path(out_dir) / "chunks");
    std::ofstream manifest((fs::path(out_dir) / "manifest.jsonl").string());

    std::vector<std::string> files;
    {
        std::ifstream f(list_path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) files.push_back(line);
        }
    }
    fprintf(stderr, "[dump] %zu source files, chunk=%d ub=%d max_tokens=%lld\n",
            files.size(), chunk, cp.n_ubatch, (long long) max_tokens);

    llama_batch batch = llama_batch_init(chunk, 0, 1);
    std::vector<llama_token> toks;
    std::vector<uint16_t>    h16((size_t) n_embd);

    int64_t total = 0, chunk_id = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (const auto & path : files) {
        if (total >= max_tokens) break;
        std::string text;
        if (!read_text(path, text)) { fprintf(stderr, "[dump] skip (read fail): %s\n", path.c_str()); continue; }

        int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, true);
        if (n <= 0) continue;
        toks.resize((size_t) n);
        int32_t n2 = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), n, false, true);
        if (n2 <= 0) continue;
        toks.resize((size_t) n2);

        for (int64_t s = 0; s < (int64_t) toks.size(); s += chunk) {
            if (total >= max_tokens) break;
            const int64_t n_tok = std::min<int64_t>(chunk, (int64_t) toks.size() - s);
            if (n_tok < 32) break;

            llama_memory_clear(mem, true);
            for (int64_t i = 0; i < n_tok; ++i) {
                batch.token[i]     = toks[(size_t) (s + i)];
                batch.pos[i]       = (llama_pos) i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = false;
            }
            batch.n_tokens = (int32_t) n_tok;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[dump] decode failed at chunk %lld\n", (long long) chunk_id);
                return 1;
            }
            const float * h = llama_get_embeddings_nextn(ctx);
            if (!h) { fprintf(stderr, "[dump] no nextn embeddings\n"); return 1; }

            char hn[64], tn[64];
            snprintf(hn, sizeof hn, "chunks/h_%07lld.f16",   (long long) chunk_id);
            snprintf(tn, sizeof tn, "chunks/tok_%07lld.i32", (long long) chunk_id);

            {
                std::ofstream fh((fs::path(out_dir) / hn).string(), std::ios::binary);
                for (int64_t r = 0; r < n_tok; ++r) {
                    const float * row = h + r * n_embd;
                    for (int64_t k = 0; k < n_embd; ++k) h16[(size_t) k] = ggml_fp32_to_fp16(row[k]);
                    fh.write((const char *) h16.data(), (std::streamsize) n_embd * 2);
                }
            }
            {
                std::ofstream ft((fs::path(out_dir) / tn).string(), std::ios::binary);
                ft.write((const char *) (toks.data() + s), (std::streamsize) n_tok * sizeof(llama_token));
            }

            manifest << "{\"h\":\"" << hn << "\",\"tok\":\"" << tn << "\",\"n\":" << n_tok
                     << ",\"src\":\"" << fs::path(path).filename().string() << "\"}\n";
            manifest.flush();

            total += n_tok;
            ++chunk_id;
            if (chunk_id % 20 == 0) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double rate = el > 0 ? total / el : 0.0;
                double eta  = rate > 0 ? (max_tokens - total) / rate : 0.0;
                fprintf(stderr, "[dump] %lld chunks | %lld/%lld tokens | %.0f tok/s | ETA %.1f min\n",
                        (long long) chunk_id, (long long) total, (long long) max_tokens, rate, eta / 60.0);
            }
        }
    }

    fprintf(stderr, "[dump] DONE: %lld tokens in %lld chunks\n", (long long) total, (long long) chunk_id);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
