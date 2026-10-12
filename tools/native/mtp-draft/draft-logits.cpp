// draft-logits — reference logits from the qwen35 MTP draft graph (DECODER_MTP ctx),
// for numerical validation of the PyTorch MTP head reimplementation.
//
// Steps:
//   1) target ctx (default): decode prompt tokens with embeddings_nextn unmasked -> h rows
//      (post-final-norm trunk hidden, same tensor the MTP draft consumes)
//   2) draft ctx (ctx_type = LLAMA_CONTEXT_TYPE_MTP):
//        rows at positions q = 1..L-1, row q = token x_q + trunk h_{q-1}
//        (real draft-flow pairing, see common/speculative.cpp draft-mtp: the target's
//         h rows are shifted right by one position; Q and K are roped at inp_pos)
//      -> logits of the last row (position L-1) = draft prediction for x_L
//
// Outputs into --out DIR:
//   h.f16     L x n_embd fp16   (the exact h rows fed to the draft graph)
//   tok.i32   L x int32         (prompt token ids)
//   logits.f32  n_vocab float   (draft logits for the next token)
//
// usage: draft-logits -m model.gguf --text "..." --out DIR [--ngl 99] [--threads 8]
//        draft-logits -m model.gguf --tokens 1,2,3 --out DIR

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
    std::string model_path, text, tokens_csv, out_dir;
    int ngl = 99, threads = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return std::string(argv[++i]); };
        if      (a == "-m")         model_path = next();
        else if (a == "--text")     text       = next();
        else if (a == "--tokens")   tokens_csv = next();
        else if (a == "--out")      out_dir    = next();
        else if (a == "--ngl")      ngl        = std::stoi(next());
        else if (a == "--threads")  threads    = std::stoi(next());
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || out_dir.empty() || (text.empty() && tokens_csv.empty())) {
        fprintf(stderr, "usage: draft-logits -m model.gguf (--text S | --tokens a,b,c) --out DIR\n");
        return 1;
    }
    fs::create_directories(out_dir);            // hnextn.f16 is written before the outputs section

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.load_mtp     = true;                    // blk.64 nextn tensors required by the draft graph

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t n_embd = llama_model_n_embd(model);

    // ---- tokenize ----
    std::vector<llama_token> toks;
    if (!tokens_csv.empty()) {
        size_t p = 0;
        while (p < tokens_csv.size()) {
            size_t q = tokens_csv.find(',', p);
            if (q == std::string::npos) q = tokens_csv.size();
            toks.push_back((llama_token) std::stoi(tokens_csv.substr(p, q - p)));
            p = q + 1;
        }
    } else {
        int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, true);
        toks.resize((size_t) n);
        llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), n, false, true);
    }
    const int32_t L = (int32_t) toks.size();
    if (L < 3) { fprintf(stderr, "need at least 3 tokens (2 MTP rows)\n"); return 1; }
    fprintf(stderr, "[draft] %d tokens\n", L);

    // ---- 1) target ctx: extract h rows ----
    std::vector<float> h((size_t) L * n_embd);
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = L + 8;
        cp.n_batch         = L + 8;
        cp.n_ubatch        = L + 8;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "target ctx failed\n"); return 1; }
        llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

        llama_batch b = llama_batch_init(L, 0, 1);
        for (int32_t i = 0; i < L; ++i) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = false;
        }
        b.n_tokens = L;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "target decode failed\n"); return 1; }
        const float * hp = llama_get_embeddings_nextn(ctx);
        if (!hp) { fprintf(stderr, "no nextn embeddings\n"); return 1; }
        std::memcpy(h.data(), hp, (size_t) L * n_embd * sizeof(float));
        llama_batch_free(b);
        llama_free(ctx);
    }

    // ---- 2) draft ctx: MTP graph ----
    std::vector<float> logits;
    {
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx           = L + 16;
        cp.n_batch         = L + 16;
        cp.n_ubatch        = L + 16;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * dctx = llama_init_from_model(model, cp);
        if (!dctx) { fprintf(stderr, "draft ctx failed\n"); return 1; }
        llama_set_embeddings_nextn(dctx, true, /*masked*/ false);   // dump the MTP block output (h_nextn)

        const int32_t nb = L - 1;                                  // MTP rows at pos 1..L-1
        llama_batch b = llama_batch_init(nb, 0, 1);
        std::vector<float> hnext((size_t) nb * n_embd);
        for (int32_t i = 0; i < nb; ++i) {
            const int32_t q = i + 1;                               // row position
            b.token[i] = toks[q]; b.pos[i] = q; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = false;
            std::memcpy(hnext.data() + (size_t) i * n_embd, h.data() + (size_t) (q - 1) * n_embd, n_embd * sizeof(float));
        }
        b.logits[nb - 1] = true;                                   // last row scores x_L
        b.n_tokens    = nb;
        b.embd_nextn  = hnext.data();

        if (llama_decode(dctx, b) != 0) { fprintf(stderr, "draft decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(dctx, -1);
        if (!lg) { fprintf(stderr, "no logits\n"); return 1; }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        logits.assign(lg, lg + n_vocab);
        // MTP block output (post shared_head_norm) for every row
        const float * hn = llama_get_embeddings_nextn(dctx);
        if (hn) {
            std::ofstream f((fs::path(out_dir) / "hnextn.f16").string(), std::ios::binary);
            std::vector<uint16_t> row((size_t) n_embd);
            for (int32_t i = 0; i < nb; ++i) {
                const float * r = hn + (size_t) i * n_embd;
                for (int64_t k = 0; k < n_embd; ++k) row[(size_t) k] = ggml_fp32_to_fp16(r[k]);
                f.write((const char *) row.data(), (std::streamsize) n_embd * 2);
            }
            fprintf(stderr, "[draft] hnextn rows: %d\n", nb);
        } else {
            fprintf(stderr, "[draft] warn: draft h_nextn unavailable\n");
        }
        llama_batch_free(b);
        llama_free(dctx);
    }

    // ---- outputs ----
    fs::create_directories(out_dir);
    {
        std::ofstream f((fs::path(out_dir) / "h.f16").string(), std::ios::binary);
        std::vector<uint16_t> row((size_t) n_embd);
        for (int32_t i = 0; i < L; ++i) {
            const float * r = h.data() + (size_t) i * n_embd;
            for (int64_t k = 0; k < n_embd; ++k) row[(size_t) k] = ggml_fp32_to_fp16(r[k]);
            f.write((const char *) row.data(), (std::streamsize) n_embd * 2);
        }
    }
    {
        std::ofstream f((fs::path(out_dir) / "tok.i32").string(), std::ios::binary);
        f.write((const char *) toks.data(), (std::streamsize) L * sizeof(llama_token));
    }
    {
        std::ofstream f((fs::path(out_dir) / "logits.f32").string(), std::ios::binary);
        f.write((const char *) logits.data(), (std::streamsize) logits.size() * sizeof(float));
    }
    fprintf(stderr, "[draft] wrote %s (L=%d, vocab=%zu)\n", out_dir.c_str(), L, logits.size());

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
