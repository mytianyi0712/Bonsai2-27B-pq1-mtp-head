// mtp-trace — generate self-distillation traces for MTP head training.
//
// Why: the draft head is only as good as its match to the DEPLOYMENT distribution.
// The corpus dump (mtp-dump) is prefill-only teacher forcing over raw session text,
// which contains no assistant reasoning (thinking) register and used f16 KV.  This
// tool runs the same model the server runs, with the same chat template, thinking
// enabled and the same KV quantization, generates text, and dumps the trunk's
// post-final-norm hidden (t_h_nextn) together with the token at every generated
// position — the exact (token, h) rows the MTP head consumes at inference.
//
// Output layout matches mtp-dump so train_mtp.py reads it unchanged:
//   DIR/chunks/h_%07d.f16    n_tok x n_embd fp16
//   DIR/chunks/tok_%07d.i32  n_tok int32
//   DIR/manifest.jsonl       {"h":..,"tok":..,"n":..,"src":..}
// Chunks never span two prompts (the training windows must stay inside one context).
//
// usage:
//   mtp-trace -m model.gguf --prompts FILE --out DIR
//             [--system TEXT] [--template-file FILE.jinja]
//             [--max-new 256] [--chunk 512] [--max-rows N]
//             [--ctk q4_0] [--ctv q4_0] [--ngl 99] [--threads 8]
//             [--temp 0] [--seed 1234] [--no-thinking]

#include "llama.h"
#include "llama-ext.h"
#include "common.h"
#include "chat.h"
#include "ggml.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static ggml_type parse_kv_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, "unknown kv type: %s\n", s.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, prompts_path, out_dir, system_text, template_file;
    int  max_new = 256, chunk = 512, ngl = 99, threads = 8, seed = 1234;
    long long max_rows = 0;
    float temp = 0.0f;
    std::string ctk = "f16", ctv = "f16";
    bool thinking = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return std::string(argv[++i]); };
        if      (a == "-m" || a == "--model")     model_path    = next();
        else if (a == "--prompts")                prompts_path  = next();
        else if (a == "--out")                    out_dir       = next();
        else if (a == "--system")                 system_text   = next();
        else if (a == "--template-file")          template_file = next();
        else if (a == "--max-new")                max_new       = std::stoi(next());
        else if (a == "--chunk")                  chunk         = std::stoi(next());
        else if (a == "--max-rows")               max_rows      = std::stoll(next());
        else if (a == "--ctk")                    ctk           = next();
        else if (a == "--ctv")                    ctv           = next();
        else if (a == "--ngl")                    ngl           = std::stoi(next());
        else if (a == "--threads")                threads       = std::stoi(next());
        else if (a == "--temp")                   temp          = std::stof(next());
        else if (a == "--seed")                   seed          = std::stoi(next());
        else if (a == "--no-thinking")            thinking      = false;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || prompts_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: mtp-trace -m model.gguf --prompts FILE --out DIR [options]\n");
        return 1;
    }

    std::vector<std::string> prompts;
    {
        std::ifstream f(prompts_path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(line);
        }
    }
    if (prompts.empty()) { fprintf(stderr, "no prompts in %s\n", prompts_path.c_str()); return 1; }
    fprintf(stderr, "[trace] %zu prompts, max_new=%d, chunk=%d, kv=%s/%s, thinking=%d\n",
            prompts.size(), max_new, chunk, ctk.c_str(), ctv.c_str(), (int) thinking);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.load_mtp     = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    std::string tmpl_src = template_file;
    if (!tmpl_src.empty()) {
        std::ifstream f(template_file);
        if (!f) { fprintf(stderr, "cannot read template %s\n", template_file.c_str()); return 1; }
        std::stringstream ss; ss << f.rdbuf();
        tmpl_src = ss.str();
    }
    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(model, tmpl_src);
    } catch (const std::exception & e) {
        fprintf(stderr, "chat template error: %s\n", e.what());
        return 1;
    }

    const int64_t n_embd = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = max_new + 8192;
    cp.n_batch         = 4096;
    cp.n_ubatch        = 512;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = parse_kv_type(ctk);
    cp.type_v          = parse_kv_type(ctv);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    llama_memory_t mem = llama_get_memory(ctx);

    fs::create_directories(fs::path(out_dir) / "chunks");
    std::ofstream manifest((fs::path(out_dir) / "manifest.jsonl").string());

    std::vector<uint16_t> hbuf;   // accumulated rows (fp16)
    std::vector<int32_t>  tbuf;
    int64_t chunk_id = 0, total = 0;
    const auto t0 = std::chrono::steady_clock::now();

    auto flush_chunk = [&](const std::string & src) {
        if (tbuf.empty()) return;
        char hn[64], tn[64];
        snprintf(hn, sizeof hn, "chunks/h_%07lld.f16",   (long long) chunk_id);
        snprintf(tn, sizeof tn, "chunks/tok_%07lld.i32", (long long) chunk_id);
        {
            std::ofstream fh((fs::path(out_dir) / hn).string(), std::ios::binary);
            fh.write((const char *) hbuf.data(), (std::streamsize) hbuf.size() * 2);
        }
        {
            std::ofstream ft((fs::path(out_dir) / tn).string(), std::ios::binary);
            ft.write((const char *) tbuf.data(), (std::streamsize) tbuf.size() * 4);
        }
        manifest << "{\"h\":\"" << hn << "\",\"tok\":\"" << tn << "\",\"n\":" << tbuf.size()
                 << ",\"src\":\"" << src << "\"}\n";
        manifest.flush();
        fprintf(stderr, "[trace] chunk %lld: %zu rows (%lld total)\n",
                (long long) chunk_id, tbuf.size(), (long long) total);
        ++chunk_id;
        hbuf.clear();
        tbuf.clear();
    };

    llama_batch batch = llama_batch_init(4096, 0, 1);
    std::vector<llama_token> toks;

    for (size_t pi = 0; pi < prompts.size() && (max_rows == 0 || total < max_rows); ++pi) {
        // ---- format the prompt exactly like the server does ----
        common_chat_templates_inputs in;
        if (!system_text.empty()) {
            common_chat_msg m; m.role = "system"; m.content = system_text; in.messages.push_back(m);
        }
        {
            common_chat_msg m; m.role = "user"; m.content = prompts[pi]; in.messages.push_back(m);
        }
        in.add_generation_prompt = true;
        in.use_jinja             = true;
        in.enable_thinking       = thinking;
        const std::string text = common_chat_templates_apply(tmpls.get(), in).prompt;

        int32_t n = -llama_tokenize(llama_model_get_vocab(model), text.data(), (int32_t) text.size(),
                                    nullptr, 0, true, true);
        toks.resize((size_t) n);
        llama_tokenize(llama_model_get_vocab(model), text.data(), (int32_t) text.size(),
                       toks.data(), n, true, true);

        llama_memory_clear(mem, true);
        // prefill
        int32_t pos = 0;
        while (pos < (int32_t) toks.size()) {
            const int32_t n_chunk = std::min(4096, (int32_t) toks.size() - pos);
            for (int32_t i = 0; i < n_chunk; ++i) {
                batch.token[i]     = toks[pos + i];
                batch.pos[i]       = pos + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (i == n_chunk - 1);
            }
            batch.n_tokens = n_chunk;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[trace] prefill failed\n"); return 1; }
            pos += n_chunk;
        }

        // ---- generate ----
        int32_t n_gen = 0;
        while (n_gen < max_new && (max_rows == 0 || total + tbuf.size() / n_embd < max_rows)) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { fprintf(stderr, "[trace] no logits\n"); return 1; }
            llama_token id;
            if (temp <= 0.0f) {
                int32_t best = 0;
                for (int32_t v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
                id = best;
            } else {
                std::vector<llama_token_data> cand(n_vocab);
                for (int32_t v = 0; v < n_vocab; ++v) cand[v] = { v, logits[v], 0.0f };
                llama_token_data_array arr { cand.data(), cand.size(), false };
                llama_sampler * smp = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(smp, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(smp, llama_sampler_init_dist((uint32_t) (seed + pi * 1000 + n_gen)));
                id = llama_sampler_sample(smp, ctx, -1);
                llama_sampler_free(smp);
                (void) arr;
            }

            // decode the sampled token at its position; its t_h_nextn row is the (token, h) pair
            batch.token[0]     = id;
            batch.pos[0]       = pos;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = true;
            batch.n_tokens     = 1;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[trace] decode failed\n"); return 1; }
            ++pos;
            ++n_gen;

            const float * h = llama_get_embeddings_nextn(ctx);
            if (!h) { fprintf(stderr, "[trace] no nextn embeddings\n"); return 1; }
            for (int64_t k = 0; k < n_embd; ++k) hbuf.push_back(ggml_fp32_to_fp16(h[k]));
            tbuf.push_back((int32_t) id);
            ++total;

            if ((int32_t) (tbuf.size()) >= chunk) flush_chunk("p" + std::to_string(pi));
            if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) break;
        }
        flush_chunk("p" + std::to_string(pi));   // never let a chunk span two prompts

        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[trace] prompt %zu/%zu: %d new tokens | %lld total | %.0f tok/s\n",
                pi + 1, prompts.size(), n_gen, (long long) total, total / std::max(0.1, el));
    }

    fprintf(stderr, "[trace] DONE: %lld rows in %lld chunks\n", (long long) total, (long long) chunk_id);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
