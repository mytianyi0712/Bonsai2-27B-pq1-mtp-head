# Bonsai2-27B — a distilled MTP draft head for the MTP-less official model

**English** | [中文](README.md)

`Ternary-Bonsai-2-27B-PTQ1_0` (architecture `qwen35`, PrismML PTQ1_0 ternary trunk, 256K
context) **ships without an MTP head** (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`: `block_count=64`,
851 tensors, no `blk.64.*` at all). What this repository provides is the **one head that adds
MTP capability**, self-distilled from "it runs" to "it's good":

1. **Add the structure** — append an MTP head (15 `blk.64.*` tensors) to the official model to
   get a draft-capable file: `block_count` 64 → 65, plus `qwen35.nextn_predict_layers=1`,
   **size +0.351 GB**. This step is a pure **append**; the file grows.
2. **Swap the weights** — replace those 15 tensors **in place** with the head trained in this
   repo (rewritten in each tensor's own type, Q6_K / Q8_0 / F32, every other byte untouched),
   giving this repo's `-mtp-distill2.gguf`. The swap itself changes **nothing** in size.

Only the **trained head weights** live in this repository — no model file is redistributed
(see [What is / is not in this repo](#what-is--is-not-in-this-repo)).

| File | Tensors | Size | `block_count` |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (official, **no MTP**) | 851 | 5,946,648,928 B (5.95 GB) | 64 |
| After adding the MTP structure (= this repo's graft source) | 866 | 6,297,658,656 B (6.30 GB) | 65 |
| After swapping in the trained head, `-mtp-distill2.gguf` | 866 | 6,297,658,656 B (same as previous row) | 65 |

The 0.351 GB adds up (cross-checked byte-for-byte against the HuggingFace original, 15/15):

| Component | Bytes |
|---|---|
| Raw data of the 15 `blk.64.*` tensors | 350,993,728 |
| 32-byte alignment padding | +15,040 |
| New tensor-table entries + `qwen35.nextn_predict_layers` metadata | +960 |
| **Total file growth** | **351,009,728** (≈0.351 GB) |

So essentially all of the 0.351 GB is the head weights themselves; the structural overhead is
only 16 KB.

* Official base model (**no MTP — add the head first**): `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
* Head: **424.7 M parameters** across 15 tensors, fp16 checkpoint → deployment QAT (Q6_K big matrices, Q8_0 k/v, F32 norms)
* Runtime: `llama.cpp` + KVMem fork with `--spec-type draft-mtp`

---

## Headline numbers

| Metric (all measured, see [Evidence](#evidence--log-excerpts)) | shipped head | **distill2 (this repo)** |
|---|---|---|
| Offline depth-1 trunk-argmax agreement | 66.9 % | **82.3 %** |
| Offline depth-2 | 51.3 % | **78.2 %** |
| Offline depth-3 | 44.7 % | **74.5 %** |
| Decode throughput, `n_max=1` | 40.3 t/s | 41.3 t/s |
| Decode throughput, `n_max=2` | — | **45.7 t/s** |
| Decode throughput, `n_max=3` | 41.0 t/s | 43.5 t/s |
| Mean accepted tokens per round τ, `n_max=2` | — | **2.02** |
| Prefill | not affected by the draft head | not affected |

Online A/B on 10 session prompts at `n_max=3` (weighted by draft events): **74.7 % → 77.7 %**
(+3.0 points), 7 wins / 2 losses / 1 tie — the harder the prompt, the larger the gain
(`+7.6`, `+6.8` points on the two longest/hardest ones); at `n_max=1` on the long prompt
**80.3 % → 88.2 %**. Greedy output was **byte-identical** in every A/B run.

---

## What the head is

Per `src/models/qwen35.cpp` (`graph_mtp`) and `common/speculative.cpp` (`draft-mtp`), one MTP
row at position `q` is built from the token `x_q` plus the trunk hidden state `h_{q-1}`
(**post final-norm**, the LM-head input), and it scores `x_{q+1}`:

```
tok_embd(x_q) → Hadamard inverse (rotate, then sign)   # prism.hadamard.inverse_weight_names
  → RMSNorm(enorm) ‖ RMSNorm(hnorm)                    # concat (e first, h second) 10240
  → eh_proj 10240→5120
  → full attention block: attn_norm → attn_q (12288 = interleaved [q(256)|gate(256)] per head)
      → per-head q_norm → gate = sigmoid(second half) → attn_k/attn_v (4×256)
      → per-head k_norm → RoPE (sections [11,11,10,0], query position = q)
      → attn × sigmoid(gate) → attn_output → +residual → post_attention_norm → SwiGLU FFN → +residual
  → RMSNorm(shared_head_norm) → shared output.weight → logits
```

Two non-obvious conventions the training code reproduces exactly (both were validated by
logits alignment against `llama.cpp`, per-row cosine ≥ 0.9999):

1. **GGML dimension order.** `ne[0]` varies fastest, so the numpy/torch shape is
   `reversed(ne)` and must **never** be transposed. A `reshape(ne).t()` produces correct
   values in a scrambled order (`cos ≈ 0`).
2. **Row semantics.** Rows are indexed by `(x_q, h_{q-1})`, RoPE position `q`, target `x_{q+1}`.
   At depth ≥ 2 the token is ground truth but the hidden state is the head's own previous
   depth output — i.e. inference self-feeds, so the training chain does too.

## How it was trained

Trunk-argmax distillation, fp16 native, one full attention block per step, chain depth 3:

| Item | Value |
|---|---|
| Objective | CE against the **frozen trunk's own argmax** (offline `head.lm_logits(h_dump)`), not against corpus text |
| Chain | `n=3` self-feeding chain (`forward_chain`), attention = window rows (pos ≤ anchor) + shallower rows of the same chain + self |
| Init | the **shipped** head (a trunk-faithful starting point) |
| Data | self-distillation traces generated with the real chat template + thinking + `q4_0` KV (**496,058 rows**, ×3 sampling weight) + `q4_0` corpus re-dump (**800,768 rows**) |
| Precision | fp16 weights/matmul, fp32 master weights + AdamW on CPU, QAT round-trip through the **deployment** types (Q6_K / Q8_0) |
| Cost | 9,996 steps / 113.3 min / peak VRAM 6.82 GiB / 1,507 tok/s |

Why distill the argmax instead of the text: acceptance is decided by
`draft argmax == trunk argmax`. The trunk agrees with the raw corpus text only ~66 % of the
time, so a head trained on text becomes *better at the text than the trunk is* and drifts
away from it. Measured: text-CE head 70.7 % acceptance vs distilled head 80.3 %
(same prompt, `n_max=1`).

Training curves (held-out chunks) from `logs/distill2.log`:

```
[train] step  500 VAL trunk-argmax depth1:79.2% | depth2:74.8% | depth3:72.6%
[train] step 5000 VAL trunk-argmax depth1:80.0% | depth2:76.5% | depth3:73.9%
[train] step 9500 VAL trunk-argmax depth1:80.3% | depth2:77.5% | depth3:75.0%
[train] FINAL VAL trunk-argmax agreement depth1:80.4% | depth2:77.5% | depth3:75.1%
[train] peak VRAM 6.82 GiB (reserved 8.06 GiB)
[train] done in 113.3 min -> D:\models\mtp-train\distill2\head_final.pt
```

## Evidence / log excerpts

Everything below is copied verbatim from the files in this repo.

**Final validation (this head)** — `logs/distill2.log`

```
[train] initialized from D:\models\mtp-train\distill1\head_final.pt
[train] trainable params: 424.7 M (fp16)
[train] steps/epoch≈4998, total≈9996, tokens≈10.2M, qat=deploy
[train] FINAL VAL trunk-argmax agreement depth1:80.4% | depth2:77.5% | depth3:75.1%
[train] peak VRAM 6.82 GiB (reserved 8.06 GiB)
[train] done in 113.3 min -> D:\models\mtp-train\distill2\head_final.pt
```

**Previous iteration (distill1, kept for reference)** — `logs/distill1.log`

```
[train] FINAL VAL trunk-argmax agreement depth1:80.0% | depth2:75.7% | depth3:74.0%
[train] done in 128.5 min -> D:\models\mtp-train\distill1\head_final.pt
```

**Raw acceptance traces** — `logs/ab-n*.log`. Each file is one `llama-kvmem-server` run with
`KVMEM_TRACE=1`; stop the log at the `spec_stats` line to read the acceptance of that run.
These particular logs were captured with the **shipped** head, so they document the harness
and the baseline distribution rather than the gain:

```
=== ab-n1.log,  n_max = 1  (shipped head) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=198 n_accept=102 n_restore=0 accept_pct=51.5
KVMEM_TRACE spec_stats n_gen=291 n_drafted=147 n_accept=145 n_restore=0 accept_pct=98.6

=== ab-n2.log,  n_max = 2  (shipped head) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=349 n_accept=125 n_restore=0 accept_pct=35.8

=== ab-n3.log,  n_max = 3  (shipped head) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=504 n_accept=131 n_restore=0 accept_pct=26.0
```

Acceptance is a property of the *content*, not just the head — prose and structured
(JSON/tool-call) tokens accept far worse than numeric sections (the same build shows >95 %
on digit runs). `accept_pct` is **per drafted token**, so values from different `n_max`
are not comparable; compare `t/s` across `n_max`.

**Graft verification.** The exported fp16 tensors are quantized to each tensor's deployment
type and patched into a byte copy of the head-bearing model; `draft-logits` re-runs the MTP graph on
the patched file and the same rows are replayed through the checkpoint in PyTorch
(`tools/verify_graft.py`):

| Check | Result |
|---|---|
| Block output, per-row cosine | min 0.9990, mean ≈0.9998 |
| Last-row logits cosine | 0.999951 |
| top-1 / top-10 | match / 10 of 10 |
| Greedy text in the acceptance A/B | byte-identical in every run |

*(Offline depth agreement from the 32+32 chunk / 16,384-row measurement; the 8-chunk
validation line printed inside the training log is the one quoted above.)*

## What is / is not in this repo

**In:**

```
head/distill2/head_final.pt   849 MB — 15 fp16 tensors (the head), tracked with Git LFS
logs/                         training logs + acceptance traces (verbatim)
tools/                        python pipeline (dump → trace → train → export → graft → verify)
tools/native/                 the four C++ tools that live in llama.cpp/tools/
scripts/                      .bat / .sh drivers used to run the pipeline on Windows
deploy/bonsai-chat-template.jinja
```

**Not in:** the official base model, the head-added / head-trained full models, and the
hidden-state dumps / self-distillation traces (hundreds of GB, fully regenerable — see the
commands below). The training corpus was built from local agent session transcripts and is
intentionally not published.

## Using the head

The official model has **no** MTP, so the full path is two steps: add the structure, then swap
the weights.

**Step 1 — give the official model an MTP structure (size +0.351 GB).** This step **appends**
15 `blk.64.*` tensors and updates the metadata; it is not an in-place edit:

```bat
rem append an MTP head -> a draft-capable model (the graft source for the rest of the toolchain)
python tools\graft_mtp.py  Ternary-Bonsai-2-27B-PTQ1_0.gguf  Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf
```

`tools/graft_mtp.py` fetches the stock Qwen MTP head over HTTP range requests (from the
`unsloth/Qwen3.8-27B-GGUF` repo on Hugging Face — step 1 therefore needs network access),
appends its 15 `blk.64.*` tensors to the official model, and sets `qwen35.block_count` 64 → 65
plus `qwen35.nextn_predict_layers=1`, then self-verifies (`tensors == base+15`,
`block_count == 65`, `blk.64 present 15/15`). The result is the **"shipped head" baseline**
used in the metric tables below; its weights are stock Qwen weights and are deliberately *not*
added to `prism.hadamard.weight_names`.

**Step 2 — replace those 15 tensors with the head trained here (zero size change).** This is
the in-place step:

```bat
rem 2a. export the fp16 tensors from the checkpoint
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export

rem 2b. replace in place in a byte copy of the head-bearing model + verify
graft-head.bat  head\distill2\head_final.pt  Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf
```

The output `-mtp-distill2.gguf` is exactly as long as **its input** (6,297,658,656 B). Note that
`graft-head.bat`'s `BASE` must be the **head-bearing** model; pointing it at the official
head-less file makes `head-graft` exit with `no blk.64.* tensors in ...` — by design, because
the official file has no tensors to replace; use step 1 to create them first.

`scripts/graft-head.bat` wraps step 2's three commands (`export_head.py` → `head-graft.exe` →
`draft-logits.exe` + `verify_graft.py`). Then serve it and read `spec_stats`:

```bat
scripts\ab-accept.bat  <patched.gguf>  <prompt.txt>  256  2
```

Measured sweet spot on the 8 GB test machine: **`n_max=2`** (45.7 t/s vs 40.3 t/s for the
shipped head at `n_max=1`, i.e. **+13.4 %**). `n_max=3` has the highest mean acceptance
length (τ 2.22) but is slower than `n_max=2` because each extra draft row costs a verify.

## Regenerating the pipeline

The native tools are additions to `llama.cpp/tools/`, so drop them into a `llama.cpp`
checkout (or the KVMem fork) and build:

```bat
cmake --build build-win --target draft-logits head-graft mtp-trace mtp-dump -j 8
```

Then, end to end:

```bat
rem 0. the official model has no MTP: add the structure first; the result feeds every later step
python tools\graft_mtp.py  Ternary-Bonsai-2-27B-PTQ1_0.gguf  Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf

rem 1. corpus: agent session transcripts -> text (>=100k chars each)
python tools\extract-sessions.py

rem 2. hidden states: (token, post-final-norm h) per position, with the deployment KV type
mtp-dump.exe -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --list filelist.txt --out dump --chunk 2048 --ub 1024 ^
             --max-tokens 8500000 --ngl 99 --threads 8 -ctk q4_0 -ctv q4_0

rem 3. self-distillation traces with the real template + thinking
mtp-trace.exe -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --prompts trace-prompts.txt --out traces ^
              --template-file bonsai-chat-template.jinja --max-new 700 --chunk 1024 ^
              --ctk q4_0 --ctv q4_0 --ngl 99 --threads 8
python tools\mtp-trace-merge.py --src traces --out traces-merged --min-rows 520

rem 4. train (trunk-argmax distillation, n=3 self-feeding chain, deployment QAT)
python tools\train_mtp.py --data dump --data traces-merged --data traces-merged --data traces-merged ^
       --out run --init-from shipped-head.pt --device cuda --seq 512 --batch 2 ^
       --epochs 2 --lr 3e-5 --accum 8 --val-every 500 --objective distill --steps 3 --qat deploy

rem 5. graft + verify + measure
python tools\export_head.py --ckpt run\head_final.pt --out head-export
head-graft.exe --src Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --out PATCHED.gguf --weights head-export
draft-logits.exe -m PATCHED.gguf --tokens <64 real token ids> --out verify-dir
python tools\verify_graft.py --val verify-dir --ckpt run\head_final.pt
python tools\mtp-bench.py --model PATCHED.gguf --n-max 2
```

`tools/mtp-retrain-v2.py` orchestrates the whole 4-stage chain (re-dump with `q4_0` →
generate traces → retrain → export/graft/verify) and is what produced this checkpoint. Every
step (dump / trace / the trainer's `--gguf` / the graft) takes the **head-bearing** model
produced in step 0 as input — the official head-less model is only ever fed to step 0.

## Rebuilding the head file from the checkpoint

```bat
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export
python -c "import torch; sd=torch.load(r'head\distill2\head_final.pt', map_location='cpu'); print(len(sd), sum(v.numel() for v in sd.values())/1e6, 'M params')"
```

Expected: `15 424.699392 M params`, all tensors fp16, mapped one-to-one onto
`blk.64.nextn.eh_proj.weight`, `blk.64.nextn.enorm.weight`, `blk.64.nextn.hnorm.weight`,
`blk.64.attn_norm.weight`, `blk.64.attn_q_norm.weight`, `blk.64.attn_k_norm.weight`,
`blk.64.post_attention_norm.weight`, `blk.64.attn_{q,k,v,output}.weight`,
`blk.64.ffn_{gate,up,down}.weight`, `blk.64.nextn.shared_head_norm.weight` (see
`tools/export_head.py` `NAME_MAP`). SHA-256 of the file in this repo:
`3c2788d6595c7f751c1cdc82657b31ffe3a67598d6eeb993239a5bb34f43ad54`.

## Known limits

* Short-context, `n_max=1` acceptance is already ~98–100 % with the shipped head — there is
  no headroom there. The gain lives in long context and harder distributions.
* Real (small) overfitting exists in the previous iteration (+1.7 / +3.4 / +4.3 points on
  depth 1/2/3); in this one the held-out gap is ≈0 (−0.3 / −0.2 / +0.8), consistent with
  training on the model's own output distribution.
* Sampling (`temp > 0`) acceptance is naturally lower than greedy — the validation above is
  greedy. A soft KL target would be the next lever for sampled decoding.
* Absolute `t/s` depends on the machine and on content; the relative ordering is the
  meaningful part.

## License

MIT — see [LICENSE](LICENSE). The base model and the KVMem `llama.cpp` fork are **not**
covered by this license; this repository only contains the trained head weights, the
pipeline code and the logs.

## Credits

* Head architecture and `draft-mtp` pairing semantics: `llama.cpp` (`src/models/qwen35.cpp`,
  `common/speculative.cpp`) and the KVMem fork (`llama-ext.h` `llama_set_embeddings_nextn`).
* Depth-decay weighting follows the FastMTP line of work (self-distilled data +
  weight-shared MTP head).
* Hadamard inverse for the MTP embedding lookup is the upstream patch in
  `tools/graft_mtp.py`'s companion (`tools/native/` tools assume a build that carries it).
