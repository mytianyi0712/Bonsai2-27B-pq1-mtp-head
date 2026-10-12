# Bonsai2-27B MTP Draft Head

**English** | [中文](README.md)

The official `Ternary-Bonsai-2-27B-PTQ1_0.gguf` released by PrismML ships **without an MTP draft head** (`block_count=64`, 851 tensors total).

This repository provides a self-distilled MTP (`nextn`) draft head and tooling to graft it onto the official base model. Adding the head introduces block 64 as the draft layer, increasing the file size by ~0.35 GB (5.95 GB → 6.30 GB).

On an 8GB VRAM consumer GPU using `llama.cpp` / KVMem speculative decoding (`--spec-type draft-mtp --spec-draft-n-max 2`), real-world decode throughput increases from 40.3 t/s to **45.7 t/s** (+13.4%).

---

## Benchmarks

### 1. Decode Throughput (Single 8GB GPU, KVMem, Greedy Sampling)

| Configuration | Decode Speed | Mean Accepted Tokens (τ) | Acceptance Rate |
|---|---|---|---|
| Base model + Stock Qwen head (`n=1`) | 40.3 t/s | 1.58 | 57.5% |
| Base model + Stock Qwen head (`n=3`) | 41.0 t/s | 2.08 | 36.1% |
| **Base model + Distilled head (`n=2`, Recommended)** | **45.7 t/s** | **2.02** | **51.3%** |
| Base model + Distilled head (`n=3`) | 43.5 t/s | 2.22 | 40.9% |

> Note: `n=2` provides the highest overall throughput. While `n=3` yields more accepted tokens per round, the additional verification overhead outweighs the benefit. Prefill latency is completely unaffected by the draft head.

### 2. Trunk Argmax Alignment (Offline Eval, 32 Chunks / 16,384 Rows)

The head is trained to predict the frozen trunk model's own argmax decisions:

| Draft Depth | Stock Qwen Head | **Distilled Head (distill2)** | Delta |
|---|---|---|---|
| depth 1 | 66.9% | **82.3%** | +15.4% |
| depth 2 | 51.3% | **78.2%** | +26.9% |
| depth 3 | 44.7% | **74.5%** | +29.8% |

In online A/B tests across 10 long-context prompt traces (`n_max=3`), weighted acceptance rate improved from 74.7% to 77.7% (7 wins, 2 losses, 1 tie), with the largest gains observed on complex prompts (+7.6% and +6.8%). Under greedy sampling, generated text output is byte-for-byte identical to the non-speculative baseline.

---

## File Size Accounting

The official base model completely omits the 15 `blk.64.*` tensors. Grafting the head adds 0.351 GB:

| File | Size |
|---|---|
| Official base (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`) | 5,946,648,928 bytes (~5.95 GB) |
| Grafted model (`Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf`) | 6,297,658,656 bytes (~6.30 GB) |
| **Delta** | **351,009,728 bytes (~0.351 GB)** |

Breakdown: 350,993,728 bytes for the 15 quantized tensors, 15,040 bytes for 32-byte alignment padding, and 960 bytes for the GGUF header tensor table and `nextn_predict_layers` metadata.

---

## Quickstart

### Prerequisites

1. Obtain the official base model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
2. Clone this repository with [Git LFS](https://git-lfs.com/) enabled:
   ```bash
   git clone https://github.com/mytianyi0712/Bonsai2-27B-pq1-mtp-head.git
   cd Bonsai2-27B-pq1-mtp-head
   git lfs pull
   ```

### Step 1: Add MTP Structure to the Base Model

The official model lacks block 64. Use `tools/graft_mtp.py` to append the MTP tensor structure from Unsloth's Qwen3.8-27B GGUF via byte-range requests:

```bash
python tools/graft_mtp.py \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf
```

This creates an intermediate 6.30 GB file with block 64 added.

### Step 2: Inject Trained Weights

Use `head-graft` to overwrite the stock weights with `head_final.pt` in place:

```bat
rem Export fp16 tensors from PyTorch checkpoint
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export

rem Patch weights in place and verify logits alignment
scripts\graft-head.bat head\distill2\head_final.pt Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf
```

The resulting `Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf` is ready for deployment.

### Step 3: Serve

Launch with `llama-server` or KVMem:

```bat
llama-server.exe ^
    -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf ^
    --spec-type draft-mtp ^
    --spec-draft-n-max 2 ^
    --flash-attn on ^
    -b 4096 -ub 256 ^
    -c 262144
```

---

## Training Highlights

- **Parameters**: 424.7M (15 tensors; full GQA 24/4 attention layer + SwiGLU FFN).
- **Objective**: Trunk argmax distillation (learning the trunk model's own predictions rather than raw corpus tokens to prevent drift).
- **Recurrent Chain**: `n=3` self-feeding chain with FastMTP depth decay ($\beta=0.6$).
- **Data**:
  - 496,058 rows of model self-distillation traces under `q4_0` KV and Thinking chat template (3x sampling weight);
  - 800,768 rows of re-exported corpus hidden states with deployment quantization.
- **Quantization-Aware Training (QAT)**: Large matrices simulated as Q6_K, K/V as Q8_0, RMSNorm as F32.
- **Resource Footprint**: Single consumer GPU (peak VRAM 6.82 GiB), 9,996 steps, ~113 minutes wall time.

Full logs are archived in `logs/distill2.log` and `logs/distill1.log`.

---

## Repository Structure

```
├── head/distill2/head_final.pt   # Trained draft head checkpoint (Git LFS, ~849MB)
├── tools/                        # Python scripts for export, grafting, and verification
│   ├── graft_mtp.py              # Appends MTP structure to base model
│   ├── export_head.py            # Converts checkpoint to raw tensor binaries
│   ├── verify_graft.py           # Validates grafted logits against PyTorch reference
│   ├── mtp_head.py               # PyTorch MTP head definition (with Hadamard inverse)
│   └── train_mtp.py              # Main training script
├── tools/native/                 # C++ tool sources for llama.cpp integration
├── scripts/                      # Windows batch automation scripts
└── logs/                         # Training and benchmark logs
```

---

## Notes & Tradeoffs

1. **Short prompts show minimal speedup**: On short context windows (<2K tokens), stock head acceptance is already ~95%+, leaving little room for acceleration. Gains concentrate in 8K+ long context and complex reasoning traces.
2. **Temperature sensitivity**: Benchmarks reflect greedy decoding (`temp=0`). Higher sampling temperatures increase trunk stochasticity, reducing acceptance rates to ~36%-50%.
3. **Training data privacy**: Training data was drawn from local interactive sessions and contains private context; it is not published. Trace extraction scripts are provided in `tools/` for users wishing to train on custom corpora.

---

## License & Credits

- Code and weights are released under the [MIT License](LICENSE).
- Base model and engine rights belong to [PrismML](https://github.com/PrismML) and [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Architecture and speculative decoding scheduling follow Qwen and `llama.cpp` `graph_mtp`.
