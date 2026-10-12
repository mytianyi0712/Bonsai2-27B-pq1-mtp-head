# Bonsai2-27B MTP Draft Head

**English** | [中文](README.md)

The official `Ternary-Bonsai-2-27B-PTQ1_0.gguf` released by PrismML ships without an MTP draft head (`block_count=64`, 851 tensors total).

This repository provides a self-distilled MTP (`nextn`) draft head and tooling to graft it onto the official base model for speculative decoding acceleration. Adding the head introduces block 64, expanding the model to 65 layers and increasing the file size by ~0.35 GB (5.95 GB → 6.30 GB).

On an 8GB VRAM consumer GPU paired with `llama.cpp` / KVMem (`--spec-type draft-mtp --spec-draft-n-max 2`), the head maintains stable speculative acceleration across real-world dialog contexts extending up to 240K tokens.

---

## Measured Performance

The following benchmarks are derived from the server runtime logs (`logs/kvmem-server.log`, 131 long-session dialog turns) under real-world multi-turn workloads with Thinking reasoning tokens and memory paging after grafting the self-trained head (`distill2`):

### 1. Long-Term Runtime Statistics (Real-World Workloads, n_max=2)

| Metric | Measured Value |
|---|---|
| Total Prompt Tokens Processed | 14,284,924 tokens (covering 2.4K ~ 244.6K ultra-long contexts) |
| Total Output Tokens Generated | 113,718 tokens |
| Total Speculative Tokens Drafted | 103,672 tokens |
| Total Speculative Tokens Accepted | 62,013 tokens |
| **Cumulative Global Acceptance Rate** | **59.82%** (Per-turn median: 61.50%, Peak: 94.10%) |
| **Decode Throughput** | **Median: 35.89 t/s, Peak: 55.54 t/s** (Short-context mean: 48.7 t/s) |

### 2. Performance Across Context Window Depths

Under KVMem dynamic memory tiering, decode throughput and acceptance rates across different prompt lengths:

| Context Window (Prompt) | Turns | Tokens Generated | Mean Decode Speed | Speed Range | Acceptance Rate |
|---|---|---|---|---|---|
| 0 ~ 10K (Short text) | 6 | 1,408 | **48.75 t/s** | 40.5 ~ 55.5 t/s | 56.00% |
| 10K ~ 30K (Standard dialog) | 11 | 2,672 | **45.63 t/s** | 35.2 ~ 52.1 t/s | **64.56%** |
| 30K ~ 70K (Long context) | 36 | 36,920 | 32.18 t/s | 22.5 ~ 44.9 t/s | 58.68% |
| 70K ~ 150K (Ultra-long context) | 38 | 42,013 | 34.26 t/s | 25.9 ~ 43.1 t/s | 59.62% |
| 150K ~ 245K (Extreme context) | 40 | 30,705 | 33.83 t/s | 20.2 ~ 46.0 t/s | **61.27%** |

> **Key Observations**:
> 1. Acceptance rate remains remarkably stable across the entire 240K window, maintaining **61.27%** even at 150K~245K depths.
> 2. Decode speed averages **45 ~ 55 t/s** in short and standard contexts. In deep contexts (>30K tokens), retrieval and memory paging overhead bring decode speed to **32 ~ 35 t/s**, still offering significant acceleration over the baseline (~20 ~ 22 t/s without MTP).

### 3. Offline Argmax Alignment Evaluation (32 Chunks / 16,384 Rows)

The training target aligns directly with the trunk model's own output predictions:

| Prediction Depth | Stock Qwen Head | **Trained Head (distill2)** | Improvement |
|---|---|---|---|
| depth 1 | 66.9% | **82.3%** | +15.4% |
| depth 2 | 51.3% | **78.2%** | +26.9% |
| depth 3 | 44.7% | **74.5%** | +29.8% |

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

## License & Credits

- Code and weights are released under the [MIT License](LICENSE).
- Base model and engine rights belong to [PrismML](https://github.com/PrismML) and [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Architecture and speculative decoding scheduling follow Qwen and `llama.cpp` `graph_mtp`.
