# Bonsai2-27B MTP 草稿头

[English](README.en.md) | **中文**

PrismML 官方发布的 `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 默认**没有打包 MTP 头**（`block_count=64`，共 851 个张量）。

这个仓库提供了训练好的 MTP（`nextn`）草稿头权重，以及将其合入官方模型、提升推理吞吐的完整工具链。合入后模型层数变为 65 层（新增第 64 层作为草稿头），文件体积增加约 0.35 GB（5.95 GB → 6.30 GB）。

在 8GB 显存设备上，配合 `llama.cpp` / KVMem 的投机解码（`--spec-type draft-mtp --spec-draft-n-max 2`），实测 decode 速度从 40.3 t/s 提升到 **45.7 t/s**（+13.4%）。

---

## 实测表现

### 1. 推理速度（单卡 8GB，KVMem，贪婪采样）

| 配置 | decode 速度 | 平均每轮接受数 (τ) | 接受率 |
|---|---|---|---|
| 官方模型 + Qwen 原始头 (`n=1`) | 40.3 t/s | 1.58 | 57.5% |
| 官方模型 + Qwen 原始头 (`n=3`) | 41.0 t/s | 2.08 | 36.1% |
| **官方模型 + 本训练头 (`n=2`，推荐)** | **45.7 t/s** | **2.02** | **51.3%** |
| 官方模型 + 本训练头 (`n=3`) | 43.5 t/s | 2.22 | 40.9% |

> 注：`n=2` 是实测的吞吐最优解；`n=3` 接受 token 虽多，但每轮验证开销更大，综合速度反而略低。Prefill 耗时不受草稿头影响。

### 2. 躯干 Argmax 对齐率（离线评测，32 组 chunk / 16,384 行）

头训练的目标是尽可能预测出主模型自身的 argmax 决策：

| 预测深度 | Qwen 原始头 | **本训练头 (distill2)** | 提升 |
|---|---|---|---|
| depth 1 | 66.9% | **82.3%** | +15.4% |
| depth 2 | 51.3% | **78.2%** | +26.9% |
| depth 3 | 44.7% | **74.5%** | +29.8% |

在 10 组真实长会话 prompt 上的在线 A/B 测试中（`n_max=3`），综合接受率由 74.7% 提升至 77.7%（7 胜 2 负 1 平），prompt 难度越大提升越明显。贪婪采样下生成结果与主模型逐字节完全一致。

---

## 体积变化说明

官方模型本身剔除了全部 15 个 `blk.64.*` 张量。补齐草稿头后新增的 0.351 GB 全部为权重与结构开销：

| 项目 | 大小 |
|---|---|
| 官方原始模型 (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`) | 5,946,648,928 字节 (~5.95 GB) |
| 合入草稿头后 (`Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf`) | 6,297,658,656 字节 (~6.30 GB) |
| **净增加** | **351,009,728 字节 (~0.351 GB)** |

其中 350,993,728 字节为 15 个头的量化权重，15,040 字节为 32 字节内存对齐填充，960 字节为 GGUF Header 索引与 `nextn_predict_layers` 元数据。

---

## 快速使用

### 准备环境与文件

1. 下载官方底模：`Ternary-Bonsai-2-27B-PTQ1_0.gguf`
2. 克隆本仓库（需安装 [Git LFS](https://git-lfs.com/) 以拉取权重文件）：
   ```bash
   git clone https://github.com/mytianyi0712/Bonsai2-27B-pq1-mtp-head.git
   cd Bonsai2-27B-pq1-mtp-head
   git lfs pull
   ```

### 步骤 1：给官方模型补齐 MTP 结构

官方底模缺少第 64 层结构。先用 `tools/graft_mtp.py` 从 Unsloth 发布的 Qwen3.8-27B GGUF 中按 byte-range 拉取原始 MTP 结构并追加进底模：

```bash
python tools/graft_mtp.py \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf
```

这一步执行完成后，会生成一个 6.30 GB 的临时文件 `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`。

### 步骤 2：注入训练好的草稿头权重

使用编译好的 `head-graft` 工具，将本仓库训练出的 `head_final.pt` 原地替换写入：

```bat
rem 导出 fp16 权重为各张量二进制文件
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export

rem 原地注入权重并校验（生成最终模型）
scripts\graft-head.bat head\distill2\head_final.pt Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf
```

最终生成的 `Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf` 即可直接用于推理。

### 步骤 3：启动服务

搭配 `llama.cpp` 或 KVMem 启动服务：

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

## 训练细节

- **参数规模**：424.7 M（15 个张量，单层完整的 GQA 24/4 注意力 + SwiGLU FFN）。
- **训练目标**：主模型 Argmax 蒸馏（直接拟合主干模型对自身的输出决策，避免拟合语料原文导致目标漂移）。
- **递归展开**：`n=3` 自喂链（深度 2 与深度 3 输入自身上一深度的隐藏态输出），配合 FastMTP 的 $\beta=0.6$ 深度衰减权重。
- **训练数据**：
  - 496,058 行通过模型自身在真实对话、Thinking 模板与 `q4_0` KV 下生成的自蒸馏轨迹（3 倍采样权重）；
  - 800,768 行在部署量化配置下重导出的语料隐藏态。
- **QAT 部署仿真**：大矩阵按 Q6_K 模拟，K/V 按 Q8_0 模拟，Norm 保留 F32，使训练权重在量化写入 GGUF 后损失趋近于 0。
- **硬件与开销**：单张消费级显卡（峰值显存 6.82 GiB），总计 9,996 步，训练耗时约 113 分钟。

日志原文保存在 `logs/distill2.log` 与 `logs/distill1.log` 中。

---

## 仓库结构

```
├── head/distill2/head_final.pt   # 训练完成的草稿头权重（Git LFS，~849MB）
├── tools/                        # 导出、嫁接、格式转换与评测脚本
│   ├── graft_mtp.py              # 将 MTP 结构合入官方无头模型的工具
│   ├── export_head.py            # checkpoint 转量化二进制文件
│   ├── verify_graft.py           # 嫁接模型 logits 精度对齐校验
│   ├── mtp_head.py               # MTP 头结构实现（含 Hadamard 逆变换适配）
│   └── train_mtp.py              # 训练主脚本
├── tools/native/                 # 对应的 C++ 辅助工具源码（用于编译进 llama.cpp）
├── scripts/                      # Windows 批处理流程脚本
└── logs/                         # 原始训练与 A/B 测试日志
```

---

## 注意事项

1. **短 Prompt 收益有限**：短上下文、少推理的场景下，原装头的 `n_max=1` 接受率本就在 95% 以上，提速空间不大；主要增益体现在 8K 以上长上下文及复杂推理链中。
2. **采样温度影响**：上述测试均基于贪婪采样（`temp=0`）。高温度采样下由于主模型输出本身带有随机性，草稿头接受率会有所回落（实测约为 36%~50% 浮动）。
3. **隐私说明**：训练语料源于本地交互记录，包含私人工作上下文，按安全规范不予公开。仓库已附带完整的数据生成与特征提取脚本供自行构建。

---

## 协议与致谢

- 本仓库代码与权重遵循 [MIT License](LICENSE)。
- 基础模型与推理引擎版权分别归属 [PrismML](https://github.com/PrismML) 与 [llama.cpp](https://github.com/ggerganov/llama.cpp) 项目。
- MTP 结构与投机解码调度逻辑源自 Qwen 与 `llama.cpp` 的 `graph_mtp` 实现。
