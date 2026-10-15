# Bonsai2-27B MTP 草稿头

[English](README.en.md) | **中文**

PrismML 官方发布的 `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 默认未打包 MTP 草稿头（`block_count=64`，851 个张量）。

本仓库提供了自蒸馏训练得到的 MTP（`nextn`）草稿头权重，以及将其合入官方模型用于投机解码加速的工具。合入后模型扩展至 65 层（新增第 64 层草稿头），文件体积增加约 0.35 GB（5.95 GB → 6.30 GB）。

在单卡 8GB 显存设备上，搭配 `llama.cpp` / KVMem（`--spec-type draft-mtp --spec-draft-n-max 2`），实测可在长达 240K 的真实对话上下文中持续维持稳定的投机加速。

---

## 实测表现

以下数据全部提取自合并自训练头（`distill2`）后、服务端在真实高压场景（含长文本检索与 Thinking 思考模式）持续运行累计的日志（`logs/kvmem-server.log`，共 131 轮长会话交互）：

### 1. 服务端长期运行统计（真实长会话负载，n_max=2）

| 统计项 | 实测数值 |
|---|---|
| 处理 Prompt 总量 | 14,284,924 tokens（覆盖 2.4K ~ 244.6K 超长上下文） |
| 实际生成 Token 总量 | 113,718 tokens |
| 草稿提出总量 | 103,672 tokens |
| 草稿接受总量 | 62,013 tokens |
| **全局加权接受率** | **59.82%**（单轮中位数 61.50%，峰值 94.10%） |
| **Decode 解码速度** | **中位数 35.89 t/s，峰值 55.54 t/s**（短上下文均值 48.7 t/s） |

本统计数据来自于本人实际工程类长会话统计，并没有刻意测试极限接受率，在单会话逐步累积上下文到240k时的实际数据。

模型在kvmem的Llama分支上运行，使用q4kv，32767的检索窗口与16384的单次生成窗口，256k逻辑上下文拉满。

### 2. 不同上下文长度下的表现

在 KVMem 动态显存分页架构下，不同 Prompt 长度区间的实测解码速度与接受率表现如下：

| 上下文长度 (Prompt) | 轮数 | 生成 Token 数 | 平均解码速度 | 速度范围 | 加权接受率 |
|---|---|---|---|---|---|
| 0 ~ 10K (短文本) | 6 | 1,408 | **48.75 t/s** | 40.5 ~ 55.5 t/s | 56.00% |
| 10K ~ 30K (常规会话) | 11 | 2,672 | **45.63 t/s** | 35.2 ~ 52.1 t/s | **64.56%** |
| 30K ~ 70K (长上下文) | 36 | 36,920 | 32.18 t/s | 22.5 ~ 44.9 t/s | 58.68% |
| 70K ~ 150K (超长文本) | 38 | 42,013 | 34.26 t/s | 25.9 ~ 43.1 t/s | 59.62% |
| 150K ~ 245K (极限上下文) | 40 | 30,705 | 33.83 t/s | 20.2 ~ 46.0 t/s | **61.27%** |

> **核心结论**：
> 1. 草稿头的接受率在整个 240K 全上下文跨度内极度稳健，即使在 150K~245K 的极限窗口下仍维持在 **61.27%**；
> 2. 短文本和常规交互下解码速度稳定在 **45 ~ 55 t/s**；而在 30K 以上深长上下文中，受检索与显存调度开销影响，整体解码速度收敛在 **32 ~ 35 t/s**，相比无草稿头基线（约 20 ~ 22 t/s）仍有明显增益。

### 3. 离线 Argmax 对齐评测（32 组 chunk / 16,384 行）

训练目标为蒸馏对齐主模型自身的输出决策：

| 预测深度 | Qwen 原始头 | **自训练头 (distill2)** | 提升幅度 |
|---|---|---|---|
| depth 1 | 66.9% | **82.3%** | +15.4% |
| depth 2 | 51.3% | **78.2%** | +26.9% |
| depth 3 | 44.7% | **74.5%** | +29.8% |

---

## 体积变化说明

官方模型剔除了全部 15 个 `blk.64.*` 张量。合入草稿头后新增的 0.351 GB 为必要的权重与结构开销：

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

此步骤会生成 6.30 GB 的中间文件 `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`。

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
- **训练目标**：主模型 Argmax 蒸馏（直接拟合主干模型自身的输出决策，避免拟合语料原文导致目标漂移）。
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

## 协议与致谢

- 本仓库代码与权重遵循 [MIT License](LICENSE)。
- 基础模型与推理引擎版权分别归属 [PrismML](https://github.com/PrismML) 与 [llama.cpp](https://github.com/ggerganov/llama.cpp) 项目。
- MTP 结构与投机解码调度逻辑源自 Qwen 与 `llama.cpp` 的 `graph_mtp` 实现。
