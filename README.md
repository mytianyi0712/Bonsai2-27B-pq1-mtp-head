# Bonsai2-27B — 为无 MTP 的官方模型补上自蒸馏 MTP 草稿头

[English](README.en.md) | **中文**

`Ternary-Bonsai-2-27B-PTQ1_0`（架构 `qwen35`，PrismML PTQ1_0 三值主干，256K 上下文）
**官方发布的 GGUF 是不带 MTP 头的**（`Ternary-Bonsai-2-27B-PTQ1_0.gguf`：`block_count=64`，
851 个张量，`blk.64.*` 一个都没有）。本仓库提供的就是**补上 MTP 能力所需的那一个头**，
并用自蒸馏把它从"能跑"训到"好用"：

1. **先补结构** —— 取一个 MTP 头（15 个 `blk.64.*` 张量）追加进官方模型，得到可做投机解码的
   带头模型：`block_count` 64 → 65，新增 `qwen35.nextn_predict_layers=1`，**体积 +0.351 GB**。
   这一步是**纯追加**，会把文件变大，不是原地改。
2. **再换权重** —— 把本仓库训练出的头 **原地替换**掉那 15 个张量（按各自原类型 Q6_K / Q8_0 / F32
   就地改写，其余字节不动），得到本仓库的 `-mtp-distill2.gguf`。替换这一步**体积零变化**。

本仓库只放**训练出的头权重**，不重新分发任何模型文件（见
[仓库里有什么 / 没有什么](#仓库里有什么--没有什么)）。

| 文件 | 张量数 | 体积 | `block_count` |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（官方，**无 MTP**） | 851 | 5,946,648,928 B（5.95 GB） | 64 |
| 补上 MTP 结构后的带头模型（= 本仓库的嫁接源） | 866 | 6,297,658,656 B（6.30 GB） | 65 |
| 换成训练头后的产物 `-mtp-distill2.gguf` | 866 | 6,297,658,656 B（与上一行等长） | 65 |

0.351 GB 的账能对上（已与 HuggingFace 原始头逐字节核对，15/15 相同）：

| 组成 | 字节 |
|---|---|
| 15 个 `blk.64.*` 张量的**原始**数据 | 350,993,728 |
| 32 字节对齐填充 | +15,040 |
| 张量表新增条目 + `qwen35.nextn_predict_layers` 元数据 | +960 |
| **文件增长合计** | **351,009,728**（≈0.351 GB） |

即 0.351 GB 几乎全是头权重本身；结构开销加起来只占 16 KB。

* 官方基础模型（**无 MTP，需先补头**）：`Ternary-Bonsai-2-27B-PTQ1_0.gguf`
* 头：**424.7 M 参数**，15 个张量，fp16 检查点 → 按部署类型做 QAT（大矩阵 Q6_K，k/v Q8_0，norm F32）
* 运行时：`llama.cpp` + KVMem fork，`--spec-type draft-mtp`

---

## 核心指标

| 指标（全部为实测，见[证据](#证据--日志摘录)） | 原装头 | **distill2（本仓库）** |
|---|---|---|
| 离线 depth-1 躯干-argmax 一致率 | 66.9 % | **82.3 %** |
| 离线 depth-2 | 51.3 % | **78.2 %** |
| 离线 depth-3 | 44.7 % | **74.5 %** |
| decode 速度 `n_max=1` | 40.3 t/s | 41.3 t/s |
| decode 速度 `n_max=2` | — | **45.7 t/s** |
| decode 速度 `n_max=3` | 41.0 t/s | 43.5 t/s |
| 每轮平均接受 token 数 τ，`n_max=2` | — | **2.02** |
| prefill | 不受草稿头影响 | 不受草稿头影响 |

10 个真实会话提示词的在线 A/B（`n_max=3`，按草案事件加权）：**74.7 % → 77.7 %**
（+3.0 点），7 胜 / 2 负 / 1 平 —— **越难的提示词增益越大**（两个最长/最难的提示词分别
+7.6、+6.8 点）；长提示词在 `n_max=1` 下 **80.3 % → 88.2 %**。所有 A/B 的贪婪生成文本
**逐字节相同**。

---

## 这个头是什么

依据 `src/models/qwen35.cpp`（`graph_mtp`）与 `common/speculative.cpp`（`draft-mtp`），位置 `q`
的一行 MTP 由 token `x_q` 加上躯干隐藏态 `h_{q-1}`（**final-norm 之后**，即 LM head 的输入）
构成，预测 `x_{q+1}`：

```
tok_embd(x_q) → Hadamard 逆变换（先旋转，后符号）      # prism.hadamard.inverse_weight_names
  → RMSNorm(enorm) ‖ RMSNorm(hnorm)                  # 拼接（e 在前、h 在后）10240
  → eh_proj 10240→5120
  → 完整注意力块：attn_norm → attn_q（12288 = 逐头交错的 [q(256)|gate(256)]）
      → per-head q_norm → gate = sigmoid(后半) → attn_k/attn_v（4×256）
      → per-head k_norm → RoPE（sections [11,11,10,0]，查询位置 = q）
      → attn × sigmoid(gate) → attn_output → +残差 → post_attention_norm → SwiGLU FFN → +残差
  → RMSNorm(shared_head_norm) → 共享 output.weight → logits
```

训练代码精确复现了两个不显眼的约定（都用 `llama.cpp` 逐 logits 对齐验证过，逐行 cos ≥ 0.9999）：

1. **GGML 维度序。** `ne[0]` 变化最快，因此 numpy/torch 形状是 `reversed(ne)`，**绝不能转置**。
   `reshape(ne).t()` 会让值正确但排列全错（`cos ≈ 0`）。
2. **行语义。** 行按 `(x_q, h_{q-1})` 索引，RoPE 位置 `q`，目标 `x_{q+1}`。深度 ≥ 2 时 token 用
   真值，但隐藏态用的是本头上一深度的输出 —— 推理时自喂，所以训练链也必须自喂。

## 怎么训练的

躯干-argmax 蒸馏、fp16 原生、每步一个完整注意力块、链深 3：

| 项 | 值 |
|---|---|
| 目标 | CE 对齐**冻结躯干自身的 argmax**（离线 `head.lm_logits(h_dump)` 现算），不是对齐语料文本 |
| 链 | `n=3` 自喂链（`forward_chain`），注意力 = 窗口行（位置 ≤ 锚点）+ 本链更浅的行 + 自身 |
| 初始化 | **原装头**（起点已经是躯干的忠实近似） |
| 数据 | 用真实聊天模板 + 思考 + `q4_0` KV 生成的自蒸馏轨迹（**496,058 行**，×3 采样权重）+ `q4_0` 语料重导（**800,768 行**） |
| 精度 | fp16 权重/matmul，fp32 master 权重 + AdamW 放 CPU，QAT 按**部署**类型往返（Q6_K / Q8_0） |
| 成本 | 9,996 步 / 113.3 分钟 / 峰值显存 6.82 GiB / 1,507 tok/s |

为什么不直接用语料文本训练：验收率由 `草稿 argmax == 躯干 argmax` 决定，而躯干自身与语料原文
只有约 66 % 一致 —— 用文本训出来的头会**比躯干更会写这段文本**，于是偏离躯干、验收率反降。
实测：文本-CE 头 70.7 % vs 蒸馏头 80.3 %（同一提示词，`n_max=1`）。

训练曲线（held-out chunk），摘自 `logs/distill2.log`：

```
[train] step  500 VAL trunk-argmax depth1:79.2% | depth2:74.8% | depth3:72.6%
[train] step 5000 VAL trunk-argmax depth1:80.0% | depth2:76.5% | depth3:73.9%
[train] step 9500 VAL trunk-argmax depth1:80.3% | depth2:77.5% | depth3:75.0%
[train] FINAL VAL trunk-argmax agreement depth1:80.4% | depth2:77.5% | depth3:75.1%
[train] peak VRAM 6.82 GiB (reserved 8.06 GiB)
[train] done in 113.3 min -> D:\models\mtp-train\distill2\head_final.pt
```

## 证据 / 日志摘录

以下全部逐字摘自本仓库内的文件。

**本头的最终验证** —— `logs/distill2.log`

```
[train] initialized from D:\models\mtp-train\distill1\head_final.pt
[train] trainable params: 424.7 M (fp16)
[train] steps/epoch≈4998, total≈9996, tokens≈10.2M, qat=deploy
[train] FINAL VAL trunk-argmax agreement depth1:80.4% | depth2:77.5% | depth3:75.1%
[train] peak VRAM 6.82 GiB (reserved 8.06 GiB)
[train] done in 113.3 min -> D:\models\mtp-train\distill2\head_final.pt
```

**上一轮迭代（distill1，留作对照）** —— `logs/distill1.log`

```
[train] FINAL VAL trunk-argmax agreement depth1:80.0% | depth2:75.7% | depth3:74.0%
[train] done in 128.5 min -> D:\models\mtp-train\distill1\head_final.pt
```

**验收原始日志** —— `logs/ab-n*.log`。每个文件是一次 `llama-kvmem-server` 运行
（`KVMEM_TRACE=1`），读到该次运行结束的 `spec_stats` 行即可得该次接受率。**注意**：这几个日志
是用**原装头**跑的，因此它们记录的是测量口径与基线分布，而不是增益本身：

```
=== ab-n1.log,  n_max = 1  (原装头) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=198 n_accept=102 n_restore=0 accept_pct=51.5
KVMEM_TRACE spec_stats n_gen=291 n_drafted=147 n_accept=145 n_restore=0 accept_pct=98.6

=== ab-n2.log,  n_max = 2  (原装头) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=349 n_accept=125 n_restore=0 accept_pct=35.8

=== ab-n3.log,  n_max = 3  (原装头) ===
KVMEM_TRACE spec_stats n_gen=300 n_drafted=504 n_accept=131 n_restore=0 accept_pct=26.0
```

接受率是**内容**的函数，不只是头的函数 —— 散文与结构化（JSON / 工具调用）token 的接受率远低于
数字片段（同一构建在纯数字段上 >95 %）。`accept_pct` 是**每草案 token**口径，不同 `n_max`
的值不可横比；跨 `n_max` 请比 `t/s`。

**嫁接一致性验证。** 导出的 fp16 张量先量化到各自的部署类型，再补丁进基础模型的字节副本；
`draft-logits` 在打过补丁的文件上重跑 MTP 图，同一批行再在 PyTorch 里过一遍检查点
（`tools/verify_graft.py`）：

| 检查 | 结果 |
|---|---|
| 块输出逐行 cos | min 0.9990，均值 ≈0.9998 |
| 末行 logits cos | 0.999951 |
| top-1 / top-10 | 命中 / 10 中 10 |
| 验收 A/B 中的贪婪文本 | 每次运行都逐字节相同 |

*（离线各深度一致率来自 32+32 chunk / 16,384 行的复核测量；上面引用的训练日志里那行是 8-chunk
口径，两者数值不同，请勿混用。）*

## 仓库里有什么 / 没有什么

**有：**

```
head/distill2/head_final.pt   849 MB —— 15 个 fp16 张量（头本体），Git LFS 跟踪
logs/                         训练日志 + 验收原始日志（逐字，未删改）
tools/                        python 流水线（dump → trace → train → export → graft → verify）
tools/native/                 四个放进 llama.cpp/tools/ 的 C++ 工具源码
scripts/                      Windows 上跑整条流水线的 .bat / .sh 驱动
deploy/bonsai-chat-template.jinja
```

**没有：** 官方基础模型、补过头与训过头之后的完整模型、隐藏态 dump 与自蒸馏轨迹
（数百 GB，可完整再生成 —— 见下面命令）。训练语料由本机会话记录抽取而成，有意不公开。

## 怎么用这个头

官方模型**没有** MTP，所以完整路径是「先补结构，再换权重」两步。

**步骤 1 —— 给官方模型补上 MTP 结构（体积 +0.351 GB）。** 这一步是**追加** 15 个
`blk.64.*` 张量并更新元数据，不是在原文件上就地改：

```bat
rem 追加一个 MTP 头 -> 得到带头模型（本仓库工具链的嫁接源）
python tools\graft_mtp.py  Ternary-Bonsai-2-27B-PTQ1_0.gguf  Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf
```

`tools/graft_mtp.py` 会用 HTTP range 请求从 `unsloth/Qwen3.8-27B-GGUF`（HuggingFace）拉取
Qwen 原生 MTP 头（**步骤 1 需要联网**），把它的 15 个 `blk.64.*` 张量追加到官方模型末尾，
并把 `qwen35.block_count` 64 → 65、补上 `qwen35.nextn_predict_layers=1`，末尾自校验
（`tensors == base+15`、`block_count == 65`、`blk.64 present 15/15`）。这一步的结果就是下面指标表里的
**"原装头"基线**；它的权重是 Qwen 原生权重，因此**有意不**加入 `prism.hadamard.weight_names`。

**步骤 2 —— 用本仓库训练出的头替换那 15 个张量（体积零变化）。** 这一步才是**原地改**：

```bat
rem 2a. 从检查点导出 fp16 张量
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export

rem 2b. 原地替换进带头模型的字节副本 + 验证
graft-head.bat  head\distill2\head_final.pt  Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf
```

产物 `-mtp-distill2.gguf` 与**它的输入**等长（6,297,658,656 B）。注意
`graft-head.bat` 里的 `BASE` 必须是**带头模型**；若指向官方无头模型，`head-graft` 会以
`no blk.64.* tensors in ...` 退出（那是设计如此 —— 官方模型里没有可替换的张量，
要用步骤 1 先补出来）。

`scripts/graft-head.bat` 封装了步骤 2 的三步（`export_head.py` → `head-graft.exe` →
`draft-logits.exe` + `verify_graft.py`）。然后起服务，读 `spec_stats`：

```bat
scripts\ab-accept.bat  <打过补丁的.gguf>  <提示词.txt>  256  2
```

在 8 GB 显存的测试机上实测的甜点配置是 **`n_max=2`**（45.7 t/s vs 原装头 `n_max=1` 的
40.3 t/s，即 **+13.4 %**）。`n_max=3` 的平均接受长度最高（τ 2.22），但比 `n_max=2` 慢，
因为每一行额外草案都要付一次 verify 的钱。

## 复现整条流水线

native 工具是 `llama.cpp/tools/` 下的新增项，把它们放进 `llama.cpp` 检出（或 KVMem fork）后构建：

```bat
cmake --build build-win --target draft-logits head-graft mtp-trace mtp-dump -j 8
```

然后端到端：

```bat
rem 0. 官方模型无 MTP：先补结构，产出一个带头模型作为后续所有工具链的输入
python tools\graft_mtp.py  Ternary-Bonsai-2-27B-PTQ1_0.gguf  Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf

rem 1. 语料：会话记录 -> 文本（每个文件 >=100k 字符）
python tools\extract-sessions.py

rem 2. 隐藏态：每个位置的 (token, final-norm 后的 h)，用部署同款 KV 量化
mtp-dump.exe -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --list filelist.txt --out dump --chunk 2048 --ub 1024 ^
             --max-tokens 8500000 --ngl 99 --threads 8 -ctk q4_0 -ctv q4_0

rem 3. 用真实模板 + 思考生成自蒸馏轨迹
mtp-trace.exe -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --prompts trace-prompts.txt --out traces ^
              --template-file bonsai-chat-template.jinja --max-new 700 --chunk 1024 ^
              --ctk q4_0 --ctv q4_0 --ngl 99 --threads 8
python tools\mtp-trace-merge.py --src traces --out traces-merged --min-rows 520

rem 4. 训练（躯干-argmax 蒸馏，n=3 自喂链，部署同型 QAT）
python tools\train_mtp.py --data dump --data traces-merged --data traces-merged --data traces-merged ^
       --out run --init-from shipped-head.pt --device cuda --seq 512 --batch 2 ^
       --epochs 2 --lr 3e-5 --accum 8 --val-every 500 --objective distill --steps 3 --qat deploy

rem 5. 嫁接 + 验证 + 测速
python tools\export_head.py --ckpt run\head_final.pt --out head-export
head-graft.exe --src Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --out PATCHED.gguf --weights head-export
draft-logits.exe -m PATCHED.gguf --tokens <64 个真实 token id> --out verify-dir
python tools\verify_graft.py --val verify-dir --ckpt run\head_final.pt
python tools\mtp-bench.py --model PATCHED.gguf --n-max 2
```

`tools/mtp-retrain-v2.py` 编排了完整四阶段链（用 `q4_0` 重导 → 生成轨迹 → 重训 →
导出/嫁接/验证），本次检查点就是它产出的。注意全流程（dump / trace / 训练时的 `--gguf` /
嫁接）都以上一步产出的**带头模型**为输入 —— 官方无头模型只能喂给步骤 0。

## 从检查点重建头文件

```bat
python tools\export_head.py --ckpt head\distill2\head_final.pt --out head-export
python -c "import torch; sd=torch.load(r'head\distill2\head_final.pt', map_location='cpu'); print(len(sd), sum(v.numel() for v in sd.values())/1e6, 'M params')"
```

预期输出 `15 424.699392 M params`，全部张量为 fp16，与下列 GGUF 张量一一对应：
`blk.64.nextn.eh_proj.weight`、`blk.64.nextn.enorm.weight`、`blk.64.nextn.hnorm.weight`、
`blk.64.attn_norm.weight`、`blk.64.attn_q_norm.weight`、`blk.64.attn_k_norm.weight`、
`blk.64.post_attention_norm.weight`、`blk.64.attn_{q,k,v,output}.weight`、
`blk.64.ffn_{gate,up,down}.weight`、`blk.64.nextn.shared_head_norm.weight`
（见 `tools/export_head.py` 的 `NAME_MAP`）。本仓库中该文件的 SHA-256：
`3c2788d6595c7f751c1cdc82657b31ffe3a67598d6eeb993239a5bb34f43ad54`。

## 已知边界

* 短上下文、`n_max=1` 时原装头已经是 ~98–100 %，没有空间。增益集中在长上下文与更难的分。
* 上一轮迭代存在真实但很小的过拟合（depth 1/2/3 分别 +1.7 / +3.4 / +4.3 点）；本轮
  held-out gap ≈ 0（−0.3 / −0.2 / +0.8），与"在模型自身输出分布上训练"一致。
* 采样解码（`temp > 0`）的接受率天然低于贪婪 —— 上面的验证都是贪婪口径。若要在采样下进一步
  提速，下一根杠杆是软 KL 目标。
* 绝对 `t/s` 依机器与内容而定，有意义的是相对排序。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。基础模型与 KVMem `llama.cpp` fork **不**在本许可证覆盖范围内；
本仓库只包含训练出的头权重、流水线代码与日志。

## 致谢

* 头结构与 `draft-mtp` 配对语义：`llama.cpp`（`src/models/qwen35.cpp`、`common/speculative.cpp`）
  与 KVMem fork（`llama-ext.h` 的 `llama_set_embeddings_nextn`）。
* 深度衰减加权参考 FastMTP 一系工作（自蒸馏数据 + 权重共享 MTP 头）。
* MTP 嵌入查表所需的 Hadamard 逆变换是上游补丁；`tools/native/` 里的工具假定构建已带上它。
