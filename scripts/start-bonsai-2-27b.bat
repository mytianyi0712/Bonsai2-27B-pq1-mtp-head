@echo off
chcp 65001 >nul
setlocal

rem ===========================================================================
rem  Bonsai 2 27B (PTQ1_0 + MTP) -- one-click start: panel + browser + llama backend
rem
rem  Runtime: kvmem-llama.cpp v0.16.0-rc3 (llama.cpp b81c99b), local 3-way merge of
rem  PrismML ptq1-accel (PTQ1_0 quant + prism.hadamard runtime).
rem  Fixes carried: GDN RECORD accepts raw gate; FA quantized KV forced to
rem  non-interleaved smem layout (q4_0 KV numeric fix); chat template allows
rem  reasoning_effort=high.
rem
rem  Model: Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf
rem    blk.64.* MTP head = self-distilled (496k rows of the model's own thinking-mode
rem    output + 800k q4_0 corpus rows; trunk-argmax distillation, n=3 self-fed chain,
rem    deployment-type QAT).  Held-out agreement with the trunk argmax:
rem    depth1/2/3 = 82.3/78.2/74.5 pct (shipped head: 66.9/51.3/44.7), train-val gap ~0.
rem    Shipped-head version: start-bonsai-2-27b-shipped.bat  (rollback / A-B)
rem
rem  256k design: -c 262144 is a logical workspace.  The full history KV lives in host
rem  memory; the GPU keeps a bounded q4_0 window (budget+reserve) and replays the
rem  blocks relevant to the current query.  Clients send the full messages history.
rem
rem  MEASURED PARAM MATRIX (same 5k prompt, greedy, q4_0, decode 384 tokens; all rows block 128, block 32 untested):
rem    config                              VRAM     prefill         decode
rem    -ub 512 budget 24576 reserve 32768  7695 MiB 18.6s 276 t/s  45.7 t/s  (old)
rem    -ub 256 budget 24576 reserve 32768  7503     14.2s 362      47.3
rem    -ub 128 budget 24576 reserve 32768  7407     14.2s 361      50.5
rem    -ub 256 budget 24576 reserve 24576  7319     14.0s 367      49.1     (least VRAM)
rem    -ub 256 budget 40960 reserve 32768  7793     22.7s 226      41.9     (bigger budget is SLOWER)
rem    -ub 256 budget 32768 reserve 24576  7474     14.0s 367      47.7     (this config)
rem  Rule: this machine sits at the VRAM edge -- every 100-200 MB saved buys 5-10 pct
rem  speed.  -b (logical batch) does NOT affect VRAM or speed (4096 vs 2048 identical).
rem
rem  WHY budget=32768, and the discipline:
rem    Measured single-turn context growth reached 17k-25k tokens.  If one turn grows by
rem    more than budget, KVMem SKIPS query_replay; after that the MTP sync degrades for
rem    the whole session and decode collapses from ~30 ms/token to ~120 ms/token (7-8 t/s).
rem    budget 32768 leaves margin.  Keep single-turn growth under 32768 (summarize or
rem    trim huge tool outputs); if it already collapsed, a restart recovers it.
rem
rem  Fallback: --spec-type none gives a steady 20-30 t/s (no MTP acceleration, but the
rem  above degradation cannot trigger).
rem ===========================================================================

set "CUDA_VISIBLE_DEVICES=0"

rem  MTP 草稿上下文一次 catch-up 失败会永久停摆：catchup_failed 标志只在构造函数初始化，
rem  且失败后 flush_deferred 已把 pending 清掉，begin() 不再重试，导致整个会话 drafted=0，
rem  decode 从 ~47 掉到 ~22 t/s，只能重启恢复（llama.cpp/common/speculative.cpp）。
rem  本开关绕过 deferred 路径，使失败不再粘滞（代价：放弃 deferred 预填优化，prefill 略慢）。
rem  已打补丁的版本（begin() 里重新武装）上线后可删除本行。
set "LLAMA_MTP_EAGER_CATCHUP=1"

rem  【为什么不保留 cache 会很难受 + 任务状态】长会话实测（110k 上下文）：
rem  检索查询退化为 "最后 64 token"（query_source=bootstrap_suffix, q_rows=0，占全部轮次
rem  的 84%），而 sink 只有 1 个块（128 token）-> 会话开头那份任务清单/系统提示会被换出，
rem  模型看不到"已完成哪几步"，于是反复重做已完成的任务（A->B->C->又回头做 A）。
rem  对策：sink 4096 常驻开头（系统提示+任务清单），recent 8192 常驻最新几步（当前工具循环）。
rem  两者从既有 budget 里划分（block 32：128+256=384 块 of 1024），不增加显存，只是少 384 块给查询检索。
set "ROOT=D:\models"
set "BINDIR=%ROOT%\kvmem-llama.cpp\build-win\bin"
set "MODEL=%ROOT%\Ternary-Bonsai-2-27B-PTQ1_0-mtp-distill2.gguf"
set "TEMPLATE=%ROOT%\bonsai-chat-template.jinja"
set "PANEL=%ROOT%\kvmem-panel.py"
set "LOG=%ROOT%\kvmem-server.log"

set "HOST=127.0.0.1"
set "PORT=18200"
set "PANELPORT=18500"

if not exist "%BINDIR%\llama-kvmem-server.exe" (
    echo [ERROR] llama-kvmem-server.exe not found: "%BINDIR%"
    pause & exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] model not found: "%MODEL%"
    pause & exit /b 1
)
if not exist "%TEMPLATE%" (
    echo [ERROR] chat template not found: "%TEMPLATE%"
    pause & exit /b 1
)
if not exist "%PANEL%" (
    echo [WARN] panel script not found: "%PANEL%" (skipping panel)
)

echo.
echo   Bonsai 2 27B  PTQ1_0 + MTP(distill2 head, n_max=2)  via kvmem
echo   ctx 262144 (256k), block 32, -ub 256, budget 32768, reserve 24576
echo   measured: prefill 367 tok/s (old 276), decode 47.7 t/s (old 45.7)
echo   panel : http://%HOST%:%PANELPORT%
echo   api   : http://%HOST%:%PORT%/v1/chat/completions
echo   log   : %LOG%
echo   stop  : Ctrl+C
echo.

if exist "%PANEL%" start "kvmem-panel" /min python "%PANEL%"
start "" "http://%HOST%:%PANELPORT%"

"%BINDIR%\llama-kvmem-server.exe" ^
    -m "%MODEL%" ^
    --host %HOST% --port %PORT% ^
    -c 262144 ^
    --kvmem-budget 32768 --kvmem-gen-reserve 24576 ^
    --kvmem-block-tokens 32 ^
    --kvmem-sink-tokens 4096 --kvmem-recent-tokens 8192 ^
    -ctk q4_0 -ctv q4_0 ^
    --flash-attn on ^
    --gpu-layers all ^
    -b 4096 -ub 256 --threads 8 ^
    --spec-type draft-mtp --spec-draft-n-max 2 ^
    --enable-thinking --reasoning-budget 4096 ^
    --chat-template-file "%TEMPLATE%" ^
    --kvmem-trace ^
    >> "%LOG%" 2>&1

echo.
echo server exited.
pause
exit /b 0
