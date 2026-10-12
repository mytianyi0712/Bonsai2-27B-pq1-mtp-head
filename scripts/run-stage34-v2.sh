#!/bin/bash
# stage 2b-4 chain (v2): merge pass-1 traces, generate a second self-distillation pass
# with the remaining prompts, merge it too, then retrain and graft.
#
# Rationale (Tencent FastMTP, arXiv 2509.18362): a single weight-shared MTP head
# fine-tuned on self-distilled data is what makes recursive multi-step drafting work;
# their corpus is 389k self-distilled samples, so more in-distribution rows is the
# main lever we still have.  Launched via `hub start` so nothing can time it out.
set -u
ROOT=/d/models
PY="$ROOT/mtp-train/.venv/Scripts/python.exe"
LOG="$ROOT/mtp-train/distill2.log"
T="$ROOT/mtp-corpus"

step() { echo "[chain] $*"; }

step "merge pass 1"
"$PY" "$ROOT/mtp-trace-merge.py" --src "$T/traces-thinking" \
      --out "$T/traces-thinking-merged" --min-rows 520 >> "$LOG" 2>&1 || exit 1

step "pass 2: self-distillation traces (remaining prompts, target 350k rows)"
# native tool must run through cmd: a login shell resets PATH and the exe then cannot
# resolve the Universal CRT DLLs (api-ms-win-crt-*.dll).
MSYS_NO_PATHCONV=1 cmd /c "D:\\models\\run-pass2.bat 350000" >> "$LOG" 2>&1 || exit 1

step "merge pass 2"
"$PY" "$ROOT/mtp-trace-merge.py" --src "$T/traces-thinking-2" \
      --out "$T/traces-thinking-2-merged" --min-rows 520 >> "$LOG" 2>&1 || exit 1

step "stage 3: retrain (distill, n_max=3 chain, decayed depth weights, lr 3e-5)"
"$PY" "$ROOT/mtp-retrain-v2.py" --stages 3 --tag distill2 --epochs 2 --lr 3e-5 \
    --base-ckpt "$ROOT/mtp-train/distill1/head_final.pt" >> "$LOG" 2>&1
rc=$?; step "stage 3 rc=$rc"; [ $rc -ne 0 ] && exit $rc

step "stage 4: export + graft + verify"
"$PY" "$ROOT/mtp-retrain-v2.py" --stages 4 --tag distill2 >> "$LOG" 2>&1
rc=$?; step "stage 4 rc=$rc"
step "DONE (log $LOG)"
exit $rc
