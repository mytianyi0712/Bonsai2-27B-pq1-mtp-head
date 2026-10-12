#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mtp-retrain-v2 — deployment-distribution retraining for the Bonsai MTP head.

Goal: fastest real-task decode speed.  The first distilled head was trained on the
raw session corpus (prefill-only teacher forcing, f16 KV, no assistant reasoning
register).  The server runs q4_0 KV, a chat template and thinking, so the head sees a
different (token, h) distribution at inference.  This pipeline closes that gap:

  stage 1  re-dump a corpus subset with the deployment's KV quantization (q4_0)
  stage 2  generate self-distillation traces with the real template + thinking + q4_0
           (mtp-trace: every generated position yields its (token, h) row)
  stage 3  retrain the head with the same trunk-argmax distillation objective over the
           traces (up-weighted by repeating the --data dir) plus the q4_0 corpus
  stage 4  graft + verify + measure decode t/s across n_max

Each stage is skippable and resumable: a stage whose output already exists is skipped.

usage:
  mtp-retrain-v2.py [--stages 1,2,3,4] [--base-ckpt PATH] [--tag NAME]
                    [--corpus-tokens 1500000] [--trace-tokens 240000] [--epochs 2]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/models")
BIN = ROOT / "kvmem-llama.cpp" / "build-win" / "bin"
PY = ROOT / "mtp-train" / ".venv" / "Scripts" / "python.exe"
BASE_MODEL = ROOT / "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
TEMPLATE = ROOT / "bonsai-chat-template.jinja"
CORPUS = ROOT / "mtp-corpus" / "dump-sessions"
SUPP = ROOT / "mtp-corpus" / "dump-supplement"


def run(name: str, args: list[str], log: Path) -> int:
    print(f"\n===== {name}\n      {' '.join(str(a) for a in args)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as lf:
        lf.write(f"\n===== {name} @ {time.strftime('%F %T')}\n")
        rc = subprocess.call([str(a) for a in args], stdout=lf, stderr=lf)
    print(f"      -> rc={rc} in {(time.time()-t0)/60:.1f} min (log {log})", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3,4")
    ap.add_argument("--base-ckpt", default=str(ROOT / "mtp-train" / "distill1" / "head_final.pt"),
                    help="head to start from (default: the current distilled head)")
    ap.add_argument("--tag", default="distill2")
    ap.add_argument("--corpus-tokens", type=int, default=1_500_000)
    ap.add_argument("--trace-tokens", type=int, default=240_000)
    ap.add_argument("--trace-max-new", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    stages = {int(x) for x in a.stages.split(",") if x.strip()}

    corpus_q4 = ROOT / "mtp-corpus" / "dump-sessions-q4"
    traces = ROOT / "mtp-corpus" / "traces-thinking"
    if (ROOT / "mtp-corpus" / "traces-thinking-merged" / "manifest.jsonl").exists():
        traces = ROOT / "mtp-corpus" / "traces-thinking-merged"   # per-prompt chunks
    out = ROOT / "mtp-train" / a.tag
    log = ROOT / "mtp-train" / f"{a.tag}.log"

    if 1 in stages:
        if (corpus_q4 / "manifest.jsonl").exists():
            print("[skip] stage 1: corpus q4_0 dump exists")
        else:
            rc = run("stage 1: corpus re-dump with q4_0 KV",
                     [BIN / "mtp-dump.exe", "-m", BASE_MODEL, "--list", ROOT / "mtp-corpus" / "filelist-sessions.txt",
                      "--out", corpus_q4, "--chunk", "2048", "--ub", "1024",
                      "--max-tokens", str(a.corpus_tokens), "--ngl", "99", "--threads", "8",
                      "--ctk", "q4_0", "--ctv", "q4_0"], log)
            if rc != 0:
                return rc

    if 2 in stages:
        man = traces / "manifest.jsonl"
        rows = 0
        if man.exists():
            for line in man.read_text(encoding="utf-8").splitlines():
                try:
                    rows += int(json.loads(line)["n"])
                except Exception:
                    pass
        if rows >= a.trace_tokens:
            print(f"[skip] stage 2: traces exist ({rows} rows)")
        else:
            rc = run("stage 2: self-distillation traces (thinking + q4_0)",
                     [BIN / "mtp-trace.exe", "-m", BASE_MODEL,
                      "--prompts", ROOT / "mtp-corpus" / "trace-prompts.txt",
                      "--out", traces, "--template-file", TEMPLATE,
                      "--max-new", str(a.trace_max_new), "--chunk", "1024",
                      "--max-rows", str(a.trace_tokens), "--ctk", "q4_0", "--ctv", "q4_0",
                      "--ngl", "99", "--threads", "8"], log)
            if rc != 0:
                return rc

    if 3 in stages:
        # traces repeated 3x in the data list -> ~3x sampling weight vs the corpus
        data: list[str] = []
        # every merged trace set (one per generation pass) gets 3x sampling weight
        merged = sorted((ROOT / "mtp-corpus").glob("traces-*-merged"))
        for mdir in merged:
            if (mdir / "manifest.jsonl").exists():
                data += ["--data", str(mdir)] * 3
                print(f"[stage3] trace data: {mdir.name}")
        if (corpus_q4 / "manifest.jsonl").exists():
            data += ["--data", str(corpus_q4)]
            print("[stage3] corpus data: dump-sessions-q4")
        if not merged:
            print("[stage3] WARNING: no merged trace dirs found")
        if not data:
            print("[error] stage 3: no data dirs")
            return 1
        rc = run("stage 3: retrain (trunk-argmax distillation, n_max=3 chain)",
                 [PY, ROOT / "mtp-train" / "train_mtp.py", *data,
                  "--out", out, "--init-from", a.base_ckpt,
                  "--device", "cuda", "--seq", "512", "--batch", "2",
                  "--epochs", str(a.epochs), "--lr", str(a.lr),
                  "--qat", "deploy", "--accum", "8", "--val-every", "500",
                  "--objective", "distill", "--steps", "3"], log)
        if rc != 0:
            return rc

    if 4 in stages:
        ckpt = out / "head_final.pt"
        if not ckpt.exists():
            print(f"[error] stage 4: {ckpt} missing")
            return 1
        patched = ROOT / f"Ternary-Bonsai-2-27B-PTQ1_0-mtp-{a.tag}.gguf"
        rc = run("stage 4a: export + graft",
                 [ROOT / "graft-head.bat", ckpt, patched], log)
        if rc != 0:
            return rc
        rc = run("stage 4b: verify graft",
                 [PY, ROOT / "mtp-train" / "verify_graft.py",
                  "--val", ROOT / "mtp-train" / "graft-verify-d1", "--ckpt", ckpt], log)
        # 4c: decode-speed sweep (needs the GPU free; run when nothing else is serving)
        print("\n[stage 4c] run the speed sweep manually once the GPU is free, e.g.:")
        for n in (1, 2, 3):
            print(f"    {PY} {ROOT/'mtp-bench.py'} --model {patched} --n-max {n} --tag '{a.tag} n={n}'")

    print(f"\n[done] tag={a.tag}  log={log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
