#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mtp-bench — measure real-task decode speed of a kvmem llama server config.

Starts its own server instance on a private port (so it never touches a server the
user is working with), replays a fixed prompt workload, and reports tokens/s plus the
speculative acceptance parsed from the server log.

The decode speed is what the MTP draft head affects, so every config is measured on
identical prompts and an identical output budget.

usage:
  mtp-bench.py --model PATH --n-max 3 [--tag NAME]
               [--port 18201] [--max-new 384] [--prompts FILE] [--limit N]
               [--ctk q4_0] [--ctv q4_0] [--no-thinking] [--keep]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("D:/models")
BIN = ROOT / "kvmem-llama.cpp" / "build-win" / "bin" / "llama-kvmem-server.exe"
TEMPLATE = ROOT / "bonsai-chat-template.jinja"
LOG = ROOT / "mtp-bench-server.log"


def wait_ready(port: int, log: Path, timeout: float = 420.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if log.exists() and "listening on" in log.read_text(encoding="utf-8", errors="replace")[-4000:]:
            time.sleep(1.0)
        time.sleep(2.0)
    return False


def chat(port: int, prompt: str, max_new: int, thinking: bool) -> dict:
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new,
        "temperature": 0,
        "reasoning_effort": "high" if thinking else "none",
    }
    # Streamed so the prefill (time to first token) is separable from decode: with a
    # 50k-token agent session the prefill dominates the request wall time and would
    # otherwise hide the draft head's effect entirely.
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    t_first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ch = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.time()
    t_end = time.time()
    comp = usage.get("completion_tokens", 0)
    prefill_s = (t_first - t0) if t_first else (t_end - t0)
    decode_s = max(0.001, t_end - (t_first or t_end))
    return {
        "wall_s": t_end - t0,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "completion_tokens": comp,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "decode_tps": comp / decode_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=18201)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--prompts", default=str(ROOT / "mtp-corpus" / "ab-prompt-1.txt"))
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--b", type=int, default=4096, help="logical batch (-b)")
    ap.add_argument("--ub", type=int, default=512, help="physical ubatch (-ub)")
    ap.add_argument("--ctk", default="q4_0")
    ap.add_argument("--ctv", default="q4_0")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--long-chars", type=int, default=0,
                    help="append one long-context request of this many chars (~1 token/3.5 ch)")
    ap.add_argument("--keep", action="store_true", help="leave the server running")
    a = ap.parse_args()

    raw = Path(a.prompts).read_text(encoding="utf-8", errors="replace")
    # split into `limit` chunks of roughly equal size (each becomes one request)
    step = max(1, len(raw) // a.limit)
    prompts = [raw[i * step:(i + 1) * step] for i in range(a.limit)]
    prompts = [p for p in prompts if len(p) > 200]
    if a.long_chars:
        # one long-context request (the deployment's real shape: a big agent session)
        big = (raw * (a.long_chars // max(1, len(raw)) + 1))[:a.long_chars]
        prompts.append(big)

    if LOG.exists():
        LOG.unlink()
    # server args mirror the user's deployment launcher (start-bonsai-2-27b*.bat)
    # exactly, except the model path and n_max -- otherwise the measured t/s does not
    # transfer to the real workload.
    args = [
        str(BIN), "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port),
        "-c", "262144", "--kvmem-budget", "24576", "--kvmem-gen-reserve", "32768",
        "-ctk", a.ctk, "-ctv", a.ctv, "--flash-attn", "on", "--gpu-layers", "all",
        "-b", str(a.b), "-ub", str(a.ub), "--threads", "8",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(a.n_max),
        "--enable-thinking", "--reasoning-budget", "4096",
        "--chat-template-file", str(TEMPLATE), "--kvmem-trace",
    ]
    if a.no_thinking:
        args.remove("--enable-thinking")

    tag = a.tag or f"{Path(a.model).stem} n_max={a.n_max}"
    print(f"[bench] {tag}: starting server on :{a.port} …", flush=True)
    with LOG.open("wb") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=lf)
    try:
        if not wait_ready(a.port, LOG):
            print("[bench] server never became ready; log tail:")
            print(LOG.read_text(encoding="utf-8", errors="replace")[-2000:])
            return 1
        print(f"[bench] ready; {len(prompts)} requests x {a.max_new} tokens", flush=True)

        tot_tok = tot_dec = tot_pre = 0.0
        for i, p in enumerate(prompts):
            r = chat(a.port, p, a.max_new, not a.no_thinking)
            tot_tok += r["completion_tokens"]
            tot_dec += r["decode_s"]
            tot_pre += r["prefill_s"]
            print(f"  req {i+1}: {r['completion_tokens']:4d} tok | prefill {r['prefill_s']:6.2f}s "
                  f"({r['prompt_tokens']} tok, {r['prompt_tokens']/max(0.01,r['prefill_s']):5.0f} tok/s) "
                  f"| decode {r['decode_s']:6.2f}s = {r['decode_tps']:5.1f} t/s", flush=True)

        log_txt = LOG.read_text(encoding="utf-8", errors="replace")
        stats = re.findall(r"spec_stats n_gen=(\d+) n_drafted=(\d+) n_accept=(\d+).*?accept_pct=([\d.]+)", log_txt)
        gen = sum(int(s[0]) for s in stats)
        drf = sum(int(s[1]) for s in stats)
        acc = sum(int(s[2]) for s in stats)
        # acceptance length tau = mean tokens produced per target forward pass
        # (FastMTP's headline metric: it, not the per-token rate, drives the speed)
        cycles = len(re.findall(r"spec_verify n_draft=", log_txt))
        print(f"\n[bench] {tag}")
        print(f"  prefill: {tot_pre:.1f}s total | decode: {tot_tok:.0f} tok / {tot_dec:.1f}s "
              f"= {tot_tok/max(0.01,tot_dec):.1f} t/s (decode-only)")
        if drf:
            print(f"  spec:   drafted {drf}, accepted {acc} -> {100.0*acc/drf:.1f}% "
                  f"({drf/max(1,gen):.2f} drafts/token)")
        if cycles:
            print(f"  tau:    {gen/max(1,cycles):.2f} tokens per verify pass "
                  f"({cycles} verify passes, {gen} tokens)")
        return 0
    finally:
        if not a.keep:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[bench] server stopped")


if __name__ == "__main__":
    sys.exit(main())
