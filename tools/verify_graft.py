#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a grafted model against the trained head in PyTorch.

draft-logits on the patched GGUF writes h.f16 / hnextn.f16 / logits.f32 for a token
sequence; this script replays the same rows through the checkpoint and compares the
MTP block output per row and the final logits.  Agreement here means the exported
(quantized) weights in the GGUF are the trained ones and the reference implementation
still matches.

usage: verify_graft.py --val DIR --ckpt head_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_head import load_head_from_gguf   # noqa: E402


def cos(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="draft-logits output dir of the patched model")
    ap.add_argument("--ckpt", required=True, help="trained head checkpoint")
    ap.add_argument("--gguf", default="D:/models/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf")
    a = ap.parse_args()

    v = Path(a.val)
    h_all = torch.from_numpy(np.fromfile(v / "h.f16", dtype=np.float16).reshape(-1, 5120))
    tok = np.fromfile(v / "tok.i32", dtype=np.int32).astype(np.int64)
    refh = np.fromfile(v / "hnextn.f16", dtype=np.float16).reshape(-1, 5120).astype(np.float32)
    ref = np.fromfile(v / "logits.f32", dtype=np.float32)

    head = load_head_from_gguf(a.gguf, device="cpu")
    sd = torch.load(a.ckpt, map_location="cpu")
    state = {k.split(".", 1)[1] if k.startswith("head.") else k: v for k, v in sd.items()}
    missing = []
    for n, p in head.named_parameters():
        if n in state:
            p.data.copy_(state[n].to(torch.float16))
        else:
            missing.append(n)
    print(f"[verify] checkpoint {a.ckpt}: {len(state)} tensors, missing {missing or 'none'}")

    L = len(tok)
    x = torch.from_numpy(tok[1:L]); hh = h_all[:L - 1]
    pos = torch.arange(1, L, dtype=torch.float32)
    with torch.no_grad():
        hout = head.forward_hidden(x, hh, pos).numpy()
        lg = head.forward_train(x, hh, pos).numpy()

    n = min(len(refh), hout.shape[0])
    cs = [cos(hout[i], refh[i]) for i in range(n)]
    print(f"[verify] block output per-row cos: min {min(cs):.6f} mean {sum(cs)/len(cs):.6f}")
    print(f"[verify] logits cos {cos(lg[-1], ref):.6f}  "
          f"top1 {int(np.argmax(lg[-1]))} vs {int(np.argmax(ref))}  "
          f"match {int(np.argmax(lg[-1])) == int(np.argmax(ref))}")
    print(f"[verify] top10 overlap "
          f"{len(set(np.argsort(-lg[-1])[:10].tolist()) & set(np.argsort(-ref)[:10].tolist()))}/10")


if __name__ == "__main__":
    main()
