#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a trained MTP head to raw fp16 tensor files for head-graft.

Writes one file per GGUF tensor into --out DIR:
    <ggml tensor name>.f16   little-endian fp16, ggml memory order (ne[0] fastest)

The torch parameters are [out, in] row-major, which IS ggml's layout for a weight with
ne = [in, out] (ne[0] = in varies fastest), so the export is a straight .tobytes().

usage:
  export_head.py --ckpt head_final.pt --out DIR          # trained checkpoint
  export_head.py --from-gguf model.gguf --out DIR        # round-trip the shipped head
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_head import load_head_from_gguf, MtpHead   # noqa: E402

# torch parameter -> GGUF tensor name
NAME_MAP = {
    "eh_proj":        "blk.64.nextn.eh_proj.weight",
    "enorm":          "blk.64.nextn.enorm.weight",
    "hnorm":          "blk.64.nextn.hnorm.weight",
    "attn_norm":      "blk.64.attn_norm.weight",
    "q_norm":         "blk.64.attn_q_norm.weight",
    "k_norm":         "blk.64.attn_k_norm.weight",
    "post_attn_norm": "blk.64.post_attention_norm.weight",
    "wq":             "blk.64.attn_q.weight",
    "wk":             "blk.64.attn_k.weight",
    "wv":             "blk.64.attn_v.weight",
    "wo":             "blk.64.attn_output.weight",
    "ffn_gate":       "blk.64.ffn_gate.weight",
    "ffn_up":         "blk.64.ffn_up.weight",
    "ffn_down":       "blk.64.ffn_down.weight",
    "head_norm":      "blk.64.nextn.shared_head_norm.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="trained head_final.pt")
    ap.add_argument("--from-gguf", dest="from_gguf", help="export the head currently in a GGUF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default="D:/models/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf")
    a = ap.parse_args()
    if not (a.ckpt or a.from_gguf):
        ap.error("need --ckpt or --from-gguf")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        state = {k.split(".", 1)[1] if k.startswith("head.") else k: v for k, v in sd.items()}
        print(f"[export] checkpoint {a.ckpt}: {len(state)} tensors")
    else:
        head = load_head_from_gguf(a.from_gguf, device="cpu")
        state = {n: p.data for n, p in head.named_parameters()}
        print(f"[export] from {a.from_gguf}: {len(state)} tensors")

    n_bytes = 0
    for pname, tname in NAME_MAP.items():
        if pname not in state:
            print(f"  [warn] {pname} missing in checkpoint")
            continue
        w = state[pname].detach().cpu().float().numpy()
        h = w.astype(np.float16)
        (out / f"{tname}.f16").write_bytes(h.tobytes())
        n_bytes += h.nbytes
        print(f"  {tname:42s} {str(tuple(w.shape)):16s} {h.nbytes/1e6:8.2f} MB")
    print(f"[export] wrote {n_bytes/1e6:.1f} MB into {out}")


if __name__ == "__main__":
    main()
