#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the MTP head (fp16-native) on dumped hidden states.

Data: mtp-dump output dirs (manifest.jsonl + chunks/h_*.f16 + chunks/tok_*.i32).
Each chunk holds 2048 rows of (token id, post-final-norm trunk hidden).  The MTP row
for position q pairs token x_q with h_{q-1} and scores x_{q+1} (see
common/speculative.cpp draft-mtp), so a training window of SEQ rows is a one-position
shift of the dumped arrays.  Windows are trained at random absolute position offsets
so the head stays robust at long-session positions (the dump itself is chunk-local).

Precision policy (per user): weights are fp16 throughout (module params fp16, updated
in place).  fp32 master weights + AdamW moments live on the CPU (the frozen 2.5 GB
output table and the head already fill an 8 GB card).  On CUDA the matmuls run fp16
with fp32 accumulation (tensor cores); the vocab-sized logits are projected in chunks
and the loss is accumulated per chunk so the graph never holds [N, V] at once.
Optional QAT: round-trip each weight through its deployment quantization type
(Q6_K big matrices / Q8_0 k,v) so the trained solution survives the real export.

Usage:
  python train_mtp.py --data D:/models/mtp-corpus/dump-sessions \
                      --data D:/models/mtp-corpus/dump-supplement \
                      --out D:/models/mtp-train/run1 \
                      [--seq 512] [--batch 2] [--epochs 2] [--lr 1e-4] [--qat deploy] \
                      [--val-frac 0.02] [--limit-chunks N]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mtp_head import load_head_from_gguf

GGUF = "D:/models/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"


# ---------------------------------------------------------------------------
# QAT: round-trip each weight through its DEPLOYMENT quantization type
# (the GGUF ships blk.64 big matrices as Q6_K, k/v as Q8_0, norms as F32)
# ---------------------------------------------------------------------------

@torch.no_grad()
def q8_0_roundtrip(w: torch.Tensor) -> torch.Tensor:
    x = w.float().reshape(-1, 32)
    amax = x.abs().amax(dim=1, keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-8)
    scale = scale.to(torch.float16).to(torch.float32)        # the block scale is stored as fp16
    q = torch.round(x / scale).clamp_(-127, 127)
    return (q * scale).reshape(w.shape).to(w.dtype)


@torch.no_grad()
def q6_k_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Q6_K-like: 256-element superblocks, 16-element sub-blocks with a 6-bit signed
    sub-scale relative to a shared fp16 super-scale, weights in [-32, 31]."""
    x = w.float()
    xb = x.reshape(-1, 256, 16)                              # [nblk, 16 subblocks, 16 w]
    amax = xb.abs().amax(dim=-1, keepdim=True)               # per-subblock amax
    sub = (amax / 31.0).clamp_min(1e-8)                      # ideal subblock scale
    d = sub.amax(dim=1, keepdim=True) / 127.0                # shared super-scale
    sc = torch.round(sub / d).clamp_(-127, 127).clamp_min(1) # 6-bit signed sub-scales
    eff = (d * sc).to(torch.float16).to(torch.float32)       # stored as fp16 d * int8 scale
    q = torch.round(xb / eff).clamp_(-32, 31)
    return (q * eff).reshape(w.shape).to(w.dtype)


QAT_TYPE = {                       # deployment type per parameter name
    "eh_proj": "q6k", "wq": "q6k", "wo": "q6k",
    "ffn_gate": "q6k", "ffn_up": "q6k", "ffn_down": "q6k",
    "wk": "q8", "wv": "q8",
}


@torch.no_grad()
def apply_qat(head) -> None:
    """Quantize-dequantize head weights in place (kept fp16), matching the GGUF types."""
    for name, p in head.named_parameters():
        t = QAT_TYPE.get(name.split(".")[0])
        if t is None or p.dim() < 2:
            continue
        p.data.copy_(q8_0_roundtrip(p.data) if t == "q8" else q6_k_roundtrip(p.data))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ChunkData:
    def __init__(self, dirs: list[str], val_frac: float, limit: int = 0, min_rows: int = 0):
        self.items = []          # (dir, h_path, tok_path, n)
        for d in dirs:
            d = Path(d)
            man = d / "manifest.jsonl"
            if not man.exists():
                print(f"  [warn] no manifest in {d}")
                continue
            for line in man.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue        # the dump appends to the manifest while it runs
                self.items.append((str(d), r["h"], r["tok"], int(r["n"])))
        self.items.sort()
        # session tails can be as short as 32 rows; a window needs seq+2 of them
        if min_rows:
            self.items = [it for it in self.items if it[3] >= min_rows]
        if limit:
            self.items = self.items[:limit]
        # A dir may be passed more than once to up-weight it.  Split train/val on the
        # UNIQUE chunks first, then drop the val chunks from the (duplicated) train
        # list -- otherwise a chunk's twin could sit in train while it is scored in val.
        uniq = sorted(set(self.items))
        n_val = max(1, int(len(uniq) * val_frac))
        self.val = uniq[:n_val]
        val_set = set(self.val)
        self.train = [it for it in self.items if it not in val_set]
        print(f"  chunks: {len(self.items)} (train {len(self.train)}, val {len(self.val)})")

    def load(self, item):
        d, h, t, n = item
        hv = np.fromfile(Path(d) / h, dtype=np.float16).reshape(n, 5120)
        tv = np.fromfile(Path(d) / t, dtype=np.int32)
        return torch.from_numpy(hv), torch.from_numpy(tv.astype(np.int64))


def batch_windows(chunks, seq: int, batch: int, rng: np.random.Generator, device):
    """Sample `batch` windows of `seq` MTP rows.

    MTP row semantics (see common/speculative.cpp draft-mtp + graph_mtp): the row at
    position q pairs token x_q with the trunk hidden of position q-1 and predicts
    x_{q+1}.  So a window of `seq` rows starting after token index s is
        x   = tok[s+1 .. s+seq]      (row tokens, at positions s+1 .. s+seq)
        h   = h[s   .. s+seq-1]      (hidden of the position before each row)
        pos = s+1 .. s+seq
        y   = tok[s+2 .. s+seq+1]    (the token each row scores)
    """
    xs, hs, ps, ys = [], [], [], []
    # The dump decodes each chunk with LOCAL positions 0..2047, but a real session
    # reaches 100k+; rope phases at large absolute positions differ, so train every
    # window at a random absolute offset to keep the head position-robust.
    OFFSETS = (0, 512, 2048, 8192, 32768, 131072)
    for _ in range(batch):
        hv, tv = chunks.load(chunks.train[rng.integers(len(chunks.train))])
        L = hv.shape[0]
        s = int(rng.integers(0, max(1, L - seq - 2)))
        hw = hv[s:s + seq].to(device)                # h of positions s .. s+seq-1
        tw = tv[s:s + seq + 2]                       # +2 for the last target
        xw = tw[1:seq + 1].to(device)                # tokens of positions s+1 .. s+seq
        yw = tw[2:seq + 2].to(device)                # the next token of each row
        off = int(OFFSETS[rng.integers(len(OFFSETS))])
        pw = torch.arange(s + 1, s + seq + 1, device=device, dtype=torch.float32) + off
        xs.append(xw); hs.append(hw); ps.append(pw); ys.append(yw)
    return torch.stack(xs), torch.stack(hs), torch.stack(ps), torch.stack(ys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(head, data, dev, n_chunks: int = 8) -> tuple[float, float]:
    """Draft top-1 agreement on held-out chunks, at near and far absolute positions.

    Returns (near %, far %).  The far pass checks the rope-phase robustness the
    deployment needs at long-session positions."""
    tot = hit = hit_far = 0
    for item in data.val[:n_chunks]:
        hv, tv = data.load(item)
        hv = hv.to(dev); tv = tv.to(dev)
        L = min(512, hv.shape[0] - 2)
        # rows at positions 1..L: token tok[1..L] + h[0..L-1] -> score tok[2..L+1]
        for off in (0.0, 131072.0):
            hout = head.forward_hidden(tv[1:L + 1], hv[:L],
                                       torch.arange(1, L + 1, device=dev, dtype=torch.float32) + off)
            preds = []
            for i in range(0, L, 256):
                preds.append(head.lm_logits(hout[i:i + 256]).argmax(-1))
            ok = int((torch.cat(preds) == tv[2:L + 2]).sum())
            if off == 0.0:
                hit += ok; tot += L
            else:
                hit_far += ok
    return 100.0 * hit / max(1, tot), 100.0 * hit_far / max(1, tot)


def save_ckpt(head, path, keep: int = 4) -> None:
    """Save only the trainable head parameters (~0.85 GB), never the frozen trunk
    tables (tok_embd / out_head are 2.5 GB each and reload from the GGUF), and keep
    at most `keep` step checkpoints so a long run cannot fill the disk."""
    sd = {k: v.data.cpu() for k, v in head.state_dict().items()
          if not k.startswith(("tok_embd", "out_head", "had_rot", "had_signs"))}
    torch.save(sd, path)
    olds = sorted(path.parent.glob("head_step*.pt"),
                  key=lambda q: int(q.stem.split("step")[1]))
    for old in olds[:-keep]:
        old.unlink()


def batch_windows_distill(chunks, seq: int, batch: int, rng: np.random.Generator, device,
                          steps: int = 3):
    """Windows for the n_max=steps distillation objective.

    A window is `seq` MTP rows (convention A): row i sits at position p = s+1+i with the
    token tok[p] and the trunk hidden of p-1.  The draft chain needs the tokens of the
    deeper rows too, and every target is the TRUNK's own argmax at the row's position,
    which the output head can read straight off the dumped hidden:

        x_ids [L+steps-1]  tokens   of positions s+1 .. s+seq+steps-1  (depth k row i uses
                           x_ids[i+k-1])
        h_in  [L]          trunk hidden of positions s .. s+seq-1     (depth-1 inputs)
        h_tgt [L+steps-1]  trunk hidden of positions s+1 .. s+seq+steps-1 (target rows;
                           depth k row i's target = argmax(out_head(h_tgt[i+k-1])))
        pos   [L]          positions of the depth-1 rows
    """
    xs, hs, hts, ps = [], [], [], []
    OFFSETS = (0.0, 512.0, 2048.0, 8192.0, 32768.0, 131072.0)
    need = seq + steps + 2
    items = [it for it in chunks.train if it[3] >= need]
    for _ in range(batch):
        hv, tv = chunks.load(items[rng.integers(len(items))])
        L = hv.shape[0]
        assert L >= need, L
        s = int(rng.integers(0, L - need))
        off = OFFSETS[rng.integers(len(OFFSETS))]
        xs.append(tv[s + 1:s + seq + steps].to(device))
        hs.append(hv[s:s + seq].to(device))
        hts.append(hv[s + 1:s + seq + steps].to(device))
        ps.append(torch.arange(s + 1, s + seq + 1, device=device, dtype=torch.float32) + off)
    return torch.stack(xs), torch.stack(hs), torch.stack(hts), torch.stack(ps)


@torch.no_grad()
def evaluate_distill(head, data, dev, steps: int = 3, n_chunks: int = 8, L: int = 512):
    """Acceptance proxy per draft depth: how often the head's argmax at depth k equals
    the trunk's own argmax at that row (what the spec loop accepts against)."""
    items = [it for it in data.val if it[3] >= L + steps + 2][:n_chunks]
    hit = [0] * (steps + 1)
    tot = 0
    for it in items:
        hv, tv = data.load(it)
        hv = hv.to(dev); tv = tv.to(dev)
        x = tv[1:L + steps].unsqueeze(0)
        h_in = hv[:L].unsqueeze(0)
        h_tgt = hv[1:L + steps]
        pos = torch.arange(1, L + 1, device=dev, dtype=torch.float32)
        for k in range(1, steps + 1):
            logits, _ = head.forward_chain(x[0], h_in[0], pos, depth=k)
            tgt = head.lm_logits(h_tgt[k - 1:k - 1 + L]).argmax(-1)
            hit[k] += int((logits.argmax(-1) == tgt).sum())
        tot += L
    return [100.0 * h / max(1, tot) for h in hit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--qat", choices=["none", "deploy"], default="deploy")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--init-from", dest="init_from", default="",
                    help="start from a saved head checkpoint (crash recovery)")
    ap.add_argument("--objective", choices=["text", "distill"], default="text",
                    help="text = corpus next-token CE (depth 1 only); "
                         "distill = trunk-argmax distillation over an n_max=steps chain")
    ap.add_argument("--steps", type=int, default=3,
                    help="draft-chain depth for --objective distill (deployment n_max)")
    ap.add_argument("--accum", type=int, default=4,
                    help="micro-batches per optimizer step (amortizes the CPU "
                         "optimizer + fp16 sync, the dominant per-step cost)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True     # attention/norm matmuls run fp32

    print("[train] loading head …")
    head = load_head_from_gguf(a.gguf, device=a.device)
    if a.init_from:
        sd = torch.load(a.init_from, map_location="cpu")
        state = {k.split(".", 1)[1] if k.startswith("head.") else k: v for k, v in sd.items()}
        for n, p in head.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(dtype=torch.float16, device=p.device))
        print(f"[train] initialized from {a.init_from}")
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)

    params = [p for p in head.parameters()]
    n_param = sum(p.numel() for p in params)
    print(f"[train] trainable params: {n_param/1e6:.1f} M (fp16)")

    # fp32 master weights + AdamW state live on the CPU: on an 8 GB card the frozen
    # output table (2.5 GB) plus the fp16 head and its grads already fill the device.
    shadow = [p.detach().float().cpu().clone().requires_grad_(True) for p in params]
    opt = torch.optim.AdamW(shadow, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    opt.zero_grad(set_to_none=True)

    data = ChunkData(a.data, a.val_frac, a.limit_chunks, min_rows=a.seq + 2)
    steps_per_epoch = max(1, len(data.train) * 2048 // (a.seq * a.batch))
    total_steps = int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"[train] steps/epoch≈{steps_per_epoch}, total≈{total_steps}, "
          f"tokens≈{total_steps*a.seq*a.batch/1e6:.1f}M, qat={a.qat}")

    t0 = time.time()
    run_loss, run_n = 0.0, 0
    accum = max(1, a.accum)
    n_cycles = max(1, total_steps // accum)
    for step in range(1, total_steps + 1):
        cycle_start = (step - 1) % accum == 0
        cycle_end = (step % accum == 0) or step == total_steps

        # Once per accumulation cycle: refresh the fp16 params from the fp32 masters
        # (post-optimizer state) and apply the deployment quantization.  Both are
        # whole-model passes (~0.4 s together), so they ride the cycle, not the step.
        if cycle_start:
            with torch.no_grad():
                for p, s in zip(params, shadow):
                    p.data.copy_(s.to(dtype=torch.float16, device=p.device))
                if a.qat == "deploy":
                    apply_qat(head)
            for p in params:
                p.grad = None

        # warmup then cosine decay on the optimizer cycles — the head starts from a
        # good initialization, so the tail of training should settle instead of bounce
        cyc = (step - 1) // accum + 1
        lr = a.lr * min(1.0, cyc / 50.0) * 0.5 * (1.0 + math.cos(math.pi * cyc / n_cycles))
        for g in opt.param_groups:
            g["lr"] = lr

        if a.objective == "distill":
            # n_max=steps draft chain against the trunk's own argmax: depth k>1 rows
            # take THIS head's hidden output (t_h_nextn) as their input, which is what
            # the spec loop feeds them.  The loss depth is sampled per batch item so the
            # expensive vocab projection runs once per step while the hidden chain (and
            # therefore the self-feeding regime) is always exercised.
            x, h_in, h_tgt, pos = batch_windows_distill(data, a.seq, a.batch, rng, dev, a.steps)
            L = a.seq
            # FastMTP (arXiv 2509.18362) weights draft depths with an exponential decay
            # alpha_k ~ beta^(k-1): deeper steps accumulate uncertainty, so near-term
            # predictions carry more weight while the tail still gets trained.
            w = np.array([0.6 ** (k - 1) for k in range(1, a.steps + 1)], dtype=np.float64)
            w /= w.sum()
            loss_sum = 0.0
            for b in range(x.shape[0]):
                k = int(rng.choice(np.arange(1, a.steps + 1), p=w))
                lg, _ = head.forward_chain(x[b], h_in[b], pos[b], depth=k)
                with torch.no_grad():
                    tgt = head.lm_logits(h_tgt[b, k - 1:k - 1 + L]).argmax(-1)
                lc = F.cross_entropy(lg.float(), tgt, reduction="sum")
                lc.backward()                                       # accumulates into p.grad
                loss_sum += float(lc.detach())
        else:
            x, h, pos, y = batch_windows(data, a.seq, a.batch, rng, dev)
            # Per batch item: the MTP block's attention is causal over the rows of ONE
            # window, so batching several windows into one call would let window A attend
            # window B (unrelated sessions).  One window per forward also keeps every
            # backward traversal of the head graph productive (a chunked loss would
            # re-traverse it per chunk) and bounds the [L, V] logits to one window.
            loss_sum = 0.0
            for b in range(x.shape[0]):
                hout = head.forward_hidden(x[b], h[b], pos[b])      # [L, E]
                lg = head.lm_logits(hout)                           # [L, V]
                lc = F.cross_entropy(lg.float(), y[b], reduction="sum")
                lc.backward()                                       # accumulates into p.grad
                loss_sum += float(lc.detach())
        n_rows = x.shape[0] * a.seq
        loss = loss_sum / n_rows

        if cycle_end:
            with torch.no_grad():
                for p, s in zip(params, shadow):
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float().cpu() / (accum * n_rows)
                    s.grad = g if s.grad is None else s.grad.add_(g)
                    p.grad = None
            opt.step()
            opt.zero_grad(set_to_none=True)

        run_loss += float(loss); run_n += 1
        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            tps = step * a.seq * a.batch / el
            print(f"[train] step {step}/{total_steps} loss {run_loss/run_n:.4f} "
                  f"({tps:.0f} tok/s, {el/60:.1f} min)", flush=True)
            run_loss, run_n = 0.0, 0
        if step % a.val_every == 0:
            if a.objective == "distill":
                ag = evaluate_distill(head, data, dev, a.steps)
                print("[train] step %d VAL trunk-argmax " % step
                      + " | ".join(f"depth{k}:{ag[k]:.1f}%" for k in range(1, a.steps + 1)),
                      flush=True)
            else:
                vn, vf = evaluate(head, data, dev)
                print(f"[train] step {step} VAL agreement {vn:.1f}% (pos 0) | {vf:.1f}% (pos 131072)", flush=True)
        if step % 500 == 0 or step == total_steps:
            ck = out / f"head_step{step}.pt"
            save_ckpt(head, ck)
            print(f"[train] saved {ck}", flush=True)

    # final validation on the deployment-like (QAT'd) weights
    head.eval()
    if a.objective == "distill":
        ag = evaluate_distill(head, data, dev, a.steps)
        print("[train] FINAL VAL trunk-argmax agreement "
              + " | ".join(f"depth{k}:{ag[k]:.1f}%" for k in range(1, a.steps + 1)))
    else:
        vn, vf = evaluate(head, data, dev)
        print(f"[train] FINAL VAL draft top-1 agreement: {vn:.1f}% (pos 0) | {vf:.1f}% (pos 131072)")
    if dev.type == "cuda":
        print(f"[train] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"(reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB)")
    save_ckpt(head, out / "head_final.pt", keep=8)
    print(f"[train] done in {(time.time()-t0)/60:.1f} min -> {out/'head_final.pt'}")


if __name__ == "__main__":
    main()
