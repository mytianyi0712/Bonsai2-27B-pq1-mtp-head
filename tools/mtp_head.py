#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MTP head (qwen35 nextn) — PyTorch reference + fp16-native trainer implementation.

Replicates llama.cpp `llama_model_qwen35::graph_mtp` exactly (verified against the
DECODER_MTP draft context logits):

  inputs per row:  token id x_t  +  h_t (post-final-norm trunk hidden at t)
  predict:         x_{t+1}
  rope positions:  KV rows at pos t, query row at pos t+1 (matches draft-mtp batch)

  forward:  e = inv_hadamard(tok_embd[x])            # rot(H) first, then signs
            z = eh_proj(cat(rms(e, enorm), rms(h, hnorm)))
            za = rms(z, attn_norm)
            qg = wq(za)  -> per-head interleaved [q(256) | gate(256)] x 24 heads
            q = rms_heads(q, attn_q_norm); k = rms_heads(wk(za), attn_k_norm); v = wv(za)
            q,k = neox_rope(first 64 dims, base 1e7, pos)
            attn = softmax(q k^T / 16) v  (GQA 24/4, causal)
            cur = wo(attn * sigmoid(gate)) + z
            cur = cur + ffn_swiglu(rms(cur, post_attn_norm))
            h_out = rms(cur, shared_head_norm)
            logits = output @ h_out

All head weights are fp16-native (params stored/updated in fp16, fp32 accumulation).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GGUF quantization dequantization (exact ports of ggml-quants.c / quants.c)
# ---------------------------------------------------------------------------

def dequant_q8_0(raw: bytes, shape) -> np.ndarray:
    """block_q8_0 = {fp16 d; int8 qs[32]} — 34 bytes / 32 weights."""
    n = int(np.prod(shape))
    nb = n // 32
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = blk[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (d * qs).reshape(shape)


def dequant_q6_K(raw: bytes, shape) -> np.ndarray:
    """block_q6_K = {u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d} — 210 B / 256 w."""
    n = int(np.prod(shape))
    nb = n // 256
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 210)
    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blk[:, 208:210].copy().view(np.float16).astype(np.float32)

    out = np.empty((nb, 256), dtype=np.float32)
    for half in range(2):  # two 128-wide groups
        o = half * 128
        ql_h, qh_h, sc_h = ql[:, half * 64:(half + 1) * 64], qh[:, half * 32:(half + 1) * 32], sc[:, half * 8:(half + 1) * 8]
        l = np.arange(32)
        is_ = l // 16
        q1 = ((ql_h[:, l] & 0xF) | (((qh_h[:, l] >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql_h[:, l + 32] & 0xF) | (((qh_h[:, l] >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql_h[:, l] >> 4) | (((qh_h[:, l] >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql_h[:, l + 32] >> 4) | (((qh_h[:, l] >> 6) & 3) << 4)).astype(np.int16) - 32
        dd = d.reshape(nb, 1)
        out[:, o + l + 0]  = dd * sc_h[:, is_ + 0] * q1
        out[:, o + l + 32] = dd * sc_h[:, is_ + 2] * q2
        out[:, o + l + 64] = dd * sc_h[:, is_ + 4] * q3
        out[:, o + l + 96] = dd * sc_h[:, is_ + 6] * q4
    return out.reshape(shape)


_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint32)


def dequant_ptq1_0(raw: bytes, shape) -> np.ndarray:
    """block_ptq1_0 = {u8 qs[24] (120 trits, 5/byte); u8 qh[2] (8 trits, 4/byte); fp16 d}
    Exact port of the CPU reference traversal (dequantize/vec_dot share it)."""
    n = int(np.prod(shape))
    nb = n // 128
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 28)   # qs24 + qh2 + fp16 d
    qs = blk[:, 0:24]
    qh = blk[:, 24:26]
    d = blk[:, 26:28].copy().view(np.float16).astype(np.float32).reshape(nb, 1)

    q = np.empty((nb, 128), dtype=np.int16)
    o, j = 0, 0
    for c in (32, 16, 8):                      # stages: chunks of qs bytes
        while j + c <= 24:
            chunk = qs[:, j:j + c].astype(np.uint32)
            for nn in range(5):
                v = (chunk * _POW3[nn]) & 0xFF  # uint8 wrap, as in the C code
                xi = (v * 3) >> 8
                q[:, o:o + c] = xi.astype(np.int16) - 1
                o += c
            j += c
    for nn in range(4):                        # qh: 4 planes x 2 bytes
        v = (qh.astype(np.uint32) * _POW3[nn]) & 0xFF
        xi = (v * 3) >> 8
        q[:, o:o + 2] = xi.astype(np.int16) - 1
        o += 2
    assert o == 128, o
    return (q.astype(np.float32) * d).reshape(shape)


def dequant_f32(raw: bytes, shape) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


DEQUANT = {"Q8_0": dequant_q8_0, "Q6_K": dequant_q6_K, "PTQ1_0": dequant_ptq1_0, "F32": dequant_f32}


# ---------------------------------------------------------------------------
# Hadamard inverse transform (prism.hadamard, block 1024, normalized SW-Hadamard)
# ---------------------------------------------------------------------------

def gen_hadamard(n: int) -> np.ndarray:
    """Orthonormal Sylvester-Walsh-Hadamard, exact port of ggml_gen_hadamard."""
    H = np.zeros((n, n), dtype=np.float32)
    H[0, 0] = 1.0 / math.sqrt(n)
    s = 1
    while s < n:
        for i in range(s):
            for j in range(s):
                v = H[i, j]
                H[i + s, j] = v
                H[i, j + s] = v
                H[i + s, j + s] = -v
        s *= 2
    return H


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class MtpMeta:
    n_embd: int = 5120
    n_head: int = 24
    n_head_kv: int = 4
    head_dim: int = 256
    n_rot: int = 64
    rope_base: float = 1.0e7
    eps: float = 1.0e-6
    hadamard_block: int = 1024
    ffn_len: int = 17408


class MtpHead(torch.nn.Module):
    """fp16-native MTP head. All parameters stored in fp16."""

    def __init__(self, meta: MtpMeta, device="cuda"):
        super().__init__()
        self.m = meta
        self.device = torch.device(device)
        e, hd, nh, nkv = meta.n_embd, meta.head_dim, meta.n_head, meta.n_head_kv

        def p(*shape):
            return torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float16, device=self.device))

        self.eh_proj   = p(e, 2 * e)          # [out=e, in=2e]
        self.enorm     = p(e)
        self.hnorm     = p(e)
        self.attn_norm = p(e)
        self.q_norm    = p(hd)
        self.k_norm    = p(hd)
        self.post_attn_norm = p(e)
        self.wq        = p(nh * 2 * hd, e)    # interleaved q|gate
        self.wk        = p(nkv * hd, e)
        self.wv        = p(nkv * hd, e)
        self.wo        = p(e, nh * hd)
        self.ffn_gate  = p(meta.ffn_len, e)
        self.ffn_up    = p(meta.ffn_len, e)
        self.ffn_down  = p(e, meta.ffn_len)
        self.head_norm = p(e)

        # frozen shared tables (dequantized trunk tensors), fp16
        self.register_buffer("tok_embd", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("out_head", torch.zeros(0, dtype=torch.float16))
        # hadamard inverse
        self.register_buffer("had_rot", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("had_signs", torch.zeros(0, dtype=torch.float32))
        # rope cache (rebuilt per call, kept for reuse)
        self._rope_cache: dict = {}

    # ---- forward helpers ----
    # Numerics contract: weights are fp16 (params), activations fp32 on the CPU and
    # fp16 on CUDA (tensor cores, fp32 accumulate) — the same mix llama.cpp uses for
    # the MTP graph.  `_mm` is the single place that decides the compute dtype: casting
    # the *weights* to fp32 instead would allocate a second full copy of every weight
    # (5 GB for the output table alone) and blow the 8 GB card.
    def _mm(self, x, w):
        """y = x @ w^T with the deployment dtype policy."""
        if x.device.type == "cuda":
            return F.linear(x.to(torch.float16), w)
        return F.linear(x, w.float())

    def _rms(self, x, w):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.m.eps) * w.float()

    def _inv_hadamard(self, x):
        # llama_mul_mat_hadamard: reshape to [*, 1024] then matmul with H (rot), then signs
        if self.had_rot.numel() == 0:
            return x.float()
        b = self.m.hadamard_block
        xf = x.float()
        shape = xf.shape
        v = xf.reshape(-1, b)
        v = v @ self.had_rot                      # H symmetric: x @ H^T == H @ x
        v = v.reshape(shape)                      # back to the full width, then signs
        return v * self.had_signs

    def _rope(self, x, pos):
        """NeoX rope over first n_rot dims (pairs i, i+n_rot/2); rest passthrough.
        x: [*, L, H, D] fp32; pos: [L] float tensor (positions per token)."""
        n_rot = self.m.n_rot
        if n_rot == 0:
            return x
        D = x.shape[-1]
        half = n_rot // 2
        idx = torch.arange(half, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.m.rope_base ** (2.0 * idx / n_rot))     # theta_scale^j
        theta = pos.float()[:, None] * inv_freq[None, :]               # [L, half]
        extra = (1,) * (x.dim() - 2)
        cos = torch.cos(theta).view(theta.shape[0], *extra, half)
        sin = torch.sin(theta).view(theta.shape[0], *extra, half)
        xf = x.float()
        xr = xf[..., :n_rot]
        x1 = xr[..., :half]
        x2 = xr[..., half:n_rot]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.cat([o1, o2, xf[..., n_rot:]], dim=-1)
        return out

    def _block_qkv(self, z_q, pos_q, kv_z, pos_kv, mask):
        """One MTP block over explicit query rows and an explicit KV row set.
        z_q: [Lq, E] query rows (also the residual base); pos_q: [Lq] their positions.
        kv_z: [Lk, E] KV rows; pos_kv: [Lk]; mask: [Lq, Lk] bool, True = attend."""
        m = self.m
        nh, nkv, hd = m.n_head, m.n_head_kv, m.head_dim
        Lq, Lk = z_q.shape[0], kv_z.shape[0]

        za_q  = self._rms(z_q,  self.attn_norm)
        za_kv = self._rms(kv_z, self.attn_norm)
        qg = self._mm(za_q, self.wq)                                    # [Lq, nh*2*hd]
        qg = qg.view(Lq, nh, 2, hd)
        q = self._rms(qg[:, :, 0, :], self.q_norm)                      # [Lq, nh, hd]
        gate = qg[:, :, 1, :].reshape(Lq, nh * hd)
        k = self._rms(self._mm(za_kv, self.wk).view(Lk, nkv, hd), self.k_norm)
        v = self._mm(za_kv, self.wv).view(Lk, nkv, hd)

        q = self._rope(q, pos_q)
        k = self._rope(k, pos_kv)

        rep = nh // nkv
        qe = q.float().permute(1, 0, 2)                                 # [nh, Lq, hd]
        ke = k.float().permute(1, 0, 2).repeat_interleave(rep, dim=0)   # [nh, Lk, hd]
        ve = v.float().permute(1, 0, 2).repeat_interleave(rep, dim=0)
        scores = torch.matmul(qe, ke.transpose(-1, -2)) / math.sqrt(hd) # [nh, Lq, Lk]
        scores = scores.masked_fill(~mask[None], float("-inf"))
        o = torch.matmul(torch.softmax(scores, dim=-1), ve).permute(1, 0, 2).reshape(Lq, nh * hd)
        cur = self._mm(o * torch.sigmoid(gate), self.wo) + z_q

        ffn_in = self._rms(cur, self.post_attn_norm)
        g = self._mm(ffn_in, self.ffn_gate)
        u = self._mm(ffn_in, self.ffn_up)
        return cur + self._mm(F.silu(g) * u, self.ffn_down)

    def _block(self, z, pos_kv, pos_q):
        """One MTP transformer block over its own rows, causal (the depth-1 pass)."""
        L = z.shape[0]
        mask = torch.ones(L, L, dtype=torch.bool, device=z.device).tril()
        return self._block_qkv(z, pos_q, z, pos_kv, mask)

    def forward_chain(self, x_ids, h_in, pos, depth=3):
        """Deployment-faithful draft chain for n_max=depth (common/speculative.cpp
        draft-mtp / kvmem-spec).

        Depth 1 row i is the trunk-supported row at pos[i]: token x_ids[i] + hidden
        h_in[i] (the trunk hidden of pos[i]-1).  Depth k>1 row i sits at pos[i]+k-1 with
        the token x_ids[i+k-1] and THIS HEAD's own hidden output of row i at depth k-1
        (the draft context feeds t_h_nextn back in).  A depth-k query attends the
        depth-1 rows at positions <= its anchor position, its own chain's shallower
        rows, and itself -- never another chain's deeper rows, matching the draft
        context's KV, which only ever holds the catch-up rows plus the running chain.

        Returns (logits [L, V] at the deepest depth, hidden outputs per depth)."""
        L = h_in.shape[0]
        if x_ids.shape[0] < L + depth - 1:
            raise ValueError(f"forward_chain needs {L + depth - 1} token ids for depth "
                             f"{depth} over {L} rows, got {x_ids.shape[0]}")
        i_idx = torch.arange(L, device=h_in.device)
        own = i_idx[:, None] == i_idx[None, :]
        causal = i_idx[:, None] >= i_idx[None, :]
        zs, hs = [], []
        prev_h = h_in
        for k in range(1, depth + 1):
            xk = x_ids[k - 1:k - 1 + L]
            pk = pos + float(k - 1)
            zk = self.z_from(xk, prev_h)
            mask = torch.cat([causal] + [own] * (k - 1), dim=1)
            kv_z = torch.cat(zs + [zk], 0) if zs else zk
            pos_kv = torch.cat([pos + float(j) for j in range(k)], 0) if k > 1 else pos
            cur = self._block_qkv(zk, pk, kv_z, pos_kv, mask)
            hs.append(cur)
            zs.append(zk)
            prev_h = cur
        logits = self._lm_head(self._rms(hs[-1], self.head_norm))
        return logits, hs

    def embed(self, x_ids):
        """Token-table lookup.  The table stays on its own (CPU) device and only the
        batch's rows travel to the compute device: the frozen trunk tables are far too
        large to keep resident alongside the trainable head on an 8 GB card."""
        return self.tok_embd[x_ids.to(self.tok_embd.device)].to(self._compute_device()).float()

    def _compute_device(self):
        if self.had_rot.numel():
            return self.had_rot.device
        return next(self.parameters()).device

    def z_from(self, x_ids, h):
        """x_ids: [L] int64; h: [L, E] fp16 -> z [L, E] (post eh_proj)."""
        e = self.embed(x_ids)                                           # [L, E]
        e = self._inv_hadamard(e)
        e_n = self._rms(e, self.enorm)
        h_n = self._rms(h, self.hnorm)
        cat = torch.cat([e_n, h_n], dim=-1)                             # e first, h second
        return self._mm(cat, self.eh_proj)

    def _fwd_hadamard(self, x):
        """Forward/folded-weight activation transform (build_lora_mm): signs first, then blockwise H."""
        if self.had_rot.numel() == 0:
            return x
        b = self.m.hadamard_block
        xf = x.float() * self.had_signs            # signs first
        shape = xf.shape
        v = xf.reshape(-1, b) @ self.had_rot       # then rotation
        return v.reshape(shape)

    def _lm_head(self, h_out):
        return self._mm(self._fwd_hadamard(h_out), self.out_head)

    def forward_hidden(self, x_ids, h, pos):
        """Pre-head hidden of every MTP row (see forward_train).  Kept separate so the
        trainer can project the vocab-sized logits in chunks instead of materialising
        [n_rows, n_vocab] at once."""
        z = self.z_from(x_ids, h)
        cur = self._block(z, pos_kv=pos, pos_q=pos)
        return self._rms(cur, self.head_norm)

    def lm_logits(self, h_out):
        return self._lm_head(h_out)

    def forward_train(self, x_ids, h, pos):
        """Parallel training pass over a chunk of MTP rows (real draft-flow semantics).

        Row i is the MTP row at absolute position pos[i]: it pairs token x_ids[i] (the
        token AT that position) with h[i] (the trunk hidden of position pos[i]-1) and
        predicts the token at pos[i]+1.  Both the KV row and the query row of the MTP
        block sit at pos[i] (graph_mtp ropes Q and K with the same inp_pos), and the
        attention is causal over the rows, so row i attends rows at positions <= pos[i].

        Returns logits [L, V]; row i scores the token at pos[i]+1."""
        z = self.z_from(x_ids, h)
        cur = self._block(z, pos_kv=pos, pos_q=pos)
        h_out = self._rms(cur, self.head_norm)
        return self._lm_head(h_out)

    @torch.no_grad()
    def forward_draft(self, x_ids, h, pos):
        """Inference-style draft step: logits for the LAST row only.
        The last row sits at pos[-1] with token x_ids[-1] and hidden h[-1] (from
        pos[-1]-1) and scores the token at pos[-1]+1."""
        z = self.z_from(x_ids, h)
        cur = self._block(z, pos_kv=pos, pos_q=pos)
        h_out = self._rms(cur[-1:], self.head_norm)
        return self._lm_head(h_out)


# ---------------------------------------------------------------------------
# GGUF loading
# ---------------------------------------------------------------------------

def load_head_from_gguf(gguf_path: str, device="cuda", trunk_meta: Optional[dict] = None) -> MtpHead:
    import sys
    sys.path.insert(0, "D:/models/llama-prism-src/gguf-py")
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    meta = MtpMeta()

    def tensor(name):
        for t in r.tensors:
            if t.name == name:
                qtype = None
                for k, v in {8: "Q8_0", 14: "Q6_K", 143: "PTQ1_0", 0: "F32"}.items():
                    if int(t.tensor_type) == k:
                        qtype = v
                raw = t.data.tobytes()
                # GGML ne[0] is the fastest-varying dim: flat memory is
                # element (i0..in-1) at i0 + i1*ne0.  A C-order numpy view is
                # therefore reversed(ne) with NO transpose, and that view is
                # already the [out=ne1, in=ne0] matrix F.linear wants.
                arr = DEQUANT[qtype](raw, tuple(int(d) for d in reversed(t.shape)))
                return arr
        raise KeyError(name)

    head = MtpHead(meta, device=device)

    def setp(param, name):
        arr = tensor(name)                                   # [out, in]
        param.data.copy_(torch.from_numpy(arr).to(torch.float16).to(param.device))

    setp(head.eh_proj,   "blk.64.nextn.eh_proj.weight")
    setp(head.enorm,     "blk.64.nextn.enorm.weight")
    setp(head.hnorm,     "blk.64.nextn.hnorm.weight")
    setp(head.attn_norm, "blk.64.attn_norm.weight")
    setp(head.q_norm,    "blk.64.attn_q_norm.weight")
    setp(head.k_norm,    "blk.64.attn_k_norm.weight")
    setp(head.post_attn_norm, "blk.64.post_attention_norm.weight")
    setp(head.wq,        "blk.64.attn_q.weight")
    setp(head.wk,        "blk.64.attn_k.weight")
    setp(head.wv,        "blk.64.attn_v.weight")
    setp(head.wo,        "blk.64.attn_output.weight")
    setp(head.ffn_gate,  "blk.64.ffn_gate.weight")
    setp(head.ffn_up,    "blk.64.ffn_up.weight")
    setp(head.ffn_down,  "blk.64.ffn_down.weight")
    setp(head.head_norm, "blk.64.nextn.shared_head_norm.weight")

    # frozen trunk tables: the token table stays on the CPU (rows are gathered per
    # batch); the output table must sit next to the head for the logits matmul.
    tok = tensor("token_embd.weight")                        # [V, E]
    head.tok_embd = torch.from_numpy(tok).to(torch.float16)
    out = tensor("output.weight")                            # [V, E]
    head.out_head = torch.from_numpy(out).to(torch.float16).to(head.device)

    # hadamard inverse (prism.hadamard metadata)
    f = r.fields
    block = int(np.asarray(f["prism.hadamard.block_size"].parts[f["prism.hadamard.block_size"].data[0]])[0])
    signs = []
    for i in f["prism.hadamard.sign_values"].data:
        a = np.asarray(f["prism.hadamard.sign_values"].parts[i])
        signs.extend(a.reshape(-1).tolist())
    signs = np.array(signs[: meta.n_embd], dtype=np.float32)
    H = gen_hadamard(block)
    head.had_rot = torch.from_numpy(H).to(head.device)
    head.had_signs = torch.from_numpy(signs).to(head.device)
    head.m.hadamard_block = block

    return head


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="D:/models/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    head = load_head_from_gguf(a.gguf, device=a.device)
    print("head loaded; params fp16 on", next(head.parameters()).device)
    for n, p in head.named_parameters():
        print(f"  {n:18s} {tuple(p.shape)}")
    print("tok_embd:", tuple(head.tok_embd.shape), "| out_head:", tuple(head.out_head.shape))
    print("had_rot:", tuple(head.had_rot.shape), "| signs nonzero:", int((head.had_signs != 0).sum()))
