#!/usr/bin/env python3
"""
Graft the Qwen3.8-27B MTP head (blk.64.*) into the PrismML Ternary-Bonsai-2-27B GGUF.

Rationale (verified before writing this):
  Bonsai's RMSNorm weights match stock Qwen3.8 elementwise (cos > 0.99996), which
  is only possible if Bonsai's residual stream is in the ORIGINAL basis. The
  prism.hadamard.* rotation is applied online per-matmul as a quantisation aid,
  not as a change of residual basis. So the MTP block is copied verbatim, with no
  rotation, and is deliberately NOT added to prism.hadamard.weight_names.
"""
import struct, sys, os, urllib.request, json

SRC_MTP_URL = ("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/"
               "Qwen3.8-27B-UD-Q2_K_XL.gguf")
BASE = sys.argv[1]      # local Bonsai gguf
OUT  = sys.argv[2]      # output path

# ---------- GGUF primitives ----------
class Rd:
    def __init__(s, b): s.b = b; s.o = 0
    def raw(s, n):
        if s.o + n > len(s.b): raise EOFError("header buffer too small")
        v = s.b[s.o:s.o+n]; s.o += n; return v
    def u32(s): return struct.unpack("<I", s.raw(4))[0]
    def u64(s): return struct.unpack("<Q", s.raw(8))[0]
    def st(s):  return s.raw(s.u64()).decode("utf-8")

def rd_val(r, t):
    """Return (type, python_value) preserving enough to re-emit byte-identically."""
    if t == 8:  return (t, r.st())
    if t in (0,1,7): return (t, r.raw(1)[0])
    if t in (2,3): return (t, struct.unpack("<H", r.raw(2))[0])
    if t in (4,5): return (t, struct.unpack("<i", r.raw(4))[0])
    if t == 6:  return (t, struct.unpack("<f", r.raw(4))[0])
    if t in (10,11): return (t, struct.unpack("<q", r.raw(8))[0])
    if t == 12: return (t, struct.unpack("<d", r.raw(8))[0])
    if t == 9:
        et = r.u32(); n = r.u64()
        vals = [rd_val(r, et)[1] for _ in range(n)]
        return (t, (et, vals))
    raise ValueError(f"unknown gguf value type {t}")

def wr_str(f, s):
    b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)

def wr_val(f, t, v):
    f.write(struct.pack("<I", t))
    _wr_raw(f, t, v)

def _wr_raw(f, t, v):
    if t == 8: wr_str(f, v)
    elif t in (0,1,7): f.write(struct.pack("<B", v))
    elif t in (2,3): f.write(struct.pack("<H", v))
    elif t in (4,5): f.write(struct.pack("<i", v))
    elif t == 6: f.write(struct.pack("<f", v))
    elif t in (10,11): f.write(struct.pack("<q", v))
    elif t == 12: f.write(struct.pack("<d", v))
    elif t == 9:
        et, vals = v
        f.write(struct.pack("<I", et)); f.write(struct.pack("<Q", len(vals)))
        for x in vals: _wr_raw(f, et, x)
    else: raise ValueError(t)

def parse(buf):
    r = Rd(buf)
    assert r.raw(4) == b"GGUF", "not a GGUF"
    ver = r.u32(); nt = r.u64(); nk = r.u64()
    kv = []           # list of (key, type, value) -- ordered
    for _ in range(nk):
        k = r.st(); t = r.u32(); _, v = rd_val(r, t)
        kv.append((k, t, v))
    tl = []           # list of (name, dims, type, offset)
    for _ in range(nt):
        nm = r.st(); nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty = r.u32(); off = r.u64()
        tl.append([nm, dims, ty, off])
    return ver, kv, tl, r.o

def align_up(x, a): return (x + a - 1) // a * a

def sizes_from_offsets(tl, data_bytes_total):
    """Tensor byte size = gap to next tensor (offsets are ascending, aligned)."""
    order = sorted(range(len(tl)), key=lambda i: tl[i][3])
    out = {}
    for j, i in enumerate(order):
        off = tl[i][3]
        nxt = tl[order[j+1]][3] if j+1 < len(order) else data_bytes_total
        out[tl[i][0]] = nxt - off
    return out

def http_range(url, lo, hi, retries=5):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            return urllib.request.urlopen(req, timeout=600).read()
        except Exception as e:
            if a == retries-1: raise
            print(f"    retry {a+1}: {e}", flush=True)

def http_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])

# ---------- load headers ----------
print("[1/6] parsing Bonsai header", flush=True)
with open(BASE, "rb") as f:
    head = f.read(48*1024*1024)
bver, bkv, btl, bhdr_end = parse(head)
BALIGN = next((v for k,t,v in bkv if k == "general.alignment"), 32)
bdata_start = align_up(bhdr_end, BALIGN)
bfile = os.path.getsize(BASE)
bsz = sizes_from_offsets(btl, bfile - bdata_start)
print(f"      tensors={len(btl)} kv={len(bkv)} data_start={bdata_start} file={bfile}", flush=True)

print("[2/6] parsing unsloth header (range request)", flush=True)
uhead = http_range(SRC_MTP_URL, 0, 48*1024*1024-1)
uver, ukv, utl, uhdr_end = parse(uhead)
UALIGN = next((v for k,t,v in ukv if k == "general.alignment"), 32)
udata_start = align_up(uhdr_end, UALIGN)
ufile = http_size(SRC_MTP_URL)
usz = sizes_from_offsets(utl, ufile - udata_start)
mtp = [t for t in utl if t[0].startswith("blk.64.")]
mtp.sort(key=lambda t: t[3])
print(f"      found {len(mtp)} blk.64 tensors, "
      f"{sum(usz[t[0]] for t in mtp)/1e9:.2f} GB", flush=True)
assert len(mtp) == 15, f"expected 15 MTP tensors, got {len(mtp)}"

# ---------- build new metadata ----------
print("[3/6] building merged metadata", flush=True)
newkv = []
seen_nextn = False
for k, t, v in bkv:
    if k == "qwen35.block_count":
        print(f"      block_count {v} -> 65", flush=True); v = 65
    if k == "qwen35.nextn_predict_layers":
        v = 1; seen_nextn = True
    newkv.append((k, t, v))
if not seen_nextn:
    # insert right after block_count for tidiness; type 4 = uint32
    idx = next(i for i,(k,_,_) in enumerate(newkv) if k == "qwen35.block_count")
    newkv.insert(idx+1, ("qwen35.nextn_predict_layers", 4, 1))
    print("      added qwen35.nextn_predict_layers = 1", flush=True)

# NOTE: blk.64.* are deliberately NOT appended to prism.hadamard.weight_names.
# They are stock unrotated Qwen weights and must not get the online transform.

# ---------- layout ----------
newtl = [[nm, dims, ty, 0] for nm, dims, ty, off in btl]
for nm, dims, ty, off in mtp:
    newtl.append([nm, dims, ty, 0])

run = 0
plan = []          # (name, src, src_off, nbytes, new_off)
for i, (nm, dims, ty, _) in enumerate(newtl):
    if i < len(btl):
        n = bsz[nm]; src = ("base", btl[i][3])
    else:
        n = usz[nm]; src = ("mtp", next(t[3] for t in mtp if t[0] == nm))
    run = align_up(run, BALIGN)
    newtl[i][3] = run
    plan.append((nm, src[0], src[1], n, run))
    run += n
total_data = run
print(f"      total tensors={len(newtl)} data={total_data/1e9:.2f} GB", flush=True)

# ---------- write ----------
print(f"[4/6] writing {OUT}", flush=True)
with open(OUT, "wb") as o:
    o.write(b"GGUF"); o.write(struct.pack("<I", bver))
    o.write(struct.pack("<Q", len(newtl))); o.write(struct.pack("<Q", len(newkv)))
    for k, t, v in newkv:
        wr_str(o, k); wr_val(o, t, v)
    for nm, dims, ty, off in newtl:
        wr_str(o, nm); o.write(struct.pack("<I", len(dims)))
        for d in dims: o.write(struct.pack("<Q", d))
        o.write(struct.pack("<I", ty)); o.write(struct.pack("<Q", off))
    hdr_end = o.tell()
    data_start = align_up(hdr_end, BALIGN)
    o.write(b"\x00" * (data_start - hdr_end))
    print(f"      data_start={data_start}", flush=True)

    CH = 32*1024*1024
    with open(BASE, "rb") as bf:
        for idx, (nm, kind, soff, n, noff) in enumerate(plan):
            want = data_start + noff
            cur = o.tell()
            if cur < want: o.write(b"\x00" * (want - cur))
            assert o.tell() == want, f"offset drift at {nm}"
            if kind == "base":
                bf.seek(bdata_start + soff)
                left = n
                while left:
                    c = bf.read(min(CH, left))
                    if not c: raise IOError("short read from base")
                    o.write(c); left -= len(c)
            else:
                lo = udata_start + soff
                left = n; pos = 0
                while left:
                    c = min(CH, left)
                    o.write(http_range(SRC_MTP_URL, lo+pos, lo+pos+c-1))
                    pos += c; left -= c
                print(f"      + {nm}  {n/1e6:.1f} MB", flush=True)
            if idx % 200 == 0:
                print(f"      .. {idx}/{len(plan)}", flush=True)

print("[5/6] verifying", flush=True)
with open(OUT, "rb") as f:
    vh = f.read(48*1024*1024)
vver, vkv, vtl, vend = parse(vh)
vdata = align_up(vend, BALIGN)
vsize = os.path.getsize(OUT)
names = {t[0] for t in vtl}
bc = next(v for k,t,v in vkv if k == "qwen35.block_count")
nx = next(v for k,t,v in vkv if k == "qwen35.nextn_predict_layers")
print(f"      tensors={len(vtl)} (base {len(btl)} + 15 = {len(btl)+15})")
print(f"      block_count={bc}  nextn_predict_layers={nx}")
print(f"      blk.64 present: {sum(1 for n in names if n.startswith('blk.64.'))}/15")
print(f"      file size={vsize/1e9:.2f} GB  data_start={vdata}")
ok = (len(vtl) == len(btl)+15 and bc == 65 and nx == 1
      and sum(1 for n in names if n.startswith("blk.64.")) == 15
      and vsize >= vdata + total_data)
print("[6/6]", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
