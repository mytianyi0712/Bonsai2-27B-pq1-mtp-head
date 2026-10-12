#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge mtp-trace chunks that belong to the same prompt.

mtp-trace flushes a chunk every --chunk rows AND at every prompt boundary, so with
--max-new 700 --chunk 512 a prompt lands as a 512-row chunk plus a 188-row chunk --
both below train_mtp.py's minimum window (seq+2).  The manifest records the source
prompt id, so the pieces can be concatenated back into one chunk per prompt, which is
what the trainer wants (a window must stay inside one context anyway).

usage: mtp-trace-merge.py --src DIR [--out DIR2]     (in place when --out is omitted)
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--min-rows", type=int, default=520)
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else src
    if out != src:
        if out.exists():
            shutil.rmtree(out)
        (out / "chunks").mkdir(parents=True)
    tmp = out / "chunks_merged"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    groups: "OrderedDict[str, list[tuple[str, str, int]]]" = OrderedDict()
    for line in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        groups.setdefault(r.get("src", "?"), []).append((r["h"], r["tok"], int(r["n"])))

    records = []
    kept = dropped = kept_rows = 0
    for i, (gid, parts) in enumerate(groups.items()):
        hs, ts = [], []
        for h, t, n in parts:
            hs.append(np.fromfile(src / h, dtype=np.uint16))
            ts.append(np.fromfile(src / t, dtype=np.int32))
        h = np.concatenate(hs) if len(hs) > 1 else hs[0]
        t = np.concatenate(ts) if len(ts) > 1 else ts[0]
        n = t.size
        if n < a.min_rows:
            dropped += 1
            continue
        hn = f"chunks/h_{i:07d}.f16"
        tn = f"chunks/tok_{i:07d}.i32"
        h.tofile(tmp / Path(hn).name)
        t.tofile(tmp / Path(tn).name)
        records.append({"h": hn, "tok": tn, "n": int(n), "src": gid})
        kept += 1
        kept_rows += int(n)

    (out / "chunks").mkdir(exist_ok=True)
    for f in tmp.iterdir():
        shutil.move(str(f), str(out / "chunks" / f.name))
    tmp.rmdir()
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"[merge] {len(groups)} prompts -> {kept} chunks, {kept_rows} rows "
          f"(dropped {dropped} prompts shorter than {a.min_rows})")
    if kept:
        ns = sorted(r["n"] for r in records)
        print(f"[merge] chunk rows: min {ns[0]} median {ns[len(ns)//2]} max {ns[-1]}")


if __name__ == "__main__":
    main()
