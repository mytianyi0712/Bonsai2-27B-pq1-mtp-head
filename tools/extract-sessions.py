#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 omp 会话 JSONL 抽取 MTP 头训练语料.

用法: python extract-sessions.py [--sessions DIR] [--out DIR] [--max-chars N]
输出: 每会话一个 .txt（按时间顺序的消息文本流）+ manifest.jsonl
"""
import argparse, base64, json, os, re, sys
from pathlib import Path

SESS = Path(os.path.expanduser("~/.omp/agent/sessions"))
OUT_DEFAULT = Path("D:/models/mtp-corpus/raw")

PART_TYPES_TEXT = {"text", "reasoning", "thinking", "summary_text", "compaction"}
MAX_PART_CHARS = 24000      # 单个 part 上限（防巨型工具输出垄断）
B64_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]{400,}$")


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 去 base64 大块 / 超长无空白行
    lines = []
    for ln in text.splitlines():
        if len(ln) > 400 and B64_RE.match(ln.strip()):
            lines.append("[BASE64-DROPPED]")
        elif len(ln) > 20000:
            lines.append(ln[:20000] + " …[TRUNC]")
        else:
            lines.append(ln)
    return "\n".join(lines)[:MAX_PART_CHARS]


def extract_message(msg: dict, out: list):
    role = msg.get("role", "?")
    content = msg.get("content")
    chunks = []
    if isinstance(content, str):
        chunks.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            pt = part.get("type", "")
            if pt in PART_TYPES_TEXT:
                chunks.append(part.get("text") or part.get("summary") or "")
            elif pt == "function_call":
                name = part.get("name") or part.get("tool") or ""
                args = part.get("arguments") or part.get("input") or ""
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                chunks.append(f"[tool_call {name}] {args}")
            elif pt in ("custom", "internal"):
                # 工具结果 / 内部文本
                t = part.get("text") or part.get("content") or ""
                if isinstance(t, (dict, list)):
                    t = json.dumps(t, ensure_ascii=False)
                chunks.append(str(t))
            elif pt == "bash":
                chunks.append(str(part.get("command") or part.get("text") or ""))
            # path/url/thinking_level_change 等元数据跳过
    body = "\n".join(c for c in (clean(c) for c in chunks) if c)
    if body:
        out.append(f"### {role}\n{body}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(SESS))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--min-bytes", type=int, default=400_000,
                    help="只取超过此大小的会话文件（≈100k token 级）")
    a = ap.parse_args()

    root = Path(a.sessions)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.jsonl"), key=lambda p: -p.stat().st_size)
    big = [p for p in files if p.stat().st_size >= a.min_bytes]
    print(f"session files: {len(files)} total, {len(big)} >= {a.min_bytes//1000}KB")

    manifest = []
    total_chars = 0
    for p in big:
        out = []
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("type") == "message" and isinstance(rec.get("message"), dict):
                        extract_message(rec["message"], out)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        text = "\n".join(out)
        if len(text) < 5000:
            continue
        name = p.stem[:64].replace(" ", "_") + ".txt"
        (out_dir / name).write_text(text, encoding="utf-8")
        total_chars += len(text)
        manifest.append({"file": name, "src_bytes": p.stat().st_size, "chars": len(text)})

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"extracted {len(manifest)} sessions, {total_chars/1e6:.1f} M chars "
          f"(≈{total_chars/3.2/1e6:.1f} M tokens est.) -> {out_dir}")


if __name__ == "__main__":
    main()
