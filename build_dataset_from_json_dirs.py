#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「依資料夾命名代表類別」的 JSON 檔，轉成可訓練的 CSV（text,label）。
------------------------------------------------------------------
假設你的資料結構：
root/
  保單查詢/            # ← 資料夾名 = 類別
    a.json
    b.json
  保單查詢+註記/
    c.json
  通知函/
    d.json
  扣押命令/
    e.json
...

每個 JSON 至少包含一個文字欄位，例如：general_subject（你說已經有）。
若你的 JSON 裡面欄位不同，也可以用 --text-keys 指定多個候選欄位，
會依序嘗試，第一個命中的就用。

用法：
    pip install -U pandas

    python build_dataset_from_json_dirs.py \
        --root data_root \
        --out data/all.csv \
        --text-keys general_subject subject title body content \
        --ext .json \
        --min-chars 3 \
        --limit-per-class 0

參數說明：
- --root：根目錄。
- --out：輸出 CSV 檔路徑（兩欄：text,label）。
- --text-keys：依序嘗試的文字欄位（命中一個就用）。
- --ext：讀取的副檔名，預設 .json。
- --min-chars：文字長度過短則捨棄（去除空白後計算）。
- --limit-per-class：每類最多保留幾筆（0=不限制）；可用來平衡資料。
- --shuffle：是否亂序（預設開啟）。

注意：
- 若 JSON 是 list 或 nested 結構，會嘗試挖出指定 key；若抓不到就跳過。
- 若你需要把多個欄位合併（例如主旨+首段），可用 --concat-keys 來指定。
"""

from __future__ import annotations
import argparse
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def try_extract_text(obj: Any, keys: List[str]) -> Optional[str]:
    """嘗試從 JSON 物件中依序抽取 keys 中的第一個命中的字串。
    支援巢狀：若 obj 是 dict，會直接找 key；若是 list，會對每個元素遞迴取第一個命中並拼接。"""
    def _from(o: Any) -> Optional[str]:
        if isinstance(o, str):
            return o
        if isinstance(o, dict):
            for k in keys:
                if k in o and isinstance(o[k], (str, list, dict)):
                    val = _from(o[k])
                    if isinstance(val, str) and val.strip():
                        return val
            # 若 key 不在最上層，遞迴所有 value
            for v in o.values():
                val = _from(v)
                if isinstance(val, str) and val.strip():
                    return val
        if isinstance(o, list):
            parts = []
            for it in o:
                val = _from(it)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
            if parts:
                return "\n".join(parts)
        return None
    return _from(obj)


def extract_concat(obj: Any, concat_keys: List[str]) -> Optional[str]:
    parts = []
    for k in concat_keys:
        t = try_extract_text(obj, [k])
        if t and t.strip():
            parts.append(t.strip())
    if parts:
        return " \n".join(parts)
    return None


def build_dataset(root: str, exts: List[str], text_keys: List[str], concat_keys: List[str], min_chars: int, limit_per_class: int) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []
    for class_dir in sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)]):
        label = os.path.basename(class_dir)
        files: List[str] = []
        for ext in exts:
            files.extend(glob(os.path.join(class_dir, f"**/*{ext}"), recursive=True))
        kept = 0
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
            except Exception:
                # 不是合法 JSON 就略過
                continue
            text = None
            if concat_keys:
                text = extract_concat(obj, concat_keys)
            if not text:
                text = try_extract_text(obj, text_keys)
            if not text:
                continue
            t = " ".join(text.split())  # normalize 空白
            if len(t.replace(" ", "")) < min_chars:
                continue
            rows.append((t, label))
            kept += 1
            if limit_per_class and kept >= limit_per_class:
                break
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df


def main():
    ap = argparse.ArgumentParser(description="Build (text,label) CSV from folder-labeled JSONs")
    ap.add_argument("--root", required=True, help="資料根目錄（子資料夾名=類別）")
    ap.add_argument("--out", required=True, help="輸出 CSV 路徑")
    ap.add_argument("--ext", nargs="*", default=[".json"], help="要讀的副檔名，預設 .json")
    ap.add_argument("--text-keys", nargs="*", default=["general_subject", "subject", "title", "body", "content"], help="文字欄位優先序")
    ap.add_argument("--concat-keys", nargs="*", default=[], help="可選：把多個欄位合併（主旨+內文）")
    ap.add_argument("--min-chars", type=int, default=3, help="最短字元數（去空白後）")
    ap.add_argument("--limit-per-class", type=int, default=0, help="每類最多保留幾筆（0=不限制）")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    df = build_dataset(
        root=args.root,
        exts=args.ext,
        text_keys=args.text_keys,
        concat_keys=args.concat_keys,
        min_chars=args.min_chars,
        limit_per_class=args.limit_per_class,
    )

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"完成：{args.out}（{len(df)} 筆）")


if __name__ == "__main__":
    main()
