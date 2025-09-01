#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從以資料夾代表類別的 JSON 檔，建立可訓練的 CSV（text,label ；可選 caption 欄位）
--------------------------------------------------------------------------
特色：
- 遞迴掃描：root/類別/(selected|others|...)/*.json
- 文本抽取：從多個 key 中擷取，或使用 --concat-keys 合併欄位
- 清理：--min-chars 過短則略過；--limit-per-class 每類最多 N 筆
- 輸出：預設輸出 text,label；若加 --keep-caption-col，另輸出 caption 欄
- 亂序：--shuffle 打亂樣本

基本用法：
    python build_dataset_from_json_dirs.py \
      --root test_json \
      --out  data/all.csv \
      --text-keys general_subject subject title body content caption \
      --concat-keys general_subject body \
      --keep-caption-col \
      --min-chars 3 --shuffle

備註：
- label 取自 root 底下第一層資料夾名稱（例如 test_json/扣押命令/selected/a.json → label=扣押命令）
- 若 JSON 缺少指定欄位，會嘗試深層搜尋；若仍找不到且長度不足，該檔略過
"""
from __future__ import annotations
import argparse
import json
import os
from glob import glob
from typing import Any, List, Optional, Tuple

import pandas as pd

# ---------------------------- 抽取工具 ----------------------------

def _deep_find_text(obj: Any, keys: List[str]) -> Optional[str]:
    """在 JSON 物件中優先以 keys 尋找文字；若未命中，嘗試深度搜尋第一個可用字串。"""
    # 先依 keys 直取
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, str) and v.strip():
                    return v
                # 若是 list/dict 再往下嘗試
                if isinstance(v, (list, dict)):
                    r = _deep_find_text(v, [])
                    if isinstance(r, str) and r.strip():
                        return r
    # 深度搜尋：找到第一個非空字串就回傳
    if isinstance(obj, dict):
        for v in obj.values():
            r = _deep_find_text(v, [])
            if isinstance(r, str) and r.strip():
                return r
    elif isinstance(obj, list):
        for it in obj:
            r = _deep_find_text(it, [])
            if isinstance(r, str) and r.strip():
                return r
    elif isinstance(obj, str):
        return obj
    return None


essential_default_keys = ["general_subject", "subject", "title", "body", "content", "caption"]


def extract_text_fields(data: Any, text_keys: List[str], concat_keys: List[str], keep_caption_col: bool) -> Tuple[str, str]:
    """回傳 (text, caption)
    - 若 keep_caption_col=True，text 不包含 caption；否則 text 會包含 caption
    """
    caption_val = ""
    # 準備合併 text 的候選欄位
    parts: List[str] = []
    for k in text_keys:
        if k == "caption":
            # 取 caption 原值（若是 list/dict 也嘗試抽文字）
            cap = None
            if isinstance(data, dict) and "caption" in data:
                v = data.get("caption")
                if isinstance(v, str):
                    cap = v
                else:
                    cap = _deep_find_text(v, [])
            if cap and isinstance(cap, str) and cap.strip():
                caption_val = cap.strip()
            # 只有在不保留獨立欄位時，才把 caption 併入 text
            if not keep_caption_col and caption_val:
                parts.append(caption_val)
            continue
        # 其餘欄位
        val = None
        if isinstance(data, dict) and k in data:
            v = data.get(k)
            if isinstance(v, str):
                val = v
            else:
                val = _deep_find_text(v, [])
        if not val:  # 退而求其次深搜
            val = _deep_find_text(data, [k])
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # concat-keys：額外按順序拼接（常用：general_subject + body）
    for k in concat_keys:
        if k == "caption" and keep_caption_col:
            continue  # 保留 caption 獨立時，不把它併進 text
        v = None
        if isinstance(data, dict) and k in data:
            vv = data.get(k)
            if isinstance(vv, str):
                v = vv
            else:
                v = _deep_find_text(vv, [])
        if not v:
            v = _deep_find_text(data, [k])
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # 組合 text
    text = " \n".join([p for p in parts if p]).strip()
    return text, caption_val


# ---------------------------- 主流程 ----------------------------

def build_dataset(root: str, exts: List[str], text_keys: List[str], concat_keys: List[str], keep_caption_col: bool,
                  min_chars: int, limit_per_class: int) -> pd.DataFrame:
    rows: List[dict] = []
    # 取得第一層類別資料夾
    class_dirs = [d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)]
    for class_dir in sorted(class_dirs):
        label = os.path.basename(class_dir)
        # 收集該類別下所有 JSON 檔（遞迴）
        files: List[str] = []
        for ext in exts:
            files.extend(glob(os.path.join(class_dir, f"**/*{ext}"), recursive=True))
        kept = 0
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
            except Exception:
                continue
            text, caption_val = extract_text_fields(data, text_keys, concat_keys, keep_caption_col)
            # 清理
            norm = " ".join(text.split())  # 壓空白
            if len(norm.replace(" ", "")) < min_chars:
                continue
            row = {"text": norm, "label": label}
            if keep_caption_col:
                row["caption"] = caption_val
            rows.append(row)
            kept += 1
            if limit_per_class and kept >= limit_per_class:
                break
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser(description="從資料夾 JSON 建立 CSV（支援 caption 獨立欄位）")
    ap.add_argument("--root", required=True, help="資料根目錄（子資料夾名=類別）")
    ap.add_argument("--out", required=True, help="輸出 CSV 路徑")
    ap.add_argument("--ext", nargs="*", default=[".json"], help="要讀取的副檔名（預設 .json）")
    ap.add_argument("--text-keys", nargs="*", default=essential_default_keys,
                    help="欲抽取文字的欄位清單，預設包含 general_subject/subject/title/body/content/caption")
    ap.add_argument("--concat-keys", nargs="*", default=[], help="可選：另外指定要合併進 text 的欄位順序（例如 general_subject body）")
    ap.add_argument("--keep-caption-col", action="store_true", help="若指定，caption 會以獨立欄位輸出，不併入 text")
    ap.add_argument("--min-chars", type=int, default=3, help="最短字元數（去空白後）不足則略過")
    ap.add_argument("--limit-per-class", type=int, default=0, help="每類最多保留幾筆（0=不限）")
    ap.add_argument("--shuffle", action="store_true", help="輸出前是否打亂樣本順序")
    args = ap.parse_args()

    df = build_dataset(
        root=args.root,
        exts=args.ext,
        text_keys=args.text_keys,
        concat_keys=args.concat_keys,
        keep_caption_col=args.keep_caption_col,
        min_chars=args.min_chars,
        limit_per_class=args.limit_per_class,
    )

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"完成：{args.out}（{len(df)} 筆） | 欄位：{', '.join(df.columns)}")


if __name__ == "__main__":
    main()
