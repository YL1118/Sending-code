#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依資料夾 → 類別 的 JSON 資料，依子資料夾（sellected/selected / others）自動切成 train/test。
---------------------------------------------------------------------------------
資料結構（你的情境）：
root/
  保單查詢/
    sellected/    # 或 selected/（兩者拼字都支援）
      *.json
    others/
      *.json
  撤銷令/
    sellected/
      *.json
    others/
      *.json
  ... 其他類別

- 類別 = 第一層資料夾名稱（例如「保單查詢」）。
- 文本欄位：預設從 JSON 的 general_subject 擷取；可用 --text-keys 或 --concat-keys 覆寫。
- 會輸出兩個 CSV：train.csv（取自 sellected/selected）、test.csv（取自 others/）。
- 可選：--keep-caption-col 讓 caption 以獨立欄位輸出；否則會併入 text。

用法：
    pip install -U pandas

    python build_dataset_split_from_json_dirs.py \
      --root data_root \
      --train-out data/train.csv \
      --test-out data/test.csv \
      --text-keys general_subject subject title body content caption \
      --concat-keys general_subject body \
      --keep-caption-col \
      --min-chars 3 --shuffle

參數摘要：
- --train-subdir-names：視為訓練集的子資料夾名稱（預設 sellected, selected）。
- --test-subdir-names：視為測試/驗證集的子資料夾名稱（預設 others）。
- --ext：要讀的副檔名（預設 .json）。
- --limit-per-class-train / --limit-per-class-test：每類最多保留幾筆，0 不限。
- --keep-caption-col：若指定，caption 會獨立輸出，不併進 text。
- --caption-keys：caption 的欄位名稱（預設 caption）。
- --shuffle：輸出前打亂列順序。
"""

from __future__ import annotations
import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------- 抽取工具 ----------------------

def try_extract_text(obj: Any, keys: List[str]) -> Optional[str]:
    """優先依 keys 取值；若未命中，對該 key 做深度搜尋，找到第一個非空字串即回傳。"""
    def _from(o: Any) -> Optional[str]:
        if isinstance(o, str):
            return o
        if isinstance(o, dict):
            for k in keys:
                if k in o and isinstance(o[k], (str, list, dict)):
                    val = _from(o[k])
                    if isinstance(val, str) and val.strip():
                        return val
            # 未命中 keys，繼續深搜
            for v in o.values():
                val = _from(v)
                if isinstance(val, str) and val.strip():
                    return val
        if isinstance(o, list):
            parts: List[str] = []
            for it in o:
                val = _from(it)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
            if parts:
                return "\n".join(parts)
        return None
    return _from(obj)


def extract_concat(obj: Any, concat_keys: List[str]) -> Optional[str]:
    parts: List[str] = []
    for k in concat_keys:
        t = try_extract_text(obj, [k])
        if t and isinstance(t, str) and t.strip():
            parts.append(t.strip())
    if parts:
        return " \n".join(parts)
    return None


# ---------------------- 檔案收集 ----------------------

def collect_jsons(root: str, class_dir: str, subdir_names: List[str], exts: List[str]) -> List[str]:
    found: List[str] = []
    for name in subdir_names:
        p = os.path.join(root, class_dir, name)
        if not os.path.isdir(p):
            continue
        for ext in exts:
            found.extend(glob(os.path.join(p, f"**/*{ext}"), recursive=True))
    return sorted(found)


# ---------------------- 文本/標籤構建 ----------------------

def build_rows(file_paths: List[str], label: str, text_keys: List[str], concat_keys: List[str],
               caption_keys: List[str], keep_caption_col: bool, min_chars: int, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    kept = 0
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
        except Exception:
            continue

        # 取 caption（單獨欄位或併入 text）
        caption_val = ""
        for ck in caption_keys:
            tmp = try_extract_text(obj, [ck])
            if tmp and tmp.strip():
                caption_val = tmp.strip()
                break

        # 先試 concat-keys，再試 text-keys
        concat_eff = [k for k in concat_keys if not (keep_caption_col and k in caption_keys)]
        text = extract_concat(obj, concat_eff) if concat_eff else None
        if not text:
            # 當 keep_caption_col=True 時，text-keys 內若包含 caption_key，需略過避免併入
            tk = [k for k in text_keys if not (keep_caption_col and k in caption_keys)]
            text = try_extract_text(obj, tk)

        # 若仍無文本且 keep-caption-col 為 False，允許把 caption 併進 text
        if (not text) and (not keep_caption_col) and caption_val:
            text = caption_val
        if not text:
            continue

        tnorm = " ".join(text.split())
        if len(tnorm.replace(" ", "")) < min_chars:
            continue

        row: Dict[str, str] = {"text": tnorm, "label": label}
        if keep_caption_col:
            row[caption_key] = caption_val
        rows.append(row)
        kept += 1
        if limit and kept >= limit:
            break
    return rows


def build_split(root: str,
                train_subdir_names: List[str],
                test_subdir_names: List[str],
                exts: List[str],
                text_keys: List[str],
                concat_keys: List[str],
                caption_keys: List[str],
                keep_caption_col: bool,
                min_chars: int,
                limit_per_class_train: int,
                limit_per_class_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    train_rows_all: List[Dict[str, str]] = []
    test_rows_all: List[Dict[str, str]] = []

    for label in class_dirs:
        train_files = collect_jsons(root, label, train_subdir_names, exts)
        test_files = collect_jsons(root, label, test_subdir_names, exts)

        train_rows = build_rows(train_files, label, text_keys, concat_keys, caption_keys,
                                keep_caption_col, min_chars, limit_per_class_train)
        test_rows = build_rows(test_files, label, text_keys, concat_keys, caption_keys,
                               keep_caption_col, min_chars, limit_per_class_test)

        train_rows_all.extend(train_rows)
        test_rows_all.extend(test_rows)

    train_df = pd.DataFrame(train_rows_all)
    test_df = pd.DataFrame(test_rows_all)
    return train_df, test_df


# ---------------------- 入口點 ----------------------

def main():
    ap = argparse.ArgumentParser(description="Build train/test CSVs from folder-labeled JSONs (sellected/selected vs others)")
    ap.add_argument("--root", required=True, help="根目錄（第一層資料夾名 = 類別名）")
    ap.add_argument("--train-out", required=True, help="輸出訓練集 CSV")
    ap.add_argument("--test-out", required=True, help="輸出測試/驗證集 CSV")
    ap.add_argument("--train-subdir-names", nargs="*", default=["sellected", "selected"], help="視為 train 的子資料夾名稱")
    ap.add_argument("--test-subdir-names", nargs="*", default=["others"], help="視為 test/valid 的子資料夾名稱")
    ap.add_argument("--ext", nargs="*", default=[".json"], help="要讀取的副檔名")
    ap.add_argument("--text-keys", nargs="*", default=["general_subject", "subject", "title", "body", "content", "caption"], help="文字欄位優先序（預設含 caption）")
    ap.add_argument("--concat-keys", nargs="*", default=[], help="可選：把多個欄位合併（例：general_subject body）")
    ap.add_argument("--keep-caption-col", action="store_true", help="若指定，caption 會以獨立欄位輸出，不併入 text")
    ap.add_argument("--caption-keyss", nargs="*", default=["caption", "Caption"], help="caption 對應的 JSON 欄位名清單（會依序嘗試；預設 caption/Caption）")
    ap.add_argument("--min-chars", type=int, default=3, help="最短字元數（去空白後）")
    ap.add_argument("--limit-per-class-train", type=int, default=0, help="每類訓練最多保留幾筆（0=不限）")
    ap.add_argument("--limit-per-class-test", type=int, default=0, help="每類測試最多保留幾筆（0=不限）")
    ap.add_argument("--shuffle", action="store_true", help="輸出前打亂順序")
    args = ap.parse_args()

    train_df, test_df = build_split(
        root=args.root,
        train_subdir_names=args.train_subdir_names,
        test_subdir_names=args.test_subdir_names,
        exts=args.ext,
        text_keys=args.text_keys,
        concat_keys=args.concat_keys,
        caption_keys=args.caption_keys,
        keep_caption_col=args.keep_caption_col,
        min_chars=args.min_chars,
        limit_per_class_train=args.limit_per_class_train,
        limit_per_class_test=args.limit_per_class_test,
    )

    if args.shuffle:
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out) or ".", exist_ok=True)
    train_df.to_csv(args.train_out, index=False, encoding="utf-8-sig")
    test_df.to_csv(args.test_out, index=False, encoding="utf-8-sig")

    # 顯示欄位與筆數（方便確認是否有 caption 欄）
    print(f"完成：{args.train_out}（{len(train_df)} 筆）欄位：{', '.join(train_df.columns)}")
    print(f"完成：{args.test_out}（{len(test_df)} 筆）欄位：{', '.join(test_df.columns)}")


if __name__ == "__main__":
    main()
