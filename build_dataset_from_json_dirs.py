#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依資料夾 → 類別 的 JSON 資料，依子資料夾（sellected/others）自動切成 train/test。
---------------------------------------------------------------------------------
資料結構（你的情境）：
root/
  保單查詢/
    sellected/    # 或 selected/（拼字都支援）
      *.json
    others/
      *.json
  撤銷令/
    sellected/
      *.json
    others/
      *.json
  ... 其他類別

- 類別 = 每個第一層資料夾名稱（例如「保單查詢」）。
- 文字欄位：預設從 JSON 的 general_subject 擷取；也可用 --text-keys 或 --concat-keys。
- 會輸出兩個 CSV：train.csv（取自 sellected/selected）、test.csv（取自 others/）。

用法：
    pip install -U pandas

    python build_dataset_split_from_json_dirs.py \
      --root data_root \
      --train-out data/train.csv \
      --test-out data/test.csv \
      --text-keys general_subject subject title body content \
      --min-chars 3

可調參數：
- --train-subdir-names：視為訓練集的子資料夾名稱（預設 sellected, selected）。
- --test-subdir-names：視為測試/驗證集的子資料夾名稱（預設 others）。
- --ext：要讀的副檔名（預設 .json）。
- --limit-per-class-train / --limit-per-class-test：每類最多保留幾筆，0 不限。
- --shuffle：輸出前是否打亂列順序。
"""

from __future__ import annotations
import argparse
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def try_extract_text(obj: Any, keys: List[str]) -> Optional[str]:
    def _from(o: Any) -> Optional[str]:
        if isinstance(o, str):
            return o
        if isinstance(o, dict):
            for k in keys:
                if k in o and isinstance(o[k], (str, list, dict)):
                    val = _from(o[k])
                    if isinstance(val, str) and val.strip():
                        return val
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


def collect_jsons(root: str, class_dir: str, subdir_names: List[str], exts: List[str]) -> List[str]:
    found = []
    for name in subdir_names:
        p = os.path.join(root, class_dir, name)
        if not os.path.isdir(p):
            continue
        for ext in exts:
            found.extend(glob(os.path.join(p, f"**/*{ext}"), recursive=True))
    return sorted(found)


def build_split(root: str,
                train_subdir_names: List[str],
                test_subdir_names: List[str],
                exts: List[str],
                text_keys: List[str],
                concat_keys: List[str],
                min_chars: int,
                limit_per_class_train: int,
                limit_per_class_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    train_rows: List[Tuple[str, str]] = []
    test_rows: List[Tuple[str, str]] = []

    for label in class_dirs:
        # 收集 train/test 檔案
        train_files = collect_jsons(root, label, train_subdir_names, exts)
        test_files = collect_jsons(root, label, test_subdir_names, exts)

        kept_tr = 0
        for fp in train_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
            except Exception:
                continue
            text = extract_concat(obj, concat_keys) if concat_keys else None
            if not text:
                text = try_extract_text(obj, text_keys)
            if not text:
                continue
            t = " ".join(text.split())
            if len(t.replace(" ", "")) < min_chars:
                continue
            train_rows.append((t, label))
            kept_tr += 1
            if limit_per_class_train and kept_tr >= limit_per_class_train:
                break

        kept_te = 0
        for fp in test_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
            except Exception:
                continue
            text = extract_concat(obj, concat_keys) if concat_keys else None
            if not text:
                text = try_extract_text(obj, text_keys)
            if not text:
                continue
            t = " ".join(text.split())
            if len(t.replace(" ", "")) < min_chars:
                continue
            test_rows.append((t, label))
            kept_te += 1
            if limit_per_class_test and kept_te >= limit_per_class_test:
                break

    train_df = pd.DataFrame(train_rows, columns=["text", "label"])
    test_df = pd.DataFrame(test_rows, columns=["text", "label"])
    return train_df, test_df


def main():
    ap = argparse.ArgumentParser(description="Build train/test CSVs from folder-labeled JSONs (sellected/others split)")
    ap.add_argument("--root", required=True, help="根目錄（第一層資料夾名 = 類別名）")
    ap.add_argument("--train-out", required=True, help="輸出訓練集 CSV")
    ap.add_argument("--test-out", required=True, help="輸出測試/驗證集 CSV")
    ap.add_argument("--train-subdir-names", nargs="*", default=["sellected", "selected"], help="視為 train 的子資料夾名稱")
    ap.add_argument("--test-subdir-names", nargs="*", default=["others"], help="視為 test/valid 的子資料夾名稱")
    ap.add_argument("--ext", nargs="*", default=[".json"], help="要讀取的副檔名")
    ap.add_argument("--text-keys", nargs="*", default=["general_subject", "subject", "title", "body", "content"], help="文字欄位優先序")
    ap.add_argument("--concat-keys", nargs="*", default=[], help="可選：把多個欄位合併（主旨+內文）")
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
    print(f"完成：{args.train_out}（{len(train_df)} 筆），{args.test_out}（{len(test_df)} 筆）")


if __name__ == "__main__":
    main()
