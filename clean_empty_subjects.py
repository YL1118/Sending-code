#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遞迴檢查 test_json 下所有子資料夾，刪除或移動主旨為空的 JSON
----------------------------------------------------
- 檔案結構： test_json/類別/selected, others
- 若 general_subject 為空或缺少，移動或刪除該檔

用法：
    # 直接刪除
    python clean_empty_subjects.py --indir test_json

    # 移動到另一個資料夾（建議）
    python clean_empty_subjects.py --indir test_json --move-to empty_jsons
"""

import os
import json
import argparse
import shutil

def is_subject_empty(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        subj = data.get("general_subject", "").strip()
        return subj == ""
    except Exception as e:
        print(f"[警告] {path} 無法讀取 ({e})，略過。")
        return False

def main():
    ap = argparse.ArgumentParser(description="刪除或移動主旨為空的 JSON 檔案 (遞迴子資料夾)")
    ap.add_argument("--indir", required=True, help="最上層資料夾 (例如 test_json)")
    ap.add_argument("--move-to", type=str, default=None, help="將空主旨檔案移動到指定資料夾，而非刪除")
    args = ap.parse_args()

    if args.move_to and not os.path.exists(args.move_to):
        os.makedirs(args.move_to)

    removed, moved, total = 0, 0, 0
    for root, _, files in os.walk(args.indir):   # 遞迴走訪所有子資料夾
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            total += 1
            fpath = os.path.join(root, fn)
            if is_subject_empty(fpath):
                if args.move_to:
                    # 保留原始類別結構： test_json/類別/selected/xxx.json
                    rel_path = os.path.relpath(fpath, args.indir)
                    target_path = os.path.join(args.move_to, rel_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.move(fpath, target_path)
                    moved += 1
                    print(f"已移動：{fpath} -> {target_path}")
                else:
                    os.remove(fpath)
                    removed += 1
                    print(f"已刪除：{fpath}")

    print(f"\n處理完成，總檔案 {total}，刪除 {removed}，移動 {moved}。")

if __name__ == "__main__":
    main()
