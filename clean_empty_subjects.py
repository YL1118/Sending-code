#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
刪除或移動主旨為空的 JSON 檔案
----------------------------------------------------
- 偵測 JSON 檔中的 key: "general_subject"
- 若為空字串 ("") 或缺少該欄位，就刪除或移動該檔案

用法：
    # 直接刪除
    python clean_empty_subjects.py --indir data/jsons

    # 移動到另一個資料夾（建議用法）
    python clean_empty_subjects.py --indir data/jsons --move-to data/empty_subjects
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
    ap = argparse.ArgumentParser(description="刪除或移動主旨為空的 JSON 檔案")
    ap.add_argument("--indir", required=True, help="JSON 檔案所在資料夾")
    ap.add_argument("--move-to", type=str, default=None, help="將空主旨檔案移動到指定資料夾，而非直接刪除")
    args = ap.parse_args()

    if args.move_to and not os.path.exists(args.move_to):
        os.makedirs(args.move_to)

    removed, moved = 0, 0
    for fn in os.listdir(args.indir):
        if not fn.lower().endswith(".json"):
            continue
        fpath = os.path.join(args.indir, fn)
        if is_subject_empty(fpath):
            if args.move_to:
                shutil.move(fpath, os.path.join(args.move_to, fn))
                moved += 1
                print(f"已移動：{fpath} -> {args.move_to}")
            else:
                os.remove(fpath)
                removed += 1
                print(f"已刪除：{fpath}")

    print(f"\n處理完成，刪除 {removed} 份，移動 {moved} 份。")

if __name__ == "__main__":
    main()
