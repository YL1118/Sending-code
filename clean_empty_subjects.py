#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
刪除主旨為空的 JSON 檔案
----------------------------------------------------
- 偵測 JSON 檔中的 key: "general_subject"
- 若為空字串 ("") 或缺少該欄位，就刪除整個檔案

用法：
    python clean_empty_subjects.py --indir data/jsons
"""

import os
import json
import argparse

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
    ap = argparse.ArgumentParser(description="刪除主旨為空的 JSON 檔案")
    ap.add_argument("--indir", required=True, help="JSON 檔案所在資料夾")
    args = ap.parse_args()

    removed = 0
    for fn in os.listdir(args.indir):
        if not fn.lower().endswith(".json"):
            continue
        fpath = os.path.join(args.indir, fn)
        if is_subject_empty(fpath):
            os.remove(fpath)
            removed += 1
            print(f"已刪除：{fpath}")

    print(f"\n處理完成，共刪除 {removed} 份檔案。")

if __name__ == "__main__":
    main()
