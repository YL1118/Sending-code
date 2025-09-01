# -*- coding: utf-8 -*-
"""
sync_general_subject.py
從 test_json_more_forall 補齊 parsed 的 general_subject（同名檔、同類別）。
若 parsed 的 general_subject 為空，且來源有值，則寫回 parsed。

結構：
  parsed/<category>/*.json
  test_json_more_forall/<category>/*.json

使用:
最安全試跑（不寫檔）：
python sync_general_subject.py --dry-run

正式執行並備份：
python sync_general_subject.py --backup

只跑特定類別（例如 通知函 與 保單查詢）：
python sync_general_subject.py --categories 通知函 保單查詢

欄位不是 general_subject 時（可改）：
python sync_general_subject.py --key subject_summary

"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Tuple

def is_empty_value(v: Any) -> bool:
    """定義 '空'：None、缺鍵、空字串或只含空白。非字串類型則以 bool(v) 判斷。"""
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    return not bool(v)

def load_json(p: Path) -> Tuple[dict, bool]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), True
    except Exception as e:
        print(f"❌ 讀取失敗：{p} -> {e}")
        return {}, False

def save_json(p: Path, obj: dict) -> bool:
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 寫入失敗：{p} -> {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="比對兩個資料夾，若 parsed 檔案的 general_subject 為空則用來源補齊")
    ap.add_argument("--parsed-root", type=str, default="parsed", help="要被補齊的根目錄")
    ap.add_argument("--source-root", type=str, default="test_json_more_forall", help="來源根目錄（提供 general_subject）")
    ap.add_argument("--key", type=str, default="general_subject", help="要同步的欄位鍵名")
    ap.add_argument("--dry-run", action="store_true", help="試跑：不寫檔，只列印動作")
    ap.add_argument("--backup", action="store_true", help="寫檔前建立備份（鏡射路徑）")
    ap.add_argument("--backup-root", type=str, default="_backup_sync", help="備份根目錄（在 parsed-root 內建立）")
    ap.add_argument("--categories", nargs="*", default=None, help="只處理指定類別資料夾（空則全部）")
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    source_root = Path(args.source_root)
    if not parsed_root.exists():
        raise SystemExit(f"找不到 parsed-root：{parsed_root.resolve()}")
    if not source_root.exists():
        raise SystemExit(f"找不到 source-root：{source_root.resolve()}")

    cats = [d for d in parsed_root.iterdir() if d.is_dir()]
    if args.categories:
        allow = set(args.categories)
        cats = [d for d in cats if d.name in allow]

    if not cats:
        print("⚠️ 沒有可處理的類別資料夾。")
        return

    backup_root = parsed_root / args.backup_root
    if args.backup and not args.dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)

    total_checked = 0
    total_updated = 0
    total_skipped_no_source = 0
    total_skipped_not_empty = 0
    total_skipped_source_empty = 0
    total_errors = 0

    print(f"🚀 開始：parsed={parsed_root.resolve()}  source={source_root.resolve()}")
    print(f"鍵：{args.key}；dry-run={'是' if args.dry_run else '否'}；backup={'是' if args.backup else '否'}")

    for cat_dir in sorted(cats, key=lambda p: p.name):
        cat = cat_dir.name
        src_cat_dir = source_root / cat
        if not src_cat_dir.exists():
            print(f"ℹ️ 類別「{cat}」在來源不存在，略過整個類別。")
            continue

        for p_json in sorted(cat_dir.glob("*.json")):
            total_checked += 1
            q_json = src_cat_dir / p_json.name
            if not q_json.exists():
                total_skipped_no_source += 1
                print(f"   ↪︎ 無對應來源：{cat}/{p_json.name}（略過）")
                continue

            parsed_obj, ok1 = load_json(p_json)
            source_obj, ok2 = load_json(q_json)
            if not (ok1 and ok2):
                total_errors += 1
                continue

            parsed_val = parsed_obj.get(args.key, None)
            source_val = source_obj.get(args.key, None)

            if not is_empty_value(parsed_val):
                total_skipped_not_empty += 1
                # 有值就不動
                continue

            if is_empty_value(source_val):
                total_skipped_source_empty += 1
                # 來源也沒值
                continue

            # 要更新
            print(f"✔ 補齊：{cat}/{p_json.name}  <-  來源 {args.key}（長度={len(source_val) if isinstance(source_val, str) else 'n/a'}）")
            parsed_obj[args.key] = source_val

            if args.dry_run:
                continue

            # 備份
            if args.backup:
                backup_path = backup_root / cat / p_json.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(p_json), str(backup_path))
                except Exception as e:
                    print(f"   ⚠️ 備份失敗：{p_json} -> {e}")

            if save_json(p_json, parsed_obj):
                total_updated += 1
            else:
                total_errors += 1

    print("—— 統計 ——")
    print(f"檢查檔案：{total_checked}")
    print(f"已更新：{total_updated}")
    print(f"略過（來源不存在）：{total_skipped_no_source}")
    print(f"略過（parsed 原本不空）：{total_skipped_not_empty}")
    print(f"略過（來源也為空）：{total_skipped_source_empty}")
    print(f"錯誤：{total_errors}")
    if args.backup and not args.dry_run and total_updated > 0:
        print(f"備份位置：{(parsed_root / args.backup_root).resolve()}")

if __name__ == "__main__":
    main()
