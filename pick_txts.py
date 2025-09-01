# -*- coding: utf-8 -*-
# pick_jsons.py  ->  parsed/<類別>/*.json  ->  <out-root>/<類別>/{selected,others}/*.json
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

def safe_place(src_path: Path, dest_dir: Path, move: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name
    if dest.exists():  # 避免覆蓋：加上 _1, _2, ...
        i = 1
        stem, suf = src_path.stem, src_path.suffix
        while (dest_dir / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = dest_dir / f"{stem}_{i}{suf}"
    if move:
        shutil.move(str(src_path), str(dest))
    else:
        shutil.copy2(str(src_path), str(dest))
    return dest

def collect_categories(src_root: Path, ext: str):
    """只掃描一層：src_root/<category>/*.ext"""
    cats = {}
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        files = sorted(p for p in d.glob(f"*{ext}") if p.is_file())
        if files:
            cats[d.name] = files
    return cats

def main():
    ap = argparse.ArgumentParser(
        description="從 parsed/<類別>/*.json 每類別隨機抽取 N 個，輸出為 <out-root>/<類別>/{selected,others}"
    )
    ap.add_argument("--src-root", type=str, default="parsed",
                    help="輸入根目錄（結構：parsed/<類別>/*.json）")
    ap.add_argument("--out-root", type=str, default=None,
                    help="輸出根目錄（預設：<src-root>/_picked_YYYYmmdd_HHMMSS）")
    ap.add_argument("--n", type=int, default=5, help="每類別抽取數量")
    ap.add_argument("--move", action="store_true",
                    help="搬移檔案（預設為複製）")
    ap.add_argument("--seed", type=int, default=None,
                    help="隨機種子（指定可重現抽樣）")
    ap.add_argument("--ext", type=str, default=".json",
                    help="要處理的副檔名（預設 .json）")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    if not src_root.exists():
        raise SystemExit(f"找不到輸入根目錄：{src_root.resolve()}")

    out_root = Path(args.out_root) if args.out_root else (src_root / f"_picked_{datetime.now():%Y%m%d_%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    cats = collect_categories(src_root, args.ext)
    if not cats:
        raise SystemExit(f"在 {src_root.resolve()} 下面找不到任何含有 {args.ext} 的類別資料夾")

    total_sel = total_oth = 0
    print(f"🚀 掃描來源：{src_root.resolve()}")
    print(f"📦 輸出位置：{out_root.resolve()}")
    print(f"參數：每類別 N={args.n}；模式={'搬移' if args.move else '複製'}；副檔名={args.ext}")

    for cat, files in cats.items():
        k = min(args.n, len(files))
        picked = set(random.sample(files, k=k))

        cat_root = out_root / cat
        sel_dir = cat_root / "selected"
        oth_dir = cat_root / "others"

        count_sel = count_oth = 0
        for p in files:
            dest_dir = sel_dir if p in picked else oth_dir
            safe_place(p, dest_dir, move=args.move)
            if p in picked:
                count_sel += 1
            else:
                count_oth += 1

        total_sel += count_sel
        total_oth += count_oth
        print(f"   📁 類別「{cat}」：selected {count_sel}、others {count_oth}  ->  {cat_root}")

    print("——")
    print(f"✅ 全部完成：selected {total_sel} 檔、others {total_oth} 檔")
    print(f"🗂️ 結構示意：")
    print(f"{out_root}/")
    print(f"  <類別A>/")
    print(f"    selected/*.json")
    print(f"    others/*.json")
    print(f"  <類別B>/ ...")

if __name__ == "__main__":
    main()
