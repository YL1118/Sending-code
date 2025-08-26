#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, csv, sys, os, glob, itertools
from typing import Any, Dict, List, Iterable, Union

try:
    import pandas as pd
except ImportError:
    pd = None

def iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    """
    逐檔輸出「記錄」(record)：
    - 若檔案是單一 JSON 物件 -> 輸出 1 筆
    - 若是 JSON 陣列 -> 陣列中每個元素各輸出 1 筆（元素需為物件）
    - 若是 JSON Lines (*.jsonl) -> 每行一筆（需為物件）
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return
        # JSON Lines
        if path.lower().endswith(".jsonl"):
            for i, line in enumerate(txt.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"{path}: 第 {i} 行不是合法 JSON。")
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}: 第 {i} 行不是物件 (dict)。")
                yield obj
            return

        # 一般 JSON
        try:
            data = json.loads(txt)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: JSON 解析失敗: {e}")

        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"{path}: 陣列元素 #{idx} 不是物件 (dict)。")
                yield item
        else:
            raise ValueError(f"{path}: 根節點不是物件或陣列。")

def get_by_path(obj: Any, path: str) -> Any:
    """
    使用點號/索引路徑抓值：
    例：user.name、items[0].price、meta.tags[2]
    若不存在，回傳 None。
    若遇到清單但未給索引，回傳以「|」串接的字串（例如 tags -> "a|b|c"）。
    """
    cur = obj
    tokens: List[str] = []
    # 先把 a[0].b[1] 轉成 a.[0].b.[1] 再以 '.' split
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == '[':
            if buf:
                tokens.append(buf); buf = ""
            j = path.find(']', i)
            if j == -1:
                raise ValueError(f"路徑語法錯誤：{path}")
            tokens.append(path[i:j+1])  # 例如 [0]
            i = j + 1
        elif c == '.':
            if buf:
                tokens.append(buf); buf = ""
            i += 1
        else:
            buf += c
            i += 1
    if buf:
        tokens.append(buf)

    for tok in tokens:
        if tok.startswith('[') and tok.endswith(']'):
            # 索引
            if not isinstance(cur, list):
                return None
            idx_str = tok[1:-1]
            if not idx_str.isdigit():
                return None
            idx = int(idx_str)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            if isinstance(cur, dict):
                if tok in cur:
                    cur = cur[tok]
                else:
                    return None
            elif isinstance(cur, list):
                # 若是清單但未指定索引 -> 將元素做扁平 join
                return "|".join(map(lambda x: str(x) if not isinstance(x, (dict, list)) else json.dumps(x, ensure_ascii=False), cur))
            else:
                return None
    # 最終輸出若是複合型別，做合理序列化
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, ensure_ascii=False)
    return cur

def discover_files(inputs: List[str], recursive: bool, patterns: List[str]) -> List[str]:
    files = []
    pats = patterns if patterns else ["*.json", "*.jsonl"]
    for inp in inputs:
        if os.path.isdir(inp):
            for pat in pats:
                globpat = os.path.join(inp, "**", pat) if recursive else os.path.join(inp, pat)
                files.extend(glob.glob(globpat, recursive=recursive))
        else:
            files.append(inp)
    # 去重+排序以穩定
    files = sorted(set(files))
    return files

def main():
    ap = argparse.ArgumentParser(description="從多個 JSON 檔抽取欄位並匯出成表格 (CSV/Excel)")
    ap.add_argument("inputs", nargs="+", help="輸入檔或資料夾（可多個）")
    ap.add_argument("-f", "--fields", required=True, nargs="+",
                    help="要抽取的欄位路徑，如 user.id items[0].price meta.tags")
    ap.add_argument("-o", "--out", required=True, help="輸出檔名：.csv 或 .xlsx")
    ap.add_argument("-r", "--recursive", action="store_true", help="遞迴掃描資料夾")
    ap.add_argument("-p", "--pattern", action="append", help="指定檔名樣式（可多次）。預設：*.json, *.jsonl")
    ap.add_argument("--add-source", action="store_true", help="額外輸出來源檔名欄位 _source")
    ap.add_argument("--encoding", default="utf-8", help="CSV 輸出編碼（預設 utf-8）。Windows Excel 可用 cp950/big5。")
    args = ap.parse_args()

    files = discover_files(args.inputs, args.recursive, args.pattern)
    if not files:
        print("找不到任何輸入檔。", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            for rec in iter_json_records(fp):
                row = {}
                for fld in args.fields:
                    row[fld] = get_by_path(rec, fld)
                if args.add_source:
                    row["_source"] = fp
                rows.append(row)
        except Exception as e:
            print(f"[警告] 跳過 {fp}: {e}", file=sys.stderr)

    if not rows:
        print("沒有可輸出的資料列。", file=sys.stderr)
        sys.exit(2)

    # 依輸出副檔名決定格式
    out_lower = args.out.lower()
    if out_lower.endswith(".csv"):
        # 以 csv 模組輸出，避免強制依賴 pandas
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", newline="", encoding=args.encoding) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已輸出 CSV：{args.out}（編碼：{args.encoding}）")
    elif out_lower.endswith(".xlsx"):
        if pd is None:
            print("需要 pandas 以輸出 .xlsx，請先安裝：pip install pandas openpyxl", file=sys.stderr)
            sys.exit(3)
        df = pd.DataFrame(rows)
        df.to_excel(args.out, index=False)
        print(f"已輸出 Excel：{args.out}")
    else:
        print("輸出副檔名不支援，請用 .csv 或 .xlsx", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()
