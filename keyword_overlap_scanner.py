#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨檔案「重複關鍵字」快速掃描器（支援中英文）
-------------------------------------------------
用途：
1) 掃描多個檔案（.txt / .md，選配 .pdf）抽取關鍵詞，
2) 找出出現在「至少 K 個檔案」的重複關鍵詞，
3) 匯出交集報表（token, 出現於幾個檔案, 總次數, 各檔案次數），
4) 額外提供：每檔 top-N 關鍵詞（TF‑IDF）與檔案兩兩 Jaccard 相似度矩陣。

安裝（選配）
    pip install -U jieba scikit-learn pandas tabulate pdfplumber

基本用法
    # 掃描資料夾下所有 .txt / .md 檔，列出出現在 >=2 個檔案的詞
    python keyword_overlap_scanner.py scan --dir ./docs --min-df 2 --export overlaps.csv

    # 指定檔案清單
    python keyword_overlap_scanner.py scan --paths a.txt b.txt c.md --min-df 3

    # 顯示每檔 top-20 TF‑IDF 關鍵詞再做交集（較貼近「關鍵字」語義）
    python keyword_overlap_scanner.py scan --dir ./docs --use-tfidf --topn 20

    # 顯示檔案兩兩 Jaccard 相似度（基於詞集合）
    python keyword_overlap_scanner.py jaccard --dir ./docs

說明：
- 中文：預設會自動偵測中文並嘗試使用 jieba 分詞；未安裝則改用中英混合的 regex 切詞。
- 關鍵詞定義：
  * 預設「詞長 >= 2」，過濾數字/符號與常見停用詞。
  * 可搭配 --use-tfidf 僅取每檔 top‑N（--topn）做交集，減少雜訊。
- PDF：若安裝 pdfplumber，會嘗試讀取 .pdf 文字（掃描影像 PDF 不保證抓到）。

作者：你（與 ChatGPT 協作）
"""

from __future__ import annotations
import argparse
import glob
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Set

# 可選依賴
try:
    import jieba  # type: ignore
    HAVE_JIEBA = True
except Exception:
    HAVE_JIEBA = False

try:
    import pdfplumber  # type: ignore
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

try:
    import pandas as pd  # type: ignore
    HAVE_PD = True
except Exception:
    HAVE_PD = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ---------------------- 基本工具 ----------------------

EN_STOP = set(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with""".split()
)

ZH_STOP = {
    "的","了","呢","嗎","哦","喔","在","是","有","與","和","及","及其","以及","並","與否","及其","且","我","你","他","她","它","我們","你們","他們","她們",
    "這","那","哪","個","些","一個","一些","如果","因為","所以","而且","但是","然而","或者","以及","並且","等","就","都","也","還","很","更","最",
}

RE_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+|[0-9]+|[\u4e00-\u9fff]{1,}")
RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")


def read_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf" and HAVE_PDF:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except Exception:
            return ""
    # 其他格式一律當純文字讀（可能會亂碼）
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def tokenize(text: str) -> List[str]:
    text = text.strip()
    # 粗略偵測中文
    has_cjk = bool(RE_HAS_CJK.search(text))
    toks: List[str] = []
    if has_cjk and HAVE_JIEBA:
        toks = [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        toks = RE_WORD.findall(text)
    # 標準化與過濾
    out = []
    for t in toks:
        tt = t.lower()
        # 去除純數字
        if tt.isdigit():
            continue
        # 長度門檻（中文單字容易雜訊，≥2）
        if RE_HAS_CJK.search(tt):
            if len(tt) < 2:  # 中文一字多義，太短噪音大
                continue
            if tt in ZH_STOP:
                continue
        else:
            if len(tt) < 2:
                continue
            if tt in EN_STOP:
                continue
        out.append(tt)
    return out

# ---------------------- 掃描與彙整 ----------------------

def collect_tokens_per_file(paths: List[str], use_tfidf: bool=False, topn: int=20) -> Tuple[Dict[str, Counter], Dict[str, Set[str]]]:
    """回傳：每檔 token Counter、以及每檔 token set（供 Jaccard/df 用）。
    若 use_tfidf=True，則只保留每檔 topn 關鍵詞（以詞頻矩陣做 TF-IDF）。
    """
    texts_by_file: Dict[str, str] = {p: read_text_from_path(p) for p in paths}

    if use_tfidf:
        if not HAVE_SK:
            raise RuntimeError("需要 scikit-learn 才能使用 --use-tfidf")
        # 先用 token 化後再拼回字串，確保中英處理一致
        docs_tokens = {p: tokenize(txt) for p, txt in texts_by_file.items()}
        docs_str = [" ".join(toks) for _, toks in docs_tokens.items()]
        vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")  # 已預切好，故用任意非空白
        X = vectorizer.fit_transform(docs_str)
        vocab = vectorizer.get_feature_names_out()
        per_file_counter: Dict[str, Counter] = {}
        per_file_set: Dict[str, Set[str]] = {}
        for i, p in enumerate(texts_by_file.keys()):
            row = X.getrow(i).toarray().ravel()
            idxs = row.argsort()[::-1][:topn]
            tokens = [vocab[j] for j in idxs if row[j] > 0]
            cnt = Counter({t: sum(1 for x in docs_tokens[p] if x == t) for t in tokens})
            per_file_counter[p] = cnt
            per_file_set[p] = set(cnt.keys())
        return per_file_counter, per_file_set
    else:
        per_file_counter = {p: Counter(tokenize(txt)) for p, txt in texts_by_file.items()}
        per_file_set = {p: set(cnt.keys()) for p, cnt in per_file_counter.items()}
        return per_file_counter, per_file_set


def compute_overlaps(per_file_counter: Dict[str, Counter], min_df: int=2) -> List[Tuple[str, int, int, Dict[str, int]]]:
    """彙整跨檔案的 token。回傳按 df->total 降序的列表。
    元素：(token, document_frequency, total_count, per_file_counts)
    """
    token_files: Dict[str, Dict[str, int]] = defaultdict(dict)
    for f, cnt in per_file_counter.items():
        for tok, c in cnt.items():
            token_files[tok][f] = token_files[tok].get(f, 0) + c

    rows: List[Tuple[str, int, int, Dict[str, int]]] = []
    for tok, perfile in token_files.items():
        df = len(perfile)
        if df >= min_df:
            total = sum(perfile.values())
            rows.append((tok, df, total, perfile))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows


def jaccard_matrix(per_file_set: Dict[str, Set[str]]) -> List[Tuple[str, str, float]]:
    files = list(per_file_set.keys())
    res = []
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a, b = files[i], files[j]
            sa, sb = per_file_set[a], per_file_set[b]
            inter = len(sa & sb)
            union = len(sa | sb) or 1
            res.append((a, b, inter/union))
    return res

# ---------------------- CLI ----------------------

def expand_paths(dir_: str=None, paths: List[str]=None, exts: List[str]=None) -> List[str]:
    exts = exts or [".txt", ".md", ".pdf"]
    out = []
    if paths:
        for p in paths:
            out.extend(glob.glob(p))
    if dir_:
        for root, _, files in os.walk(dir_):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    out.append(os.path.join(root, fn))
    # 去重並排序
    out = sorted(list(dict.fromkeys(out)))
    return out


def cmd_scan(args):
    paths = expand_paths(args.dir, args.paths, args.ext)
    if not paths:
        raise SystemExit("找不到可讀取的檔案。請用 --dir 或 --paths 指定。")

    per_counter, per_set = collect_tokens_per_file(paths, use_tfidf=args.use_tfidf, topn=args.topn)
    overlaps = compute_overlaps(per_counter, min_df=args.min_df)

    # 顯示前幾列
    print(f"共分析 {len(paths)} 個檔案；符合 min_df={args.min_df} 的詞數：{len(overlaps)}")
    head = overlaps[: args.show]
    for tok, df, total, perfile in head:
        files_str = ", ".join([f"{os.path.basename(f)}:{c}" for f, c in sorted(perfile.items())])
        print(f"{tok}\tDF={df}\tTOTAL={total}\t[{files_str}]")

    # 匯出 CSV
    if args.export:
        if not HAVE_PD:
            raise RuntimeError("需要 pandas 才能匯出 CSV，請 pip install pandas")
        rows = []
        for tok, df, total, perfile in overlaps:
            row = {"token": tok, "df": df, "total": total}
            for f in paths:
                row[os.path.basename(f)] = perfile.get(f, 0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"已匯出：{args.export}")

    # 額外：每檔 topN（若使用 TF‑IDF）
    if args.use_tfidf:
        print(f"\n各檔 top-{args.topn} TF‑IDF 關鍵詞（僅列前 5 檔）：")
        for i, (f, cnt) in enumerate(per_counter.items()):
            if i >= 5:
                print("...（其餘省略）")
                break
            tops = ", ".join([f"{t}({c})" for t, c in cnt.most_common(args.topn)[:10]])
            print(f"- {os.path.basename(f)}: {tops}")


def cmd_jaccard(args):
    paths = expand_paths(args.dir, args.paths, args.ext)
    if not paths:
        raise SystemExit("找不到可讀取的檔案。請用 --dir 或 --paths 指定。")
    per_counter, per_set = collect_tokens_per_file(paths, use_tfidf=args.use_tfidf, topn=args.topn)
    pairs = jaccard_matrix(per_set)
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("檔案兩兩 Jaccard（詞集合）相似度：")
    for a, b, j in pairs[: args.show]:
        print(f"{os.path.basename(a)} <-> {os.path.basename(b)}\t{j:.4f}")


def build_parser():
    p = argparse.ArgumentParser(description="Cross-file duplicate keywords scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan", help="掃描並輸出跨檔案重複關鍵詞")
    ps.add_argument("--dir", type=str, default=None, help="目錄（遞迴）")
    ps.add_argument("--paths", nargs="*", help="檔案或 glob，如 data/*.txt")
    ps.add_argument("--ext", nargs="*", default=[".txt", ".md", ".pdf"], help="要納入的副檔名")
    ps.add_argument("--min-df", type=int, default=2, help="至少出現在幾個檔案")
    ps.add_argument("--use-tfidf", action="store_true", help="改用每檔 TF‑IDF topN 再做交集")
    ps.add_argument("--topn", type=int, default=20, help="每檔取前 N 個關鍵詞（use-tfidf 時生效）")
    ps.add_argument("--show", type=int, default=50, help="終端列印前幾列")
    ps.add_argument("--export", type=str, default=None, help="匯出 CSV 路徑")
    ps.set_defaults(func=cmd_scan)

    pj = sub.add_parser("jaccard", help="輸出檔案兩兩 Jaccard 相似度")
    pj.add_argument("--dir", type=str, default=None)
    pj.add_argument("--paths", nargs="*")
    pj.add_argument("--ext", nargs="*", default=[".txt", ".md", ".pdf"])
    pj.add_argument("--use-tfidf", action="store_true")
    pj.add_argument("--topn", type=int, default=20)
    pj.add_argument("--show", type=int, default=50)
    pj.set_defaults(func=cmd_jaccard)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
