#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨類別重複關鍵字清除器（CSV 專用）
-------------------------------------------------
目標：
- 你有「每個類別一個 CSV」或「單一 CSV 內含 label 欄位」。
- 找出「出現在至少 K 個類別」的關鍵字（可用 TF‑IDF 取每類 top‑N 再做交集），
  然後**從各類的文字欄位中移除這些跨類重複關鍵字**，輸出清洗後 CSV。

安裝：
    pip install -U pandas scikit-learn jieba

用法一（每個類別一個 CSV，檔名即類別名，或用 --category-col 指定固定欄位）：
    python cross_category_keyword_pruner.py multi \
        --paths data/*.csv \
        --text-col text \
        --min-cat-df 2 \
        --use-tfidf --topn 50 \
        --outdir cleaned/

用法二（單一 CSV，含 label 欄）：
    python cross_category_keyword_pruner.py single \
        --csv data/all.csv --label-col label --text-col text \
        --min-cat-df 2 --use-tfidf --topn 50 \
        --outdir cleaned/

注意：
- token 化：預設偵測中文時用 jieba，否則使用 regex 切詞；會做小寫化與簡易停用詞過濾。
- 重建文字：以空白把 tokens 接回（中英混合時最穩定；若要中文無空白可改程式或加旗標）。
- 你可以加自己的停用詞或同義詞映射以更精準（程式內留擴充點）。
"""

from __future__ import annotations
import argparse
import os
import re
from glob import glob
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Set

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import jieba  # type: ignore
    HAVE_JIEBA = True
except Exception:
    HAVE_JIEBA = False

# ---------------------- 基本工具 ----------------------
EN_STOP = set(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with""".split()
)
ZH_STOP = {
    "的","了","呢","嗎","哦","喔","在","是","有","與","和","及","以及","並","或","而","及其",
    "我","你","他","她","它","我們","你們","他們","這","那","哪","個","些","如果","因為","所以",
    "而且","但是","然而","以及","並且","等","就","都","也","還","很","更","最","請問","可以","是否",
}
RE_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+|[0-9]+|[\u4e00-\u9fff]{1,}")
RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

# 可自訂同義詞映射；例如 {"加簽":"加選", "申請":"申請"}
SYN_MAP: Dict[str, str] = {}


def normalize_token(t: str) -> str:
    t = t.lower().strip()
    if t.isdigit():
        return ""  # 去數字
    # 同義詞映射
    if t in SYN_MAP:
        t = SYN_MAP[t]
    return t


def tokenize(text: str) -> List[str]:
    text = text if isinstance(text, str) else str(text)
    has_cjk = bool(RE_HAS_CJK.search(text))
    if has_cjk and HAVE_JIEBA:
        toks = [w for w in jieba.lcut(text) if w.strip()]
    else:
        toks = RE_WORD.findall(text)
    out: List[str] = []
    for t in toks:
        tt = normalize_token(t)
        if not tt:
            continue
        if RE_HAS_CJK.search(tt):
            if len(tt) < 2 or tt in ZH_STOP:
                continue
        else:
            if len(tt) < 2 or tt in EN_STOP:
                continue
        out.append(tt)
    return out

# ---------------------- 主邏輯 ----------------------

def per_category_token_sets(cat_to_texts: Dict[str, List[str]], use_tfidf: bool, topn: int) -> Dict[str, Set[str]]:
    """回傳每類的 token 集合。若 use_tfidf，僅取該類 top‑N 關鍵詞。"""
    if use_tfidf:
        if not HAVE_SK:
            raise RuntimeError("需要 scikit-learn 才能使用 --use-tfidf")
        # 先把每句 token 化，之後用 token 序列組回空白分隔字串，便於 TF‑IDF
        docs_by_cat = {c: [" ".join(tokenize(t)) for t in texts] for c, texts in cat_to_texts.items()}
        all_docs = []
        doc_cats = []
        for c, docs in docs_by_cat.items():
            for d in docs:
                if d:
                    all_docs.append(d)
                    doc_cats.append(c)
        if not all_docs:
            return {c: set() for c in cat_to_texts}
        vec = TfidfVectorizer(token_pattern=r"[^\s]+")
        X = vec.fit_transform(all_docs)
        vocab = vec.get_feature_names_out()
        # 聚合每類 TF‑IDF 分數並取 topN
        cat_scores: Dict[str, Counter] = defaultdict(Counter)
        for i, c in enumerate(doc_cats):
            row = X.getrow(i).toarray().ravel()
            cat_scores[c].update({vocab[j]: row[j] for j in row.nonzero()[0]})
        cat_sets = {}
        for c, ctr in cat_scores.items():
            tops = [w for w, _ in ctr.most_common(topn)]
            cat_sets[c] = set(tops)
        return cat_sets
    else:
        return {c: set(token for txt in texts for token in tokenize(txt)) for c, texts in cat_to_texts.items()}


def cross_category_common_terms(cat_sets: Dict[str, Set[str]], min_cat_df: int) -> Set[str]:
    df: Dict[str, int] = defaultdict(int)
    for s in cat_sets.values():
        for tok in s:
            df[tok] += 1
    return {tok for tok, d in df.items() if d >= min_cat_df}


def remove_terms_from_text(text: str, banned: Set[str]) -> str:
    toks = tokenize(text)
    kept = [t for t in toks if t not in banned]
    # 以空白接回，對中英混合較穩定；若要中文黏回可改 "".join(kept)
    return " ".join(kept)


def clean_category_frames(cat_to_df: Dict[str, pd.DataFrame], text_col: str, banned: Set[str], drop_empty: bool) -> Dict[str, pd.DataFrame]:
    cleaned: Dict[str, pd.DataFrame] = {}
    for cat, df in cat_to_df.items():
        df2 = df.copy()
        if text_col not in df2.columns:
            raise ValueError(f"找不到文字欄位：{text_col}（類別 {cat}）")
        df2[text_col] = df2[text_col].astype(str).map(lambda x: remove_terms_from_text(x, banned))
        if drop_empty:
            df2 = df2[df2[text_col].str.strip().astype(bool)]
        cleaned[cat] = df2
    return cleaned


def save_category_frames(frames: Dict[str, pd.DataFrame], outdir: str, suffix: str = "_cleaned") -> None:
    os.makedirs(outdir, exist_ok=True)
    for cat, df in frames.items():
        safe = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(cat))[:120]
        path = os.path.join(outdir, f"{safe}{suffix}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"已輸出：{path}（{len(df)} 列）")

# ---------------------- 兩種模式 ----------------------

def run_multi(args):
    # 檔名視為類別名，或若提供 --category-col，則讀取檔案中該欄做類別名（每檔單一值）。
    paths = []
    for p in args.paths:
        paths.extend(glob(p))
    if not paths:
        raise SystemExit("找不到 CSV，請確認 --paths")

    cat_to_df: Dict[str, pd.DataFrame] = {}
    cat_to_texts: Dict[str, List[str]] = {}

    for p in sorted(set(paths)):
        df = pd.read_csv(p)
        if args.category_col:
            if args.category_col not in df.columns:
                raise ValueError(f"{p} 缺少欄位：{args.category_col}")
            cats = df[args.category_col].dropna().unique().tolist()
            if len(cats) != 1:
                raise ValueError(f"{p} 的 {args.category_col} 不唯一，請改用 single 模式或先拆檔")
            cat = str(cats[0])
        else:
            # 用檔名（去副檔名）做類別
            cat = os.path.splitext(os.path.basename(p))[0]
        if args.text_col not in df.columns:
            raise ValueError(f"{p} 找不到文字欄位：{args.text_col}")
        cat_to_df[cat] = df
        cat_to_texts[cat] = df[args.text_col].astype(str).tolist()

    cat_sets = per_category_token_sets(cat_to_texts, args.use_tfidf, args.topn)
    banned = cross_category_common_terms(cat_sets, args.min_cat_df)
    print(f"跨類別（min_cat_df={args.min_cat_df}）重複關鍵字數：{len(banned)}")

    cleaned = clean_category_frames(cat_to_df, args.text_col, banned, args.drop_empty)
    save_category_frames(cleaned, args.outdir, suffix=args.suffix)


def run_single(args):
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise SystemExit(f"找不到標籤欄位：{args.label_col}")
    if args.text_col not in df.columns:
        raise SystemExit(f"找不到文字欄位：{args.text_col}")

    # 收集每類文本
    cat_to_texts: Dict[str, List[str]] = defaultdict(list)
    for _, row in df[[args.label_col, args.text_col]].iterrows():
        cat_to_texts[str(row[args.label_col])].append(str(row[args.text_col]))

    cat_sets = per_category_token_sets(cat_to_texts, args.use_tfidf, args.topn)
    banned = cross_category_common_terms(cat_sets, args.min_cat_df)
    print(f"跨類別（min_cat_df={args.min_cat_df}）重複關鍵字數：{len(banned)}")

    # 逐類清洗並各自輸出
    out_frames: Dict[str, pd.DataFrame] = {}
    for cat, grp in df.groupby(args.label_col):
        df2 = grp.copy()
        df2[args.text_col] = df2[args.text_col].astype(str).map(lambda x: remove_terms_from_text(x, banned))
        if args.drop_empty:
            df2 = df2[df2[args.text_col].str.strip().astype(bool)]
        out_frames[str(cat)] = df2
    save_category_frames(out_frames, args.outdir, suffix=args.suffix)

# ---------------------- CLI ----------------------

def build_parser():
    p = argparse.ArgumentParser(description="Remove cross-category duplicate keywords from CSVs and export per-category cleaned CSVs")
    sub = p.add_subparsers(dest="cmd", required=True)

    # multi-files 模式
    pm = sub.add_parser("multi", help="多個 CSV，各代表一個類別")
    pm.add_argument("--paths", nargs="+", help="CSV 路徑或 glob，如 data/*.csv")
    pm.add_argument("--text-col", required=True, help="文字欄位名")
    pm.add_argument("--category-col", default=None, help="（可選）CSV 內的固定類別欄位名")
    pm.add_argument("--min-cat-df", type=int, default=2, help="至少出現在幾個類別才當作跨類重複關鍵字")
    pm.add_argument("--use-tfidf", action="store_true", help="每類先以 TF‑IDF 取 topN 再做交集")
    pm.add_argument("--topn", type=int, default=50, help="TF‑IDF 模式下每類取前 N 關鍵詞")
    pm.add_argument("--drop-empty", action="store_true", help="清洗後若文字為空則刪除列")
    pm.add_argument("--outdir", required=True, help="輸出資料夾")
    pm.add_argument("--suffix", default="_cleaned", help="輸出檔名後綴")
    pm.set_defaults(func=run_multi)

    # single-file 模式
    ps = sub.add_parser("single", help="單一 CSV，含 label 欄位，按類別輸出多檔")
    ps.add_argument("--csv", required=True, help="CSV 路徑")
    ps.add_argument("--label-col", required=True, help="標籤欄位名")
    ps.add_argument("--text-col", required=True, help="文字欄位名")
    ps.add_argument("--min-cat-df", type=int, default=2)
    ps.add_argument("--use-tfidf", action="store_true")
    ps.add_argument("--topn", type=int, default=50)
    ps.add_argument("--drop-empty", action="store_true")
    ps.add_argument("--outdir", required=True)
    ps.add_argument("--suffix", default="_cleaned")
    ps.set_defaults(func=run_single)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
