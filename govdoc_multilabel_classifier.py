#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府/保險公文 多標籤分類器（支援你提供的 8 類合成）
----------------------------------------------------------------
目標：把原始標籤（如：保單查詢、保單查詢+註記、撤銷令、收取+撤銷、通知函…）
拆成「原子標籤」再做多標籤學習：
原子標籤集合 = {保單, 公職, 查詢, 註記, 撤銷, 收取, 令, 函}

優點：
- 你有像「保單查詢+註記」「收取+撤銷」這種組合標籤時，多標籤更自然；
- 之後要新增新組合，只要在推論階段組回去，不必重訓整個多類別分類器。

依賴：
    pip install -U pandas scikit-learn joblib
（可選）中文分詞若要加入，可另外用 jieba；此處先採「字詞元」TF‑IDF（char n‑grams），
對中文實務上很強，且免分詞。

用法：
# 訓練與評估（單一 CSV，欄位：text, label）
python govdoc_multilabel_classifier.py train \
  --csv data/all.csv --text-col text --label-col label \
  --test-size 0.2 --out model_govdoc.joblib

# 預測（單句）
python govdoc_multilabel_classifier.py predict \
  --model model_govdoc.joblib --text "主旨：請查詢王OO之保單並加註注意事項"

# 批次預測（每行一筆）
python govdoc_multilabel_classifier.py predict \
  --model model_govdoc.joblib --infile new_docs.txt --export preds.csv

備註：
- 閾值（--threshold）預設 0.5。資料很少時可調 0.4；若要嚴謹可調 0.6。
- 匯出時同時給「原子標籤」與「組合標籤」兩種結果。
"""

from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# ----------------------------- 標籤處理 -----------------------------
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函"]

# 把你的現有合成標籤映射成原子標籤集合
LABEL_TO_ATOMS: Dict[str, List[str]] = {
    "保單查詢": ["保單", "查詢"],
    "保單註記": ["保單", "註記"],
    "保單查詢+註記": ["保單", "查詢", "註記"],
    "公職查詢": ["公職", "查詢"],
    "撤銷令": ["撤銷", "令"],
    "收取令": ["收取", "令"],
    "收取+撤銷": ["收取", "撤銷"],
    "通知函": ["通知", "函"],  # 注意："通知" 不是原子集合，稍後會自動擴充
}

# 若出現未在上表的標籤，做一個盡力拆解：以 + 分割，再把已知詞映射到原子，
# 也接受本身就是原子詞（如直接標成 "令"、"函"）。
KNOWN_ATOMS = set(ATOMS + ["通知"])  # 先把 "通知" 也視作可接受的原子詞


def label_to_atoms(label: str) -> List[str]:
    label = str(label).strip()
    if label in LABEL_TO_ATOMS:
        return LABEL_TO_ATOMS[label]
    parts = [p.strip() for p in label.replace("/", "+").split("+") if p.strip()]
    atoms: List[str] = []
    for p in parts:
        # 直接匹配常見片語
        if p in KNOWN_ATOMS:
            atoms.append(p)
            continue
        # 常見詞根（保單、公職、查詢、註記、撤銷、收取、令、函、通知）
        for a in KNOWN_ATOMS:
            if a and a in p:
                atoms.append(a)
    atoms = sorted(set(atoms))
    return atoms

# 把原子標籤組回你的合成標籤（供報表/相容用）
# 規則：
# - (保單, 查詢, 註記) -> 保單查詢+註記
# - (保單, 查詢) -> 保單查詢
# - (保單, 註記) -> 保單註記
# - (公職, 查詢) -> 公職查詢
# - (撤銷, 令) -> 撤銷令
# - (收取, 令) -> 收取令
# - (收取, 撤銷) -> 收取+撤銷
# - (通知, 函) -> 通知函
# 其餘：以 "+" 連接原子標籤作為 fallback


RULES_RECOMPOSE: List[Tuple[Tuple[str, ...], str]] = [
    (("保單", "查詢", "註記"), "保單查詢+註記"),
    (("保單", "查詢"), "保單查詢"),
    (("保單", "註記"), "保單註記"),
    (("公職", "查詢"), "公職查詢"),
    (("撤銷", "令"), "撤銷令"),
    (("收取", "令"), "收取令"),
    (("收取", "撤銷"), "收取+撤銷"),
    (("通知", "函"), "通知函"),
]


def atoms_to_composite(pred_atoms: List[str]) -> str:
    s = set(pred_atoms)
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"

# ----------------------------- 向量化與模型 -----------------------------
# 對中文/混合文本，char n-gram 是穩健基線；不用分詞也能抓到語塊。

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",  # 字元 n-gram（以單詞邊界為視窗，對混合文本較穩）
        ngram_range=(2, 5),   # 2~5 字元
        min_df=2,
        max_features=200000,
        lowercase=False,
    )


def build_model() -> OneVsRestClassifier:
    base = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    return OneVsRestClassifier(base)

# ----------------------------- 命令 -----------------------------

def cmd_train(args):
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    raw_labels = df[args.label_col].astype(str).fillna("").tolist()
    y_atoms = [label_to_atoms(lb) for lb in raw_labels]

    mlb = MultiLabelBinarizer(classes=ATOMS + (["通知"] if "通知" not in ATOMS else []))
    Y = mlb.fit_transform(y_atoms)

    X_train, X_test, Y_train, Y_test = train_test_split(
        texts, Y, test_size=args.test_size, random_state=42, stratify=Y if Y.sum(axis=1).min() > 0 else None
    )

    vec = build_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = build_model()
    clf.fit(Xtr, Y_train)

    # 評估（原子標籤層）
    Y_pred = (clf.predict_proba(Xte) >= args.threshold).astype(int)
    target_names = mlb.classes_.tolist()
    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_test, Y_pred, target_names=target_names, digits=4, zero_division=0))

    # 保存
    obj = {
        "vectorizer": vec,
        "classifier": clf,
        "mlb_classes": target_names,
        "threshold": args.threshold,
    }
    joblib.dump(obj, args.out)
    print(f"模型已保存：{args.out}")


def _load(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return vec, clf, mlb, threshold


def classify_texts(texts: List[str], vec, clf, mlb, threshold: float) -> List[Dict[str, object]]:
    X = vec.transform(texts)
    proba = clf.predict_proba(X)  # shape: (n_samples, n_classes)
    preds = (proba >= threshold).astype(int)
    results = []
    for i in range(len(texts)):
        atom_scores = {mlb.classes_[j]: float(proba[i, j]) for j in range(len(mlb.classes_))}
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        composite = atoms_to_composite(atoms)
        results.append({
            "text": texts[i],
            "atoms": atoms,
            "atom_scores": atom_scores,
            "composite": composite,
        })
    return results


def cmd_predict(args):
    vec, clf, mlb, threshold = _load(args.model)

    if args.text:
        texts = [args.text]
    elif args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        raise SystemExit("請用 --text 或 --infile 提供輸入")

    res = classify_texts(texts, vec, clf, mlb, args.threshold or threshold)

    if args.export:
        rows = []
        for r in res:
            row = {
                "text": r["text"],
                "composite": r["composite"],
                "atoms": "+".join(r["atoms"]),
            }
            # 也把各原子分數輸出
            for k, v in r["atom_scores"].items():
                row[f"score_{k}"] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"已匯出：{args.export}")
    else:
        for r in res:
            print("\n== 文本 ==\n" + r["text"])
            print("原子標籤：", ", ".join(r["atoms"]) or "(無)")
            print("組合標籤：", r["composite"])
            # 顯示最高的 5 個分數
            top5 = sorted(r["atom_scores"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top scores:")
            for k, v in top5:
                print(f"  {k}: {v:.3f}")


# ----------------------------- CLI -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Gov/insurance doc multilabel classifier (atoms + recomposition)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="訓練")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--text-col", default="text")
    pt.add_argument("--label-col", default="label")
    pt.add_argument("--test-size", type=float, default=0.2)
    pt.add_argument("--threshold", type=float, default=0.5)
    pt.add_argument("--out", default="model_govdoc.joblib")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="預測")
    pp.add_argument("--model", required=True)
    pp.add_argument("--text", type=str, default=None)
    pp.add_argument("--infile", type=str, default=None)
    pp.add_argument("--threshold", type=float, default=None)
    pp.add_argument("--export", type=str, default=None)
    pp.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
