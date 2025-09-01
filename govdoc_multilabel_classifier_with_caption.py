#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府/保險公文 多標籤分類器（加入 caption 訓練與推論支援）
────────────────────────────────────────────────────────
- 既有：原子標籤學習（保單/公職/查詢/註記/撤銷/收取/令/函/扣押命令）+ 規則重組
- 新增：可從 CSV 的 caption 欄位讀入，或在 predict 提供 --caption 一起預測
- 權重：--caption-weight 可放大/縮小 caption 的影響（以文字重複方式近似加權）

常用：
# 訓練（有 caption 欄位）
python govdoc_multilabel_classifier.py train \
  --csv data/train.csv --text-col text --label-col label \
  --use-caption --caption-col caption --caption-weight 1.5 \
  --test-size 0.2 --out model_govdoc.joblib

# 預測（單句 + caption）
python govdoc_multilabel_classifier.py predict \
  --model model_govdoc.joblib \
  --text "主旨：核發扣押命令書" \
  --caption "影像說明：扣押命令樣式" \
  --export preds.csv
"""
from __future__ import annotations
import argparse
from typing import List, Dict, Tuple
import math

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
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押命令"]

LABEL_TO_ATOMS: Dict[str, List[str]] = {
    "保單查詢": ["保單", "查詢"],
    "保單註記": ["保單", "註記"],
    "保單查詢+註記": ["保單", "查詢", "註記"],
    "公職查詢": ["公職", "查詢"],
    "撤銷令": ["撤銷", "令"],
    "收取令": ["收取", "令"],
    "收取+撤銷": ["收取", "撤銷"],
    "通知函": ["通知", "函"],
    "扣押命令": ["扣押命令"],
}

KNOWN_ATOMS = set(ATOMS + ["通知"])  # for fallback parsing

RULES_RECOMPOSE: List[Tuple[Tuple[str, ...], str]] = [
    (("保單", "查詢", "註記"), "保單查詢+註記"),
    (("保單", "查詢"), "保單查詢"),
    (("保單", "註記"), "保單註記"),
    (("公職", "查詢"), "公職查詢"),
    (("撤銷", "令"), "撤銷令"),
    (("收取", "令"), "收取令"),
    (("收取", "撤銷"), "收取+撤銷"),
    (("通知", "函"), "通知函"),
    (("扣押命令",), "扣押命令"),
]

EXCLUSIVE_HEADS = {"扣押命令", "撤銷令", "收取令", "通知函"}


def label_to_atoms(label: str) -> List[str]:
    label = str(label).strip()
    if label in LABEL_TO_ATOMS:
        return LABEL_TO_ATOMS[label]
    parts = [p.strip() for p in label.replace("/", "+").split("+") if p.strip()]
    atoms: List[str] = []
    for p in parts:
        if p in KNOWN_ATOMS:
            atoms.append(p); continue
        for a in KNOWN_ATOMS:
            if a and a in p:
                atoms.append(a)
    return sorted(set(atoms))


def atoms_to_composite_constrained(pred_atoms: List[str], atom_scores: Dict[str, float] | None = None) -> str:
    s = set(pred_atoms)
    heads = s & EXCLUSIVE_HEADS
    if heads:
        if atom_scores and len(heads) > 1:
            return max(heads, key=lambda a: atom_scores.get(a, 0.0))
        return next(iter(heads))
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"

# ----------------------------- 向量化與模型 -----------------------------

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=200000,
        lowercase=False,
    )


def build_model() -> OneVsRestClassifier:
    base = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    return OneVsRestClassifier(base)

# ----------------------------- caption 合成工具 -----------------------------

def fuse_text_with_caption(text: str, caption: str | None, weight: float, tag: str = "[CAP]") -> str:
    """將 caption 以重複附加或標記方式併入 text，用重複次數近似權重。"""
    text = (text or "").strip()
    cap = (caption or "").strip()
    if not cap:
        return text
    # 以重複次數近似權重（至少 1 次）；小數部分轉為機率保留一次
    repeats = max(1, int(math.floor(weight)))
    frac = weight - math.floor(weight)
    extra = 1 if np.random.RandomState(42).rand() < frac else 0
    blocks = (f" {tag} " + cap) * (repeats + extra)
    return (text + blocks).strip()

# ----------------------------- 命令 -----------------------------

def cmd_train(args):
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts_raw = df[args.text_col].astype(str).fillna("").tolist()
    captions_raw = df[args.caption_col].astype(str).fillna("").tolist() if (args.use_caption and args.caption_col in df.columns) else [""] * len(texts_raw)
    texts = [fuse_text_with_caption(t, c, args.caption_weight) for t, c in zip(texts_raw, captions_raw)]

    raw_labels = df[args.label_col].astype(str).fillna("").tolist()
    y_atoms = [label_to_atoms(lb) for lb in raw_labels]

    mlb = MultiLabelBinarizer(classes=ATOMS + (["通知"] if "通知" not in ATOMS else []))
    Y = mlb.fit_transform(y_atoms)

    X_train, X_test, Y_train, Y_test = train_test_split(
        texts, Y, test_size=args.test_size, random_state=42,
        stratify=Y if Y.sum(axis=1).min() > 0 else None
    )

    vec = build_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = build_model()
    clf.fit(Xtr, Y_train)

    # 評估（原子層）
    Y_pred = (clf.predict_proba(Xte) >= args.threshold).astype(int)
    target_names = mlb.classes_.tolist()
    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_test, Y_pred, target_names=target_names, digits=4, zero_division=0))

    # 保存模型與 caption 設定
    obj = {
        "vectorizer": vec,
        "classifier": clf,
        "mlb_classes": target_names,
        "threshold": args.threshold,
        "use_caption": bool(args.use_caption),
        "caption_weight": float(args.caption_weight),
        "caption_col": args.caption_col,
        "cap_tag": "[CAP]",
    }
    joblib.dump(obj, args.out)
    print(f"模型已保存：{args.out}")


def _load(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    use_caption = bool(obj.get("use_caption", False))
    caption_weight = float(obj.get("caption_weight", 1.0))
    cap_tag = obj.get("cap_tag", "[CAP]")
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return vec, clf, mlb, threshold, use_caption, caption_weight, cap_tag


def classify_texts(texts: List[str], captions: List[str] | None, vec, clf, mlb, threshold: float,
                   use_caption: bool, caption_weight: float, cap_tag: str) -> List[Dict[str, object]]:
    if use_caption and captions is not None:
        texts = [fuse_text_with_caption(t, c, caption_weight, tag=cap_tag) for t, c in zip(texts, captions)]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    preds = (proba >= threshold).astype(int)
    results = []
    for i in range(len(texts)):
        atom_scores = {mlb.classes_[j]: float(proba[i, j]) for j in range(len(mlb.classes_))}
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        composite = atoms_to_composite_constrained(atoms, atom_scores)
        results.append({
            "text": texts[i],
            "atoms": atoms,
            "atom_scores": atom_scores,
            "composite": composite,
        })
    return results


def cmd_predict(args):
    vec, clf, mlb, threshold, use_caption, caption_weight, cap_tag = _load(args.model)

    if args.text:
        texts = [args.text]
        caps = [args.caption or ""] if use_caption else None
    elif args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        # 批次檔的 caption（若需要）可用 --caption-file 對應每行一個 caption
        caps = None
        if use_caption and args.caption_file:
            with open(args.caption_file, "r", encoding="utf-8") as f:
                caps = [ln.strip() for ln in f]
            if len(caps) != len(texts):
                raise SystemExit("caption_file 行數需與 infile 相同")
    else:
        raise SystemExit("請用 --text 或 --infile 提供輸入")

    res = classify_texts(texts, caps, vec, clf, mlb, args.threshold or threshold,
                         use_caption, caption_weight, cap_tag)

    if args.export:
        rows = []
        for r in res:
            row = {
                "text": r["text"],
                "composite": r["composite"],
                "atoms": "+".join(r["atoms"]),
            }
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
            top5 = sorted(r["atom_scores"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top scores:")
            for k, v in top5:
                print(f"  {k}: {v:.3f}")

# ----------------------------- CLI -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Gov/insurance doc multilabel classifier (atoms + recomposition + caption)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="訓練")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--text-col", default="text")
    pt.add_argument("--label-col", default="label")
    pt.add_argument("--use-caption", action="store_true", help="啟用 caption 作為額外文字來源")
    pt.add_argument("--caption-col", default="caption", help="CSV 中 caption 欄位名稱")
    pt.add_argument("--caption-weight", type=float, default=1.0, help="caption 權重（以重複文字近似，1.0=一次）")
    pt.add_argument("--test-size", type=float, default=0.2)
    pt.add_argument("--threshold", type=float, default=0.5)
    pt.add_argument("--out", default="model_govdoc.joblib")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="預測")
    pp.add_argument("--model", required=True)
    pp.add_argument("--text", type=str, default=None)
    pp.add_argument("--caption", type=str, default=None, help="單筆預測時的 caption 文字（可留空）")
    pp.add_argument("--infile", type=str, default=None)
    pp.add_argument("--caption-file", type=str, default=None, help="批次預測時，每行一個 caption 對應 infile")
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
