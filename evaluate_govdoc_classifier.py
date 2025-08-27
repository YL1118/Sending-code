#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate govdoc multilabel classifier directly on a CSV (text,label)
-------------------------------------------------------------------
- Loads the trained model (saved by govdoc_multilabel_classifier.py)
- Reads test CSV (columns: text, label)
- Converts composite labels to atomic labels, runs prediction, and reports:
  * Atomic-level multilabel metrics (macro/micro F1, per-label report)
  * Composite-level multiclass report (via recomposition)
- Optionally exports per-row predictions to a CSV.

Usage (Windows / PowerShell):
    python evaluate_govdoc_classifier.py \
      --model D:\govdata\model_govdoc.joblib \
      --csv   D:\govdata\test.csv \
      --text-col text --label-col label \
      --export D:\govdata\preds_eval.csv

Dependencies:
    pip install -U pandas scikit-learn joblib
"""
from __future__ import annotations
import argparse
from typing import List, Dict, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

# --- Keep label logic in sync with govdoc_multilabel_classifier.py ---
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押"]
KNOWN_ATOMS = set(ATOMS + ["通知"])  # include 通知 as atomic token, paired with 函 to form 通知函

LABEL_TO_ATOMS: Dict[str, List[str]] = {
    "保單查詢": ["保單", "查詢"],
    "保單註記": ["保單", "註記"],
    "保單查詢+註記": ["保單", "查詢", "註記"],
    "公職查詢": ["公職", "查詢"],
    "撤銷令": ["撤銷", "令"],
    "收取令": ["收取", "令"],
    "收取+撤銷": ["收取", "撤銷"],
    "通知函": ["通知", "函"],
    "扣押命令": ["扣押", "令"],
}

RULES_RECOMPOSE: List[Tuple[Tuple[str, ...], str]] = [
    (("保單", "查詢", "註記"), "保單查詢+註記"),
    (("保單", "查詢"), "保單查詢"),
    (("保單", "註記"), "保單註記"),
    (("公職", "查詢"), "公職查詢"),
    (("撤銷", "令"), "撤銷令"),
    (("收取", "令"), "收取令"),
    (("收取", "撤銷"), "收取+撤銷"),
    (("通知", "函"), "通知函"),
    (("扣押", "令"), "扣押命令"),
]

def label_to_atoms(label: str) -> List[str]:
    label = str(label).strip()
    if label in LABEL_TO_ATOMS:
        return LABEL_TO_ATOMS[label]
    parts = [p.strip() for p in label.replace("/", "+").split("+") if p.strip()]
    atoms: List[str] = []
    for p in parts:
        if p in KNOWN_ATOMS:
            atoms.append(p)
            continue
        for a in KNOWN_ATOMS:
            if a and a in p:
                atoms.append(a)
    return sorted(set(atoms))

def atoms_to_composite(pred_atoms: List[str]) -> str:
    s = set(pred_atoms)
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"

# --- Eval pipeline ---

def load_model(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return vec, clf, mlb, threshold


def predict_atoms(texts: List[str], vec, clf, mlb, threshold: float):
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    preds = (proba >= threshold).astype(int)
    pred_atoms = []
    for i in range(len(texts)):
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        pred_atoms.append(atoms)
    return pred_atoms, proba


def main():
    ap = argparse.ArgumentParser(description="Evaluate govdoc classifier on CSV (text,label)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--export", type=str, default=None, help="Export per-row predictions to CSV")
    args = ap.parse_args()

    vec, clf, mlb, thr_model = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else thr_model

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(str).fillna("").tolist()

    # Ground truth atoms aligned to model classes
    y_atoms = [label_to_atoms(lb) for lb in labels]
    # Filter atoms to model's classes
    y_atoms = [[a for a in atoms if a in set(mlb.classes_)] for atoms in y_atoms]
    Y_true = mlb.transform(y_atoms)

    # Predictions
    pred_atoms, proba = predict_atoms(texts, vec, clf, mlb, threshold)
    Y_pred = mlb.transform([[a for a in atoms if a in set(mlb.classes_)] for atoms in pred_atoms])

    # Atomic-level report
    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_true, Y_pred, target_names=mlb.classes_.tolist(), digits=4, zero_division=0))

    micro = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    print(f"Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f}")

    # Composite-level report
    gt_composites = labels
    pred_composites = [atoms_to_composite(a) for a in pred_atoms]

    # Build composite label set from ground truth + predictions
    comps = sorted(set(gt_composites) | set(pred_composites))
    # Convert to indices
    comp_to_idx = {c:i for i,c in enumerate(comps)}
    y_comp_true = np.array([comp_to_idx[c] for c in gt_composites])
    y_comp_pred = np.array([comp_to_idx[c] for c in pred_composites])

    print("\n[Composite-label evaluation]")
    # Per-class report via pandas crosstab (optional) and sklearn report
    from sklearn.metrics import classification_report as clsrep
    print(clsrep(y_comp_true, y_comp_pred, target_names=comps, digits=4, zero_division=0))

    exact_acc = accuracy_score(y_comp_true, y_comp_pred)
    print(f"Exact-match Accuracy (composite): {exact_acc:.4f}")

    if args.export:
        rows = []
        for i, t in enumerate(texts):
            row = {
                "text": t,
                "label_true": gt_composites[i],
                "label_pred": pred_composites[i],
                "atoms_true": "+".join(y_atoms[i]),
                "atoms_pred": "+".join(pred_atoms[i]),
            }
            # also store top scores for transparency
            for j, a in enumerate(mlb.classes_):
                row[f"score_{a}"] = float(proba[i, j])
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"已匯出逐列預測：{args.export}")

if __name__ == "__main__":
    main()
