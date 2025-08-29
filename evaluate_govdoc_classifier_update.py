#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate govdoc multilabel classifier directly on a CSV (text,label)
──────────────────────────────────────────────────────────────────
功能新增：
1) 匯出「僅錯誤樣本」清單（含：真實/預測複合標籤、原子差異、各原子分數）
   參數：--errors-out D:\govdata\pred_errors.csv
2) 產生「複合標籤」的混淆矩陣（off-diagonal 即誤填去向）
   參數：--confusion-out D:\govdata\confusion.csv
3) 列出 Top-K 常見誤填對（True→Pred）
   參數：--top-confusions 10

用法（Windows / PowerShell）：
    python evaluate_govdoc_classifier.py \
      --model D:\\govdata\\model_govdoc.joblib \
      --csv   D:\\govdata\\test.csv \
      --text-col text --label-col label \
      --export D:\\govdata\\preds_eval.csv \
      --errors-out D:\\govdata\\pred_errors.csv \
      --confusion-out D:\\govdata\\confusion.csv \
      --top-confusions 10

依賴：
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

# --- 與 govdoc_multilabel_classifier.py 一致的標籤邏輯 ---
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押"]
KNOWN_ATOMS = set(ATOMS + ["通知"])  # 通知 與 函 組成 通知函

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

# --- 推論流程 ---

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
    ap = argparse.ArgumentParser(description="Evaluate govdoc classifier on CSV (text,label) with error analysis")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--export", type=str, default=None, help="匯出逐列預測（含分數）")
    ap.add_argument("--errors-out", type=str, default=None, help="僅匯出錯誤樣本")
    ap.add_argument("--confusion-out", type=str, default=None, help="匯出複合標籤混淆矩陣")
    ap.add_argument("--top-confusions", type=int, default=10, help="列出前 K 大的誤填對")
    args = ap.parse_args()

    vec, clf, mlb, thr_model = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else thr_model

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(str).fillna("").tolist()

    # 真實原子標籤（對齊模型類別）
    y_atoms = [label_to_atoms(lb) for lb in labels]
    y_atoms = [[a for a in atoms if a in set(mlb.classes_)] for atoms in y_atoms]
    Y_true = mlb.transform(y_atoms)

    # 預測
    pred_atoms, proba = predict_atoms(texts, vec, clf, mlb, threshold)
    Y_pred = mlb.transform([[a for a in atoms if a in set(mlb.classes_)] for atoms in pred_atoms])

    # --- 原子層報告 ---
    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_true, Y_pred, target_names=mlb.classes_.tolist(), digits=4, zero_division=0))
    micro = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    print(f"Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f}")

    # --- 複合層（重組後）---
    gt_composites = labels
    pred_composites = [atoms_to_composite(a) for a in pred_atoms]

    comps = sorted(set(gt_composites) | set(pred_composites))
    comp_to_idx = {c: i for i, c in enumerate(comps)}
    y_comp_true = np.array([comp_to_idx[c] for c in gt_composites])
    y_comp_pred = np.array([comp_to_idx[c] for c in pred_composites])

    from sklearn.metrics import classification_report as clsrep
    print("\n[Composite-label evaluation]")
    print(clsrep(y_comp_true, y_comp_pred, target_names=comps, digits=4, zero_division=0))

    exact_acc = accuracy_score(y_comp_true, y_comp_pred)
    print(f"Exact-match Accuracy (composite): {exact_acc:.4f}")

    # --- 錯誤分析：誤填去向（混淆矩陣 + Top-K off-diagonal）---
    cm = pd.crosstab(pd.Series(gt_composites, name='True'), pd.Series(pred_composites, name='Pred'))
    if args.confusion_out:
        cm.to_csv(args.confusion_out, encoding='utf-8-sig')
        print(f"已匯出混淆矩陣：{args.confusion_out}")

    # 列出 Top-K 的 off-diagonal（真實!=預測）
    cm_off = cm.copy()
    for c in cm_off.columns:
        if c in cm_off.index:
            cm_off.loc[c, c] = 0
    flat = cm_off.stack().reset_index(name='count')
    flat = flat[flat['count'] > 0].sort_values('count', ascending=False)
    k = min(args.top_confusions, len(flat))
    if k > 0:
        print(f"\n[Top {k} 誤填對（True→Pred）]")
        for _, row in flat.head(k).iterrows():
            print(f"  {row['True']} → {row['Pred']}: {int(row['count'])}")

    # --- 匯出逐列預測（含分數）---
    rows = []
    for i, t in enumerate(texts):
        true_atoms_set = set(y_atoms[i])
        pred_atoms_set = set(pred_atoms[i])
        missing = sorted(true_atoms_set - pred_atoms_set)  # 漏掉的原子
        extra = sorted(pred_atoms_set - true_atoms_set)    # 多預測的原子
        row = {
            "text": t,
            "label_true": gt_composites[i],
            "label_pred": pred_composites[i],
            "atoms_true": "+".join(y_atoms[i]),
            "atoms_pred": "+".join(pred_atoms[i]),
            "missing_atoms": "+".join(missing),
            "extra_atoms": "+".join(extra),
        }
        for j, a in enumerate(mlb.classes_):
            row[f"score_{a}"] = float(proba[i, j])
        rows.append(row)
    pred_df = pd.DataFrame(rows)

    if args.export:
        pred_df.to_csv(args.export, index=False, encoding='utf-8-sig')
        print(f"已匯出逐列預測：{args.export}")

    if args.errors_out:
        err_df = pred_df[pred_df['label_true'] != pred_df['label_pred']]
        err_df.to_csv(args.errors_out, index=False, encoding='utf-8-sig')
        print(f"已匯出錯誤樣本：{args.errors_out}（{len(err_df)} 筆）")

if __name__ == "__main__":
    main()
