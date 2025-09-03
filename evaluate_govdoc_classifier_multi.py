#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate govdoc multilabel classifier directly on a CSV (text,label[,caption])
────────────────────────────────────────────────────────────────────────────
功能：
1) 原子/複合兩層評估（F1、Exact-match、per-class accuracy）
2) 匯出逐列預測（含各原子分數、缺漏/多預測原子）
3) 混淆矩陣與 Top-K 誤填對
4) **加入 caption 評估**（選擇性）：可從 CSV 讀取 caption 欄位，
   以與訓練相同的方式將 caption 併入 text（字元 n-gram + LR 相容）。
5) **Per-class 門檻（本版新增）**：
   以 CLASS_THRESHOLDS 設定各原子門檻；未列者使用 default。

用法（Windows / PowerShell）：
    python evaluate_govdoc_classifier.py \
      --model D:\govdata\model_govdoc.joblib \
      --csv   D:\govdata\test.csv \
      --text-col text --label-col label \
      --use-caption --caption-col caption --caption-weight 1.5 \
      --export D:\govdata\preds_eval.csv \
      --errors-out D:\govdata\pred_errors.csv \
      --confusion-out D:\govdata\confusion.csv \
      --perclass-out D:\govdata\per_class_acc.csv \
      --top-confusions 10

依賴：
    pip install -U pandas scikit-learn joblib
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Tuple
import math
import re

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

# --- 與訓練腳本一致的標籤邏輯 ---
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押命令"]
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
    "扣押命令": ["扣押命令"],
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
    (("扣押命令",), "扣押命令"),
]

# 註：這些是複合頭標籤名稱，不是原子
EXCLUSIVE_HEADS = {"扣押命令", "撤銷令", "收取令", "通知函"}

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
    # 互斥頭標籤（僅在 pred_atoms 本身含有複合名稱時才會命中；通常不會）
    heads = s & EXCLUSIVE_HEADS
    if heads:
        return next(iter(heads)) if len(heads) == 1 else sorted(heads)[0]
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"

# --- 後置覆寫規則：你要求的邏輯 ---
def post_override_route(text_original: str, composite: str) -> str:
    """
    若預測為『撤銷令』但原始文本未含『撤銷』兩字，則覆寫為『通知函』。
    - 僅檢查原始 OCR 文字（text_original），不含 caption。
    - 允許 OCR 斷字與簡體：撤\s*[銷销]
    """
    if composite == "撤銷令":
        if not re.search(r"撤\s*[銷销]", text_original):
            return "通知函"
    return composite

# --- Per-class thresholds（本版新增） ---
# 未列在此 dict 的原子，一律用 "default"
CLASS_THRESHOLDS: Dict[str, float] = {
    "default": 0.5,
    "扣押命令": 0.4,  # 你要的調整
}

# --- Caption 融合：與訓練一致 ---
def fuse_text_with_caption(text: str, caption: str | None, weight: float, tag: str = "[CAP]") -> str:
    text = (text or "").strip()
    cap = (caption or "").strip()
    if not cap or weight <= 0:
        return text
    repeats = max(1, int(math.ceil(weight)))
    block = (f" {tag} " + cap) * repeats
    return (text + block).strip()

# --- 推論流程 ---
def load_model(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    # 讀取模型中的 caption 設定（若無則採預設）
    use_caption_model = bool(obj.get("use_caption", False))
    caption_weight_model = float(obj.get("caption_weight", 1.0))
    cap_tag = obj.get("cap_tag", "[CAP]")
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return vec, clf, mlb, threshold, use_caption_model, caption_weight_model, cap_tag

def predict_atoms(texts: List[str], vec, clf, mlb, default_thr: float):
    """
    依 per-class 門檻做二值化：對於 mlb.classes_ 中每個原子，
    取 CLASS_THRESHOLDS.get(atom, default_thr) 作為其門檻。
    """
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    preds = np.zeros_like(proba, dtype=int)

    for j, atom in enumerate(mlb.classes_):
        thr = CLASS_THRESHOLDS.get(atom, default_thr)
        preds[:, j] = (proba[:, j] >= thr).astype(int)

    pred_atoms = []
    for i in range(len(texts)):
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        pred_atoms.append(atoms)
    return pred_atoms, proba

def main():
    ap = argparse.ArgumentParser(description="Evaluate govdoc classifier on CSV (text,label[,caption]) with error analysis")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    # Caption 評估選項
    ap.add_argument("--use-caption", action="store_true", help="啟用 caption；若未指定則沿用模型內設定")
    ap.add_argument("--caption-col", default="caption", help="CSV 中 caption 欄位名")
    ap.add_argument("--caption-weight", type=float, default=None, help="caption 權重（未指定則沿用模型內設定）")

    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--export", type=str, default=None, help="匯出逐列預測（含分數）")
    ap.add_argument("--errors-out", type=str, default=None, help="僅匯出錯誤樣本")
    ap.add_argument("--confusion-out", type=str, default=None, help="匯出複合標籤混淆矩陣")
    ap.add_argument("--top-confusions", type=int, default=10, help="列出前 K 大的誤填對")
    ap.add_argument("--perclass-out", type=str, default=None, help="匯出每類別正確率表（per-class accuracy），以及列出 macro/weighted 平均")
    args = ap.parse_args()

    vec, clf, mlb, thr_model, use_caption_model, caption_weight_model, cap_tag = load_model(args.model)

    # 門檻（作為 default，個別類別可被 CLASS_THRESHOLDS 覆蓋）
    default_threshold = args.threshold if args.threshold is not None else thr_model

    # 決定是否使用 caption 與權重（CLI 優先，否則沿用模型設定）
    use_caption = args.use_caption or use_caption_model
    caption_weight = caption_weight_model if args.caption_weight is None else float(args.caption_weight)

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts_raw = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(str).fillna("").tolist()

    # 讀取 caption 欄位（若存在且啟用）
    if use_caption and (args.caption_col in df.columns):
        caps_raw = df[args.caption_col].astype(str).fillna("").tolist()
        texts = [fuse_text_with_caption(t, c, caption_weight, tag=cap_tag) for t, c in zip(texts_raw, caps_raw)]
        non_empty_caps = sum(1 for c in caps_raw if str(c).strip())
        print(f"[Caption] 已啟用：欄位 '{args.caption_col}'，非空筆數 {non_empty_caps}/{len(caps_raw)}，權重 {caption_weight}")
    else:
        texts = texts_raw
        if use_caption:
            print(f"[Caption] 已啟用，但 CSV 缺少欄位 '{args.caption_col}'，將僅使用 text。")

    # 真實原子標籤（對齊模型類別）
    y_atoms = [label_to_atoms(lb) for lb in labels]
    y_atoms = [[a for a in atoms if a in set(mlb.classes_)] for atoms in y_atoms]
    Y_true = mlb.transform(y_atoms)

    # 預測（使用 per-class 門檻）
    pred_atoms, proba = predict_atoms(texts, vec, clf, mlb, default_threshold)
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

    # 後置覆寫：撤銷令但原始文內無「撤銷」→ 通知函
    pred_composites = [
        post_override_route(texts_raw[i], c)
        for i, c in enumerate(pred_composites)
    ]

    from sklearn.metrics import classification_report as clsrep
    comps = sorted(set(gt_composites) | set(pred_composites))
    comp_to_idx = {c: i for i, c in enumerate(comps)}
    y_comp_true = np.array([comp_to_idx.get(c, -1) for c in gt_composites])
    y_comp_pred = np.array([comp_to_idx.get(c, -1) for c in pred_composites])

    print("\n[Composite-label evaluation]")
    print(clsrep(y_comp_true, y_comp_pred, target_names=comps, digits=4, zero_division=0))

    exact_acc = accuracy_score(y_comp_true, y_comp_pred)
    print(f"Exact-match Accuracy (composite): {exact_acc:.4f}")

    # --- 錯誤分析：誤填去向（混淆矩陣 + Top-K off-diagonal）---
    cm = pd.crosstab(pd.Series(gt_composites, name='True'), pd.Series(pred_composites, name='Pred'))
    if args.confusion_out:
        cm.to_csv(args.confusion_out, encoding='utf-8-sig')
        print(f"已匯出混淆矩陣：{args.confusion_out}")

    # --- 依類別計算正確率（per-class accuracy），再取平均 ---
    classes = cm.index.tolist()
    diag = []
    support = []
    for c in classes:
        correct = int(cm.loc[c, c]) if c in cm.columns else 0
        total_c = int(cm.loc[c].sum())
        diag.append(correct)
        support.append(total_c)
    per_acc = [(diag[i] / support[i]) if support[i] > 0 else np.nan for i in range(len(classes))]

    per_class_df = pd.DataFrame({
        'class': classes,
        'support': support,
        'correct': diag,
        'per_class_accuracy': per_acc,
    })
    macro_per_class_acc = np.nanmean(per_class_df['per_class_accuracy'].values)
    weighted_per_class_acc = (per_class_df['correct'].sum() / max(1, per_class_df['support'].sum()))

    print("\n[Per-class accuracy]")
    for _, row in per_class_df.iterrows():
        acc_disp = "NA" if pd.isna(row['per_class_accuracy']) else f"{row['per_class_accuracy']:.4f}"
        print(f"  {row['class']}: acc={acc_disp} (support={row['support']})")
    print(f"Macro Avg Per-class Accuracy: {macro_per_class_acc:.4f}")
    print(f"Weighted Avg Per-class Accuracy: {weighted_per_class_acc:.4f}")

    if args.perclass_out:
        per_class_df.to_csv(args.perclass_out, index=False, encoding='utf-8-sig')
        print(f"已匯出每類別正確率：{args.perclass_out}")

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
