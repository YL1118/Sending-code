# -*- coding: utf-8 -*-
"""
extract_body.py
----------------
從 OCR 文字檔（.txt）抽取臺灣制式公文欄位，重點擷取正文（內文）。
改版：輸出結構改為「每類別一個資料夾，每個 .txt 產生同名 .json」。

使用方式：
  基本：
    python extract_body.py
  指定輸入/輸出資料夾：
    python extract_body.py --input-root output --output-root parsed
  僅處理特定子資料夾（可多個）：
    python extract_body.py --folders 保單查詢 通知函
  僅處理檔名包含關鍵字的 .txt：
    python extract_body.py --filename-contains 註記

輸出：
  parsed/保單查詢/<原檔名去副檔名>.json
  parsed/通知函/<原檔名去副檔名>.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# =========================
# 1) 欄位別名與容錯正規化
# =========================

FIELD_ALIASES = {
    "recipient":  ["受文者", "受文機關", "受文單位"],
    "doc_no":     ["發文字號", "文號", "案號", "來文字號"],
    "date":       ["發文日期", "日期", "中華民國"],
    "priority":   ["速別"],
    "security":   ["密等"],
    "subject":    ["主旨"],
    "body":       ["說明", "內文", "正文", "本文"],
    "attachment": ["附件"],
    "cc":         ["副本", "正本", "抄送"],
    "contact":    ["承辦", "承辦人", "聯絡", "聯絡電話", "連絡電話"],
}

CANONICAL_REPLACEMENTS = [
    ("王旨", "主旨"), ("圭旨", "主旨"),
    ("說朋", "說明"), ("說眀", "說明"),
    ("：", ":"), ("﹕", ":"), ("︰", ":"), ("：", ":"),
    ("　", " "), ("﻿", ""), ("\ufeff", ""),
    ("－", "-"), ("—", "-"), ("–", "-"),
]

def normalize_text(raw: str) -> str:
    text = raw
    for a, b in CANONICAL_REPLACEMENTS:
        text = text.replace(a, b)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if re.fullmatch(r"-{3,}|_{3,}|=+|~+|\d+/\d+|第\d+頁", s):
            continue
        s = re.sub(r"^\s*\(?\d{1,3}\)?\s+", "", s)
        cleaned.append(s)
    return "\n".join(cleaned)

def build_field_regex() -> re.Pattern:
    names: List[str] = []
    for _, alias in FIELD_ALIASES.items():
        names.extend(alias)
    names = sorted(set(names), key=lambda x: -len(x))
    pattern = r"^(?P<field>(" + "|".join(map(re.escape, names)) + r"))\s*:?\s*(?P<after>.*)$"
    return re.compile(pattern, flags=re.MULTILINE)

FIELD_PATTERN = build_field_regex()

# =========================
# 2) 核心抽取
# =========================

def split_sections(text: str) -> Dict[str, str]:
    matches = list(FIELD_PATTERN.finditer(text))
    if not matches:
        return {}
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        field_name = m.group("field")
        after = m.group("after").strip()
        block = (after + "\n" + text[start:end]).strip() if after else text[start:end].strip()
        block = block.strip()
        if field_name in sections and sections[field_name]:
            sections[field_name] = (sections[field_name] + "\n" + block).strip()
        else:
            sections[field_name] = block
    return sections

def canonicalize_keys(sections: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canon, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a in sections and sections[a].strip():
                out[canon] = sections[a].strip()
                break
    return out

def _span_of_field(field_cn: str, text: str) -> Optional[Tuple[int, int]]:
    pat = re.compile(r"^" + re.escape(field_cn) + r"\s*:?\s*(.*)$", flags=re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    start = m.end()
    nxt = FIELD_PATTERN.search(text, pos=start)
    end = nxt.start() if nxt else len(text)
    return (start, end)

def _middle_block(text: str) -> str:
    n = len(text)
    if n == 0:
        return ""
    return text[int(n * 0.2): int(n * 0.8)]

def heuristic_body(text: str, sections_raw: Dict[str, str], sections: Dict[str, str]) -> str:
    if "body" in sections and sections["body"].strip():
        return sections["body"].strip()

    tail_markers = FIELD_ALIASES["attachment"] + FIELD_ALIASES["cc"] + FIELD_ALIASES["contact"]

    subject_span = _span_of_field("主旨", text)
    if subject_span:
        start = subject_span[1]
        tail_positions = []
        for t in tail_markers:
            sp = _span_of_field(t, text)
            if sp and sp[0] > start:
                tail_positions.append(sp[0])
        end = min(tail_positions) if tail_positions else len(text)
        body = text[start:end].strip()
        body = re.sub(FIELD_PATTERN, "", body).strip()
        if body:
            return body

    m = re.search(r"^[（(]?(一|二|三|四|五|六|七|八|九|十)[)）]?[、.．]", text, flags=re.MULTILINE)
    if m:
        return text[m.start():].strip()

    return _middle_block(text).strip()

def extract_body_from_txt(txt_path: str, encoding: str = "utf-8") -> Dict[str, object]:
    with open(txt_path, "r", encoding=encoding, errors="ignore") as f:
        raw = f.read()
    norm = normalize_text(raw)
    sections_raw = split_sections(norm)
    sections = canonicalize_keys(sections_raw)
    body = heuristic_body(norm, sections_raw, sections)

    return {
        "subject": sections.get("subject", ""),
        "body": body,
        "attachment": sections.get("attachment", ""),
        "meta": {
            "recipient": sections.get("recipient", ""),
            "doc_no": sections.get("doc_no", ""),
            "date": sections.get("date", ""),
            "priority": sections.get("priority", ""),
            "security": sections.get("security", ""),
        }
    }

# =========================
# 3) 批次處理與 CLI（逐檔輸出）
# =========================

def process_all(
    input_root: Path,
    output_root: Path,
    only_folders: Optional[List[str]] = None,
    filename_contains: Optional[str] = None,
    encoding: str = "utf-8"
) -> None:
    """
    遍歷 input_root 下所有子資料夾與 .txt，
    對每個類別建立 output_root/<類別>/，
    並將每個 .txt 轉為同名 .json （一檔一 JSON）。
    """
    output_root.mkdir(exist_ok=True, parents=True)

    folders = [p for p in input_root.iterdir() if p.is_dir()]
    if only_folders:
        target_set = set(only_folders)
        folders = [p for p in folders if p.name in target_set]

    if not folders:
        print(f"⚠️ 找不到子資料夾可處理（root: {input_root}）")
        return

    for folder in sorted(folders, key=lambda p: p.name):
        category = folder.name
        out_dir = output_root / category
        out_dir.mkdir(exist_ok=True, parents=True)

        txt_files = sorted(folder.glob("*.txt"))
        if filename_contains:
            txt_files = [p for p in txt_files if filename_contains in p.name]

        if not txt_files:
            print(f"ℹ️ 類別「{category}」沒有符合條件的 .txt 檔，略過。")
            continue

        count_ok, count_err = 0, 0
        for txt_path in txt_files:
            try:
                result = extract_body_from_txt(str(txt_path), encoding=encoding)
                result["filename"] = txt_path.name
                result["category"] = category

                out_path = out_dir / (txt_path.stem + ".json")
                with open(out_path, "w", encoding="utf-8") as fout:
                    json.dump(result, fout, ensure_ascii=False, indent=2)

                count_ok += 1
            except Exception as e:
                print(f"❌ 處理失敗：{txt_path} -> {e}")
                count_err += 1

        print(f"✅ 類別「{category}」完成：成功 {count_ok} 檔，失敗 {count_err} 檔 -> {out_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取 OCR 公文內文與欄位（逐檔輸出 JSON）"
    )
    parser.add_argument("--input-root", type=str, default="output",
                        help="輸入根資料夾（底下每個子資料夾是一種類別）")
    parser.add_argument("--output-root", type=str, default="parsed",
                        help="輸出根資料夾（底下會建立 <類別>/ 子資料夾）")
    parser.add_argument("--folders", nargs="*", default=None,
                        help="僅處理指定子資料夾名稱（可多個），預設處理全部")
    parser.add_argument("--filename-contains", type=str, default=None,
                        help="僅處理檔名包含此關鍵字的 .txt")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="輸入檔案編碼（預設 utf-8）")
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        print(f"❌ 找不到輸入資料夾：{input_root.resolve()}")
        return

    print(f"🚀 開始處理：input_root={input_root.resolve()}  ->  output_root={output_root.resolve()}")
    if args.folders:
        print(f"   只處理子資料夾：{', '.join(args.folders)}")
    if args.filename_contains:
        print(f"   只處理檔名包含：{args.filename_contains}")
    process_all(
        input_root=input_root,
        output_root=output_root,
        only_folders=args.folders,
        filename_contains=args.filename_contains,
        encoding=args.encoding
    )
    print("🎉 全部完成。")

if __name__ == "__main__":
    main()
