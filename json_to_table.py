# main.py
import json, csv, sys

def generate_records():
    # TODO: 這裡換成你的資料來源或既有邏輯
    yield {"title": "人工智慧導論", "general_subject": "AI"}
    yield {"title": "線性代數", "general_subject": ["Math", "Linear Algebra"]}

def pick(obj, path, joiner="|", default=""):
    # 簡版的 a.b.c / arr[0] 取值（沿用路徑 B 思路）
    cur = obj
    tokens, buf, i = [], "", 0
    while i < len(path):
        c = path[i]
        if c == '[':
            if buf: tokens.append(buf); buf = ""
            j = path.find(']', i); 
            if j == -1: return default
            tokens.append(path[i:j+1]); i = j+1
        elif c == '.':
            if buf: tokens.append(buf); buf = ""
            i += 1
        else:
            buf += c; i += 1
    if buf: tokens.append(buf)

    for tok in tokens:
        if tok.startswith('[') and tok.endswith(']'):
            if not isinstance(cur, list): return default
            idx = tok[1:-1]
            if not idx.isdigit(): return default
            idx = int(idx)
            if idx < 0 or idx >= len(cur): return default
            cur = cur[idx]
        else:
            if isinstance(cur, dict) and tok in cur:
                cur = cur[tok]
            elif isinstance(cur, list):
                # 未指定索引就合併
                return joiner.join(map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x,(dict,list)) else "" if x is None else str(x), cur))
            else:
                return default

    if isinstance(cur, list):
        return "|".join(map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x,(dict,list)) else "" if x is None else str(x), cur))
    if isinstance(cur, dict):
        return json.dumps(cur, ensure_ascii=False)
    return cur if cur is not None else default

def write_csv(out_path, fields, records, joiner="|", default=""):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({fld: pick(r, fld, joiner=joiner, default=default) for fld in fields})

if __name__ == "__main__":
    fields = ["title", "general_subject"]  # 你要的兩欄
    recs = list(generate_records())        # 或者你原本的記錄來源
    write_csv("result.csv", fields, recs)
    print("OK -> result.csv", file=sys.stderr)
