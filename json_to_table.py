# stdin_json_to_csv.py
import sys, json, csv, argparse

def parse_args():
    ap = argparse.ArgumentParser(description="從 STDIN 擷取欄位並輸出 CSV")
    ap.add_argument("-f", "--fields", nargs="+", default=["title", "general_subject"],
                    help="要擷取的欄位路徑（支援 a.b.c 與 arr[0]），預設：title general_subject")
    ap.add_argument("--join-arrays", default="|",
                    help="遇到陣列但未指定索引時的合併字元，預設 |")
    ap.add_argument("--default", default="",
                    help="欄位缺值時填入的預設字串，預設空字串")
    return ap.parse_args()

def read_stdin():
    buf = sys.stdin.read()
    return buf.strip()

def iter_records(buf):
    """
    支援三種輸入：
    1) JSON Lines (每行一筆物件)
    2) 單一 JSON 物件
    3) JSON 陣列（元素為物件）
    """
    if not buf:
        return []
    # 判斷 JSON Lines：多行且不像是以 '[' 開頭的大陣列
    looks_like_array = buf.lstrip().startswith("[")
    if ("\n" in buf) and not looks_like_array:
        recs = []
        for i, line in enumerate(buf.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"第 {i} 行不是物件 JSON")
            recs.append(obj)
        return recs

    data = json.loads(buf)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        for idx, it in enumerate(data):
            if not isinstance(it, dict):
                raise ValueError(f"陣列第 {idx} 個不是物件 JSON")
        return data
    raise ValueError("輸入需為物件、物件陣列或 JSON Lines")

def get_by_path(obj, path, joiner="|", default=""):
    """
    取值：a.b.c、items[0].price、tags（若 tags 是陣列且未索引 -> join）
    若不存在 -> default
    """
    cur = obj
    # 斷詞：把 a[0].b -> a, [0], b
    tokens = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == '[':
            if buf:
                tokens.append(buf); buf = ""
            j = path.find(']', i)
            if j == -1: return default
            tokens.append(path[i:j+1])
            i = j + 1
        elif c == '.':
            if buf:
                tokens.append(buf); buf = ""
            i += 1
        else:
            buf += c; i += 1
    if buf: tokens.append(buf)

    for tok in tokens:
        if tok.startswith("[") and tok.endswith("]"):
            if not isinstance(cur, list): return default
            idx_str = tok[1:-1]
            if not idx_str.isdigit(): return default
            idx = int(idx_str)
            if idx < 0 or idx >= len(cur): return default
            cur = cur[idx]
        else:
            if isinstance(cur, dict):
                if tok in cur:
                    cur = cur[tok]
                else:
                    return default
            elif isinstance(cur, list):
                # 未給索引，將整個陣列 join
                return join_list(cur, joiner)
            else:
                return default

    if isinstance(cur, list):
        return join_list(cur, joiner)
    if isinstance(cur, (dict,)):
        # 複合型別序列化，避免 CSV 爆欄
        return json.dumps(cur, ensure_ascii=False)
    return cur if cur is not None else default

def join_list(lst, joiner):
    parts = []
    for x in lst:
        if isinstance(x, (dict, list)):
            parts.append(json.dumps(x, ensure_ascii=False))
        elif x is None:
            parts.append("")
        else:
            parts.append(str(x))
    return joiner.join(parts)

def main():
    args = parse_args()
    buf = read_stdin()
    recs = iter_records(buf)
    if not recs:
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=args.fields)
    writer.writeheader()
    for r in recs:
        row = {fld: get_by_path(r, fld, joiner=args.join_arrays, default=args.default)
               for fld in args.fields}
        writer.writerow(row)

if __name__ == "__main__":
    main()
