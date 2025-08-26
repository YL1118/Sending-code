# json_to_csv_min.py
# 極簡：records(Iterable[dict]) + fields(List[str]) -> CSV 檔
import csv, json

def write_csv_from_records(records, fields, out_path, joiner="|", default=""):
    def _join_list(lst):
        parts = []
        for x in lst:
            if isinstance(x, (dict, list)):
                parts.append(json.dumps(x, ensure_ascii=False))
            elif x is None:
                parts.append("")
            else:
                parts.append(str(x))
        return joiner.join(parts)

    def _get_by_path(obj, path):
        cur = obj
        tokens, buf, i = [], "", 0
        while i < len(path):
            c = path[i]
            if c == '[':
                if buf: tokens.append(buf); buf = ""
                j = path.find(']', i)
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
                    # 未指定索引 -> 直接合併整個 list
                    return _join_list(cur)
                else:
                    return default

        if isinstance(cur, list):  # 結尾仍是 list -> 合併
            return _join_list(cur)
        if isinstance(cur, dict):  # dict -> JSON 字串
            return json.dumps(cur, ensure_ascii=False)
        return default if cur is None else cur

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in records:
            row = {fld: _get_by_path(r, fld) for fld in fields}
            w.writerow(row)
