# pick_txts.py
import random, shutil
from pathlib import Path

# === 修改這裡 ===
SRC = Path(r"C:\path\to\your\folder")  # 放 .txt 的資料夾
N = 5                                   # 要隨機挑選的數量
MOVE = True                             # True=搬移；False=複製
# =================

SELECTED = SRC / "selected"
OTHERS = SRC / "others"
SELECTED.mkdir(exist_ok=True)
OTHERS.mkdir(exist_ok=True)

# 只抓 SRC 目錄下的 .txt（不含子資料夾）
files = [p for p in SRC.glob("*.txt") if p.is_file()]
if not files:
    raise SystemExit("找不到任何 .txt 檔")

k = min(N, len(files))
picked = set(random.sample(files, k=k))

def place(src_path: Path, dest_dir: Path):
    dest = dest_dir / src_path.name
    if dest.exists():  # 避免覆蓋：加上 _1, _2, ...
        i = 1
        stem, suf = src_path.stem, src_path.suffix
        while (dest_dir / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = dest_dir / f"{stem}_{i}{suf}"
    if MOVE:
        shutil.move(str(src_path), str(dest))
    else:
        shutil.copy2(str(src_path), str(dest))

for p in files:
    place(p, SELECTED if p in picked else OTHERS)

print(f"已放入 selected: {k} 個；others: {len(files)-k} 個")
print(f"selected => {SELECTED}")
print(f"others   => {OTHERS}")
