# main.py
import json
from json_to_csv import records_to_csv   # <-- 直接 import

def generate_records():
    yield {"title": "人工智慧導論", "general_subject": "AI"}
    yield {"title": "線性代數", "general_subject": ["Math", "Linear Algebra"]}

if __name__ == "__main__":
    records = list(generate_records())
    fields = ["title", "general_subject"]
    records_to_csv(records, fields, "result.csv")
    print("已輸出 result.csv")
