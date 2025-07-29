import json
import csv
import re

input_path = "/root/Prob3/Prob3/dataset/DJIA_generate.json"
output_path = "/root/Prob3/Prob3/dataset/extracted_test.csv"

# 统一处理 Day+1 / Day 1 / Day1 的形式，并支持任意风格（*, **, 无）
pattern = re.compile(
    r"\*\*Day\s*\+?\s*1:\*\*.*?([+-]?\d+\.?\d*)%?"
)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# 仅测试前 3 条
for i, item in enumerate(data):
    output = item.get("output", "") if isinstance(item, dict) else ""
    value = "null"

    if "**Predicted Changes for Next 3 Days:**" in output:
        after_section = output.split("**Predicted Changes for Next 3 Days:**", 1)[1]
        for line in after_section.splitlines():
            line = line.strip()
            match = pattern.search(line)
            if match:
                try:
                    value = float(match.group(1))
                except:
                    value = "null"
                break

    print(f"[Sample {i}] Extracted Day+1 = {value}")
    results.append([i, value])

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "day+1_change"])
    writer.writerows(results)

print(f"✅ 提取完成，已保存前 {len(results)} 条样本到 {output_path}")