import pandas as pd
import json

# 文件路径
csv_path = "dataset/DJIA_change.csv"
output_json_path = "dataset/example_instruction.json"

# 读取数据
df = pd.read_csv(csv_path)

# 构建 instruction
def build_instruction(past_rows):
    instr = (
        "Analyze the following news and stock price changes over the past 5 days, and generate a reasoning chain (CoT) "
        "that leads to the next 3 days' market trend. The output should be plain text reasoning and end with predicted "
        "changes for the next 3 days.\n\n"
    )
    for i, row in enumerate(past_rows):
        instr += f"---Day-{5 - i}---\n"
        for j in range(1, 26):
            instr += f"Top{j}: {row[f'Top{j}']}\n"
        instr += f"Change: {row['Change']}%\n\n"
    return instr

# 构建 ground_truth 输出
def build_output(gt_rows):
    return ", ".join([f"{row['Change']}%" for _, row in gt_rows.iterrows()])

# 取第一个样本
past_rows = df.iloc[0:5].to_dict(orient="records")
future_rows = df.iloc[5:8]

instruction = build_instruction(past_rows)
ground_truth = build_output(future_rows)

# 写入 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "instruction": instruction,
        "ground_truth": ground_truth
    }, f, indent=2, ensure_ascii=False)

print(f"✅ 示例 instruction 已写入 {output_json_path}")