import requests
import pandas as pd
import json
import time
from tqdm import tqdm

# === 配置 ===
api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # DeepSeek-R1 API endpoint
api_key = "sk-4451b25792374b2ebe24e81119233eaf"
model = "deepseek-r1"

csv_path = "dataset/DJIA_change.csv"
output_json_path = "dataset/DJIA_prompt.json"

# === 读取数据 ===
df = pd.read_csv(csv_path)

# === 构建 instruction 字符串 ===
def build_instruction(past_rows, gt_rows):
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

    instr += "Next 3 days' actual changes are: " + \
        ", ".join([f"{row['Change']}%" for _, row in gt_rows.iterrows()]) + "\n"
    return instr

# === 调用 DeepSeek-R1 API ===
def call_deepseek(instruction):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": instruction}
        ],
        "temperature": 0.7
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print("API error:", response.status_code, response.text)
        return ""

# === 主循环，构建 prompts 并生成 CoT ===
results = []
for idx in tqdm(range(5, 6), desc="Generating prompts"):
    past_rows = df.iloc[idx - 5: idx].to_dict(orient="records")
    future_rows = df.iloc[idx: idx + 3]

    instruction = build_instruction(past_rows, future_rows)
    model_output = call_deepseek(instruction)

    results.append({
        "instruction": instruction,
        "output": model_output
    })

    # time.sleep(1.5)  # 防止请求过快被限速

# === 保存为 JSON 文件 ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 所有 prompt 和输出已保存到 {output_json_path}")


