import requests
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 配置 ===
api_url = "https://api.deepseek.com/v1/chat/completions"
api_key = "sk-0c426db9df5b4162a43fea9a9ffab13c"
model = "deepseek-reasoner"

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

# === 请求函数 ===
def call_deepseek_with_data(idx):
    past_rows = df.iloc[idx - 5: idx].to_dict(orient="records")
    future_rows = df.iloc[idx: idx + 3]
    instruction = build_instruction(past_rows, future_rows)

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

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        result = response.json()
        return {
            "instruction": instruction,
            "output": result["choices"][0]["message"]["content"]
        }
    except Exception as e:
        return {
            "instruction": instruction,
            "output": f"[ERROR] {str(e)}"
        }

# === 多线程并发调用 ===
results = []
with ThreadPoolExecutor(max_workers=10) as executor:  # 可以调大到 10 或 20，根据 API 限制----int(len(df) * 0.8) - 3)
    future_to_idx = {executor.submit(call_deepseek_with_data, idx): idx for idx in range(5, int(len(df) * 0.8) - 3)}

    for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Generating prompts"):
        results.append(future.result())

# === 保存为 JSON 文件 ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nprompt和输出已保存到 {output_json_path}")