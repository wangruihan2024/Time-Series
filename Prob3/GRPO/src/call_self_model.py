import requests
import pandas as pd
import json
import time
from tqdm import tqdm

# === 配置 ===
api_url = "http://localhost:6006/v1/chat/completions"  # 本地 vLLM API
api_key = None  
model = "qwen-local"

csv_path = "/root/Prob3/Prob3/dataset/DJIA_with_change.csv"
output_json_path = "/root/Prob3/Prob3/dataset/DJIA_generate.json"

# === 读取数据 ===
df = pd.read_csv(csv_path)

# === 构建 instruction 字符串 ===
def build_instruction(past_rows):
    instr = (
        "Analyze the following news and stock price changes over the past 5 days, and generate a reasoning chain (CoT) "
        "that leads to the next 1 days' market trend. The output should be plain text reasoning and end with predicted "
        "changes for the next 1 days.\n\n"
    )
    for i, row in enumerate(past_rows):
        instr += f"---Day-{5 - i}---\n"
        for j in range(1, 26):
            instr += f"Top{j}: {row[f'Top{j}']}\n"
        instr += f"Change: {row['Change']}%\n\n"
    return instr

# === 调用 DeepSeek-R1 API ===
def call_local_model(instruction):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
        print(json.dumps(result, indent=2))
    else:
        print("API error:", response.status_code, response.text)
        return ""

# === 主循环，构建 prompts 并生成 CoT ===
n_total = len(df)
start_idx = max(5, int(n_total * 0.8))  
end_idx = n_total - 1

results = []
for idx in tqdm(range(start_idx, end_idx), desc="Generating prompts"):
    past_rows = df.iloc[idx - 5: idx].to_dict(orient="records")

    instruction = build_instruction(past_rows)
    model_output = call_local_model(instruction)

    results.append({
        "instruction": instruction,
        "output": model_output
    })

    # time.sleep(1.5)  # 防止请求过快被限速

# === 保存为 JSON 文件 ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n所有 prompt 和输出已保存到 {output_json_path}")


