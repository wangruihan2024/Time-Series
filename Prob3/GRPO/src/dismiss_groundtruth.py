import json
import re

input_json_path = "/root/Prob3/Prob3/dataset/DJIA_prompt_1day_groundtruth.json"
output_json_path = "/root/Prob3/Prob3/dataset/DJIA_prompt_1day.json"

def strip_gt_from_instruction(instruction):
    lines = instruction.strip().split("\n")
    if lines and lines[-1].startswith("Next day's actual change is:"):
        lines = lines[:-1]
    return "\n".join(lines)

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for item in data:
    new_instruction = strip_gt_from_instruction(item["instruction"])
    new_data.append({
        "instruction": new_instruction,
        "output": item["output"]
    })

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"已生成新的 JSON 文件：{output_json_path}")