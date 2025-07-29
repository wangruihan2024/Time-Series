import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 路径 ===
input_path = '/Users/wang/Documents/StockForcast/Prob3/GRPO/Prob3/DJIA_estimate_vs_truth.csv'
output_path = '/Users/wang/Documents/StockForcast/Prob3/GRPO/Prob3/eval_result.txt'

# === 读取数据 ===
df = pd.read_csv(input_path)

# === 提取数据列 ===
y_pred = df['estimate_close'].values
y_true = df['ground_truth'].values
y_prev = df['prev_close'].values

# === 计算各项指标 ===
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

true_dir = np.sign(y_true - y_prev)
pred_dir = np.sign(y_pred - y_prev)
acc = np.mean(true_dir == pred_dir)

true_return_pct = (y_true - y_prev) / y_prev
money = 1.0
for i in range(len(y_true)):
    if pred_dir[i] > 0:
        money *= (1 + true_return_pct[i])
cr = (money - 1) * 100  # 累计收益率百分比

strategy_return = np.where(pred_dir > 0, y_true - y_prev, 0)
sr = np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) != 0 else np.nan

# === 输出结果 ===
with open(output_path, "w") as f:
    f.write(f"评估样本数: {len(y_true)}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"CR: {cr:.2f}%\n")
    f.write(f"SR:  {sr:.4f}\n")

print(f"评估结果已保存至：{output_path}")