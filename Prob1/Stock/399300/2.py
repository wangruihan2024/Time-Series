import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 设置使用前 N 天均值
N = 6

# 读取数据
df = pd.read_csv("Stock/399300/dataset/399300.csv", parse_dates=["date"])
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

output_path = "Stock/399300/prob1/2.out"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df["real_close"] = df["OT"]
df["pred_close"] = df["real_close"].shift(1).rolling(window=N).mean()
df["prev_close"] = df["real_close"].shift(1)

# 删除任何含 NaN 的行（保证三个列都对齐）
df = df.dropna().reset_index(drop=True)

df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

y_true = df["real_close"].values
y_pred = df["pred_close"].values
y_prev = df["prev_close"].values

# 1. RMSE & MAE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)


# 3. ACC（涨跌方向准确率）
true_dir = np.sign(y_true - y_prev)
pred_dir = np.sign(y_pred - y_prev)
acc = np.mean(true_dir == pred_dir)

# 4. CR（cumulative return）
true_return_pct = (y_true - y_prev) / y_prev
money = 1.0
for i in range(len(y_true)):
    if pred_dir[i] > 0:
        money *= (1 + true_return_pct[i])
cr = (money - 1) * 100  # 累计收益率百分比

# 5. SR（夏普比率）——仍用之前的策略收益
strategy_return = np.where(pred_dir > 0, y_true - y_prev, 0)
sr = np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) != 0 else np.nan

with open(output_path, "w") as f:
    f.write(f"评估样本数: {len(y_true)}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"CR: {cr:.2f}%\n")
    f.write(f"SR:  {sr:.4f}\n")