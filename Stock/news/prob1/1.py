import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 读取数据
df = pd.read_csv("Stock/news/dataset/DJIA_merged_data.csv", parse_dates=["Date"])  
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

output_path = "Stock/news/prob1/1.out"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df["y_true"] = df["Close"]
df["y_prev"] = df["Close"].shift(1)  
df["y_pred"] = df["Close"].shift(1) 

# 去除前1个NA
df = df.dropna().reset_index(drop=True)

# 选取后20%的数据用于评估
df = df.iloc[int(len(df)*0.8):].reset_index(drop=True)

# 提取 numpy 数组
y_true = df["y_true"].values
y_prev = df["y_prev"].values
y_pred = df["y_pred"].values

# 1. RMSE & MAE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

# 2. ACC（涨跌方向）
true_dir = np.sign(y_true - y_prev)
prev_dir = np.sign(y_pred - y_prev)  
acc = np.mean(true_dir == 1)# 一直是0所以默认一直在涨

# 4. CR（cumulative return）
true_return_pct = (y_true - y_prev) / y_prev
money = 1.0
for i in range(len(y_true)):
    if prev_dir[i] >= 0:
        money *= (1 + true_return_pct[i])
cr = (money - 1) * 100  # 累计收益率百分比

# 5. SR（夏普比率）——仍用之前的策略收益
strategy_return = np.where(prev_dir >= 0, y_true - y_prev, 0)
sr = np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) != 0 else np.nan

with open(output_path, "w") as f:
    f.write(f"评估样本数: {len(y_true)}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"CR: {cr:.2f}%\n")
    f.write(f"SR:  {sr:.4f}\n")