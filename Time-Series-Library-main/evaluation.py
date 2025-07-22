import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred_path = './output/TimesNet.csv'
true_path = './dataset/DJIA_clean.csv'
output_path = './output/TimesNet.txt'

pred_df = pd.read_csv(pred_path)
true_df = pd.read_csv(true_path)

y_pred = pred_df['estimate_OT'].values
start_idx = int(len(true_df) * 0.8)
ot_series = true_df['Adj_Close'].reset_index(drop=True)

y_prev = ot_series[start_idx:].values            
y_true = ot_series[start_idx + 1:].values     

min_len = min(len(y_true), len(y_prev), len(y_pred))
y_true = y_true[:min_len]
y_prev = y_prev[:min_len]
y_pred = y_pred[:min_len]

# print(f'y_pred: {len(y_pred)}, y_true: {len(y_true)}, y_prev: {len(y_prev)}')
# print(f'last pred:{y_pred[-1]}, last true:{y_true[-1]}, last prev:{y_prev[-1]}')
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

with open(output_path, "w") as f:
    f.write(f"评估样本数: {len(y_true)}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"CR: {cr:.2f}%\n")
    f.write(f"SR:  {sr:.4f}\n")