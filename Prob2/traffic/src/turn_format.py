import pandas as pd

# 输入路径
input_path = "/root/Time-Series-Library/results/long_term_forecast_TimeXer_traffic_PatchTST_traffic_ftMS_sl24_ll24_pl1_dm128_nh8_el3_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/submission.csv"

# 输出路径
output_path = "/root/Time-Series-Library/results/q_estimate_output.csv"

# 读取原始 submission 文件
df = pd.read_csv(input_path)

# 构造新表
result = pd.DataFrame({
    'id': range(1, len(df) + 1),
    'q_estimate': df['predictions']
})

# 保存为新 CSV
result.to_csv(output_path, index=False)