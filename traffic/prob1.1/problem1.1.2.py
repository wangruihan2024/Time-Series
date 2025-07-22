import pandas as pd
import numpy as np

# 读入数据
train_df = pd.read_csv("traffic/dataset/loop_sensor_train.csv")
test_df = pd.read_csv("traffic/dataset/loop_sensor_test_x.csv")

# 转换时间格式
train_df['t_1h'] = pd.to_datetime(train_df['t_1h'])
test_df['t_1h'] = pd.to_datetime(test_df['t_1h'])

# 排序以便后续查找
train_df = train_df.sort_values(['iu_ac', 't_1h']).reset_index(drop=True)
test_df = test_df.sort_values(['iu_ac', 't_1h']).reset_index(drop=True)

# 为每个设备构建历史时间和q值
device_history = {}
for iu_ac, group in train_df.groupby('iu_ac'):
    device_history[iu_ac] = {
        'times': group['t_1h'].values.astype('datetime64[ns]'),
        'qs': group['q'].values
    }

# 查找设备当前时间点的上一个 q
def find_previous_q(row):
    t_np = np.datetime64(row['t_1h'])
    iu_ac = row['iu_ac']
    
    if iu_ac in device_history:
        history = device_history[iu_ac]
        idx = np.searchsorted(history['times'], t_np, side='right') - 1
        if idx >= 0:
            return history['qs'][idx]
    
    # 若找不到对应设备或前一个时间，则返回 np.nan 或其他默认值
    return 0

# 应用函数
test_df["estimate_q"] = test_df.apply(find_previous_q, axis=1)

# 保存预测结果
output_path = "traffic/prob1.1/loop_sensor_test_baseline_1.1.2.csv"
test_df[["id", "estimate_q"]].to_csv(output_path, index=False)

print("✅ 基于前一个时间点的q预测完成，结果保存在：", output_path)