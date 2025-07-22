import pandas as pd
import numpy as np

train_path = "traffic/dataset/loop_sensor_train.csv"
test_path = "traffic/dataset/loop_sensor_test_x.csv"
output_path = "traffic/prob1.2/loop_sensor_test_prev_6past.csv"

# 读数据
train_df = pd.read_csv(train_path, parse_dates=["t_1h"])
test_df = pd.read_csv(test_path, parse_dates=["t_1h"])
print("✅ 数据加载完成")

# 提取时间特征
train_df["hour"] = train_df["t_1h"].dt.hour
train_df["minute"] = train_df["t_1h"].dt.minute

test_df["hour"] = test_df["t_1h"].dt.hour
test_df["minute"] = test_df["t_1h"].dt.minute

# 分组为字典：每个设备一个DataFrame
grouped_dict = {k: v for k, v in train_df.groupby("iu_ac")}

# 预测函数
def get_estimate_q(row):
    iu = row["iu_ac"]
    t_target = row["t_1h"]
    hour = row["hour"]
    minute = row["minute"]

    if iu not in grouped_dict:
        return 0.0

    group = grouped_dict[iu]

    # 只保留小时+分钟相同，且时间早于当前测试点的记录
    history = group[(group["hour"] == hour) &
                    (group["minute"] == minute) &
                    (group["t_1h"] < t_target)].copy()

    if history.empty:
        return 0.0

    # 根据时间差排序（越接近当前越好）
    history["time_diff"] = (t_target - history["t_1h"])
    nearest_6 = history.nsmallest(6, "time_diff")

    return nearest_6["q"].mean()

print("✅ 查找逻辑构造完成，开始预测...")
test_df["estimate_q"] = test_df.apply(get_estimate_q, axis=1)

# 保存预测结果
test_df[["id", "estimate_q"]].to_csv(output_path, index=False)
print(f"✅ 仅使用过去的最近6个时间点策略完成，结果保存至: {output_path}")

# 打印前10条调试信息
print("\n前10条测试样本使用的历史记录：")
for idx, row in test_df.head(10).iterrows():
    print(f"\nTest ID: {row['id']}, 当前时间: {row['t_1h']}")
    print("使用的历史记录：")

    iu = row["iu_ac"]
    t_target = row["t_1h"]
    hour = row["hour"]
    minute = row["minute"]

    if iu not in grouped_dict:
        print("  ⚠️ 无任何历史记录")
        continue

    group = grouped_dict[iu]
    history = group[(group["hour"] == hour) &
                    (group["minute"] == minute) &
                    (group["t_1h"] < t_target)].copy()

    if history.empty:
        print("  ⚠️ 无匹配记录")
        continue

    history["time_diff"] = (t_target - history["t_1h"])
    nearest_6 = history.nsmallest(6, "time_diff")

    for t_used, q in zip(nearest_6["t_1h"], nearest_6["q"]):
        print(f"  使用历史时间: {t_used}, q: {q}")