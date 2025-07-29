import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# ========= 配置 =========
LOOKBACK = 24
BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("使用 CUDA GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("使用 Apple MPS 加速器")
else:
    DEVICE = torch.device("cpu")
    print("使用 CPU 设备")

train_path = "traffic/dataset/loop_sensor_train.csv"
test_path = "traffic/dataset/loop_sensor_test_x.csv"
output_path = "traffic/prob1.3/loop_sensor_test_MLP.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ========= 数据读取 =========
df = pd.read_csv(train_path, parse_dates=["t_1h"])
df.sort_values(["iu_ac", "t_1h"], inplace=True)

# ========= 构造训练样本 =========
X = []
y = []
for _, group in df.groupby("iu_ac"):
    group = group.sort_values("t_1h").reset_index(drop=True)
    feat = group[["etat_barre", "q"]].values
    for i in range(LOOKBACK, len(group)):
        X.append(feat[i - LOOKBACK:i].flatten())
        y.append(feat[i][1])  # 当前时刻的 q

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# ========= 归一化 =========
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# ========= 自定义 Dataset =========
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TrafficDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

# ========= 定义 MLP 模型 =========
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(X.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========= 训练 =========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_X)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader.dataset):.6f}")

# ========= 读取测试集 =========
test_df = pd.read_csv(test_path, parse_dates=["t_1h"])
test_df = test_df.sort_values(["iu_ac", "t_1h"]).reset_index(drop=True)

# ========= 建立训练数据索引表 =========
df.set_index(["iu_ac", "t_1h"], inplace=True)
df = df.sort_index()

# ========= 预测 =========
model.eval()
predictions = []

for row in test_df.itertuples():
    iu = row.iu_ac
    t = row.t_1h
    id_ = row.id
    
    try:
        # 取对应设备过去 24 小时的数据（1小时粒度）
        past_times = [t - pd.Timedelta(hours=i) for i in range(LOOKBACK, 0, -1)]
        past_data = []
        for past_t in past_times:
            try:
                etat = df.loc[(iu, past_t), "etat_barre"]
                q_val = df.loc[(iu, past_t), "q"]
                past_data.append([etat, q_val])
            except KeyError:
                past_data.append([0, 0])  # 缺失数据补零
        
        past_data = np.array(past_data).flatten().reshape(1, -1)
        past_data = x_scaler.transform(past_data)
        past_tensor = torch.tensor(past_data, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = model(past_tensor).cpu().numpy()
            pred_q = y_scaler.inverse_transform(pred)[0][0]

    except Exception as e:
        pred_q = 0.0  # 如果出错（例如无历史记录），就预测为 0

    predictions.append((id_, pred_q))

# ========= 保存为 CSV =========
output_df = pd.DataFrame(predictions, columns=["id", "q"])
output_df.to_csv(output_path, index=False)
print(f"✅ 预测完成，结果已保存至：{output_path}")