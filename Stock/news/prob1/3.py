import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ========== 参数 ==========
LOOKBACK = 5
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

df = pd.read_csv("Stock/news/dataset/DJIA_merged_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

output_path = "Stock/news/prob1/3.out"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

drop_cols = ["Date", "Label", "Close"] + [f"Top{i}" for i in range(1, 26)]
feature_cols = [col for col in df.columns if col not in drop_cols]
features = df[feature_cols].values
target = df["Close"].values

X, y = [], []
for i in range(LOOKBACK, len(df)):
    X.append(features[i - LOOKBACK:i].flatten())
    y.append(target[i])
X = np.array(X)
y = np.array(y).reshape(-1, 1)

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)

# ========== Dataset 类 ==========
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += loss_fn(pred, y_batch).item() * len(X_batch)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(X_train):.4f} | Val Loss: {val_loss/len(X_val):.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_scaled = model(X_test_tensor).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test).flatten()


y_prev = y_true[:-1]
y_true = y_true[1:]
y_pred = y_pred[1:]

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
cr = (money - 1) * 100 

strategy_return = np.where(pred_dir > 0, y_true - y_prev, 0)
sr = np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) != 0 else np.nan


with open(output_path, "w") as f:
    f.write(f"评估样本数: {len(y_true)}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"CR: {cr:.2f}%\n")
    f.write(f"SR:  {sr:.4f}\n")