import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 读取 grey.csv 和 spectrum.csv
grey_df = pd.read_csv('grey.csv')
spectrum_df = pd.read_csv('spectrum.csv')

# 对 spectrum_df 进行归一化
scaler = MinMaxScaler()
spectrum_norm = scaler.fit_transform(spectrum_df.values)

# 假设 grey_df 是输入，spectrum_df 是输出
X = torch.tensor(grey_df.values, dtype=torch.float32)
y = torch.tensor(spectrum_norm, dtype=torch.float32)

# 构建数据集和数据加载器
batch_size = 32
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义简单的残差网络
class SimpleResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        out = torch.relu(self.fc1(x))
        residual = out
        out = torch.relu(self.fc2(out))
        out = out + residual  # 避免 in-place
        out = self.fc3(out)
        return out

input_dim = X.shape[1]
output_dim = y.shape[1] if len(y.shape) > 1 else 1
model = SimpleResNet(input_dim, output_dim)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 训练网络
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# 测试
import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    idx = np.random.randint(0, X.shape[0])
    x_sample = X[idx:idx+1]
    y_true = y[idx].cpu().numpy().flatten()
    y_pred = model(x_sample).cpu().numpy().flatten()

plt.figure(figsize=(10,5))
wavelengths = np.arange(400, 801)
plt.plot(wavelengths, y_true, label='True Spectrum')
plt.plot(wavelengths, y_pred, label='Predicted Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Value')
plt.title('Spectrum Prediction Visualization')
plt.legend()
plt.show()
