import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ========= Load and Prepare Data ==========
data_dir = "/Users/milescai/PycharmProjects/StockPredict/price/raw/"
files = os.listdir(data_dir)

all_data = []
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        df['Stock'] = file.split('.')[0]
        all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# Use features: Open, High, Low, Close, Volume
features = ['Open', 'High', 'Low', 'Close', 'Volume']
future_days = [0, 3, 5]  # Predict day d, d+3, d+5

sequence_length = 5
X = []
y = []

for i in range(len(combined_df) - sequence_length - max(future_days)):
    sequence_data = combined_df[features].iloc[i:i+sequence_length].values
    X.append(sequence_data)

    # Predict movement for future_days
    future_movements = []
    for day in future_days:
        close_now = combined_df['Close'].iloc[i + sequence_length + day - 1]
        close_future = combined_df['Close'].iloc[i + sequence_length + day]
        movement = (close_future > close_now).astype(int)
        future_movements.append(movement)

    y.append(future_movements)

X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in X])
y = torch.tensor(y, dtype=torch.float32)

# ========= Dataset & Dataloader ==========
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = PriceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ========= Model Definition ==========
class PricePredictor(nn.Module):
    def __init__(self, future_days_count):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5 * 5, 64),  # 5 days * 5 features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, future_days_count),  # Output for each prediction task
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)

model = PricePredictor(future_days_count=len(future_days))

# ========= Training Setup ==========
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

# ========= Training Loop ==========
for epoch in range(20):
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        preds = model(X_batch)

        if torch.isnan(preds).sum() > 0 or torch.isinf(preds).sum() > 0:
            print("Error: NaN or Inf values found in predictions")
            continue

        loss = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")

# ========= Plot Training Loss ==========
plt.plot(range(1, 21), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.show()

# ========= Final Predictions ==========
with torch.no_grad():
    preds = model(X).numpy()

predicted_classes = (preds > 0.5).astype(int)

# Print example: movements for day d, d+3, and d+5
print("Predictions (first 5 samples):")
for i in range(5):
    print(f"Sample {i+1}: D={predicted_classes[i][0]}, D+3={predicted_classes[i][1]}, D+5={predicted_classes[i][2]}")
