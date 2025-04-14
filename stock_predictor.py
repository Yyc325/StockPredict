import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
# Load all stock data
#test
data_dir = "/Users/milescai/PycharmProjects/stocknet-dataset/price/raw/"
files = os.listdir(data_dir)

all_data = []
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        df['Stock'] = file.split('.')[0]  # Add a column for stock symbol
        all_data.append(df)

# Combine all data into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)
print(combined_df.columns)

# Use the 'Close' price and calculate price movement
prices = combined_df['Close'].values
movement = (prices[1:] > prices[:-1]).astype(int)  # 1 if price increased, 0 if decreased

# Use past 5 days as input (X) and movement as output (y)
sequence_length = 5
X = []
y = []
for i in range(len(prices) - sequence_length - 1):
    X.append(prices[i:i+sequence_length])
    y.append(movement[i+sequence_length])

X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in X])
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Dataset & Dataloader
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

# Model Definition for Binary Classification
class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid for binary output
        )
    def forward(self, x):
        return self.model(x)

# Model, loss function, optimizer
model = PricePredictor()
loss_fn = nn.BCELoss()  # Binary Cross-Entropy loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with loss tracking
losses = []

for epoch in range(20):
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        preds = model(X_batch).squeeze()  # Output is already between 0 and 1 due to Sigmoid in the final layer

        # Debugging: Print first 10 predictions to check if they are between 0 and 1
        print(preds[:10])

        # Check for NaN or Inf values in preds
        if torch.isnan(preds).sum() > 0 or torch.isinf(preds).sum() > 0:
            print("Error: NaN or Inf values found in predictions")
            continue

        print(f"y_batch shape: {y_batch.shape}")
        loss = loss_fn(preds, y_batch.squeeze())  # preds are already in the correct range
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")

# Plot loss over epochs
plt.plot(range(1, 21), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.show()

# After predictions
with torch.no_grad():
    preds = model(X).squeeze()

# Convert predictions to binary (0 or 1)
predicted_classes = (preds > 0.5).int().numpy()


