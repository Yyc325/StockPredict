import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load stock data
data_dir = "/Users/milescai/PycharmProjects/StockPredict/price/raw/"
tweet_dir = "/Users/milescai/PycharmProjects/StockPredict/tweet/raw/"
sequence_length = 5

all_data = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        df['Stock'] = file.split('.')[0]
        all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
stock_data = combined_df[features].values
dates = pd.to_datetime(combined_df['Date']).dt.strftime('%Y-%m-%d').tolist()

# Load tweet embeddings by date
tweet_embeddings_by_date = {}
for fname in os.listdir(tweet_dir):
    if fname.endswith('.npy'):
        date_str = fname[:-4]  # safely remove .npy
        emb = np.load(os.path.join(tweet_dir, fname))
        if not np.isnan(emb).any():  # Skip bad files
            tweet_embeddings_by_date[date_str] = emb

# Helper: get tweet embedding for past 5 days
def get_tweet_sequence_embedding(start_date, window=5):
    embeddings = []
    base = datetime.strptime(start_date, "%Y-%m-%d")
    for offset in range(-window + 1, 1):
        d = (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        emb = tweet_embeddings_by_date.get(d, np.zeros(768))
        if np.isnan(emb).any():
            emb = np.zeros(768)
        embeddings.append(emb)
    return np.concatenate(embeddings)

# Create dataset
movement = (combined_df['Close'][1:].values > combined_df['Close'][:-1].values).astype(int)
X, y = [], []

for i in range(len(stock_data) - sequence_length - 4):  # -4 for d+3
    stock_seq = stock_data[i:i + sequence_length].flatten()
    tweet_seq = get_tweet_sequence_embedding(dates[i + sequence_length - 1])
    combined_input = np.concatenate([stock_seq, tweet_seq])
    X.append(combined_input)
    y.append([
        movement[i + sequence_length],
        movement[i + sequence_length + 1],
        movement[i + sequence_length + 3]
    ])

X = np.array(X)
y = np.array(y)

# Debugging output
print("Any NaNs in X:", np.isnan(X).any())
print("Any NaNs in y:", np.isnan(y).any())
print("Max value in X:", np.max(X))
print("Min value in X:", np.min(X))

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Dataset & Dataloader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = StockDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Neural Network
class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 5 * len(features) + 5 * 768  # stock + tweet
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Predicting 3 outputs: d, d+1, d+3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # raw logits for BCEWithLogitsLoss

model = PricePredictor()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # lowered learning rate

# Train the model
train_losses = []

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Plot training loss
plt.plot(range(1, 21), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()