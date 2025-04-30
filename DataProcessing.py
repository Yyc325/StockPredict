import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

# Load stock data
data_dir = "/Users/milescai/PycharmProjects/StockPredict/price/raw/"
tweet_dir = "/Users/milescai/PycharmProjects/StockPredict/tweet/raw/"
sequence_length = 5

Data = {

}

all_data = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        stock_name = file.split('.')[0]
        stock_data = {}
#Date,Open,High,Low,Close,Adj Close,Volume
        for index, row in df.iterrows():
            stock_data[row["Date"]] = {
                "open": row["Open"],
                "close": row["Close"],
                "high": row["High"],
                "low": row["Low"],
                "adj_close": row["Adj Close"],
                "volume": row["Volume"],
                "tweet": {},
                "tweet_embedding": {},
            }
        Data[stock_name] = stock_data


with open('data.pickle', 'wb') as handle:
    pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
