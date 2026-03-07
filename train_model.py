from custom_modules import dataparser
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from custom_modules.xgboost_trainer import xgbSignals

# GLOBAL VARIABLES
hiddenSize = 64
numLayers = 2
dropOut = 0.2
learningRate = 0.001
lookback = 20
deadzone = 0.0015
featureList = []

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
df = dataparser.parseData("json_data/EUR_USD_H4_2010-01-01_2026-01-01.json")
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

# TARGET VARIABLE: net return over next 4 candles
df["forward_return"] = (df["close"].shift(-4) / df["close"]) - 1
conditions = [
    df["forward_return"] < -deadzone, # downward move
    df["forward_return"] > deadzone # upward move
]
choices = [0, 2]
df["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
df.dropna(inplace=True)

# SEPARATE FEATURES AND LABELS (input and output)
labels = df["target"]
features = df[featureList]

# INTEGRATE XGBOOST SIGNALS
assert features.index == xgbSignals.index # check for index misalignment
features = pd.concat([features, xgbSignals], axis=1)
featureList.append["xgb_0", "xgb_1", "xgb_2"]
features.dropna(inplace=True) # drop rows with no xgbSignals (warmup rows)

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA
splitDate = "2025-01-01"
splitIdx = timestamps[timestamps > splitDate].index[0] # get index of first row after split date
splitIdx = timestamps.index.get_loc(splitIdx) # realign indexes

features_train = features.iloc[:splitIdx]
features_test = features.iloc[splitIdx:]
labels_train = labels.iloc[:splitIdx].values # convert to numpy array for correct indexing in createSequences()
labels_test = labels.iloc[splitIdx:].values

# SCALE FEATURES
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test) # avoid fitting on test to prevent data leakage

# CREATE DATA SEQUENCES
def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

X_train, y_train = createSequences(features_train, labels_train, lookback)
X_test, y_test = createSequences(features_test, labels_test, lookback)

# CONVERT TO TENSORS
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)
# shape of X: (samples, timesteps, features)
# shape of y: (samples, output_classes)

# BUILD MODEL
class ForexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(ForexRNN, self).__init__()
        # LSTM layer: takes 3D tensor as input (batch_size, timesteps, features)
        self.lstm = nn.LSTM(
            input_size=input_size, # no. of features per datapoint
            hidden_size=hidden_size, # no. of neurons in hidden state
            num_layers=num_layers, # no. of layers in the LSTM
            batch_first=True, # set batch size as first dimension of input tensor
            dropout=dropout # equivalent of subsample for RNN
        )
        # Output layer (maps final pattern produced by LSTM to actual prediction) (fully connected)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x is 3D tensor (batch_size, timesteps, features)
        lstmOutput, (hidden, cell) = self.lstm(x)
        # lstmOutput: hidden state of EVERY timestep for the LAST layer only (batch_size, timesteps, hidden_size)
        # hidden: final hidden state (LAST timestep) for EVERY layer (layers, batch_size, hidden_size)
        # cell: similar to hidden but contains cell state instead of hidden state
        lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
        finalOutput = self.fc(lastTimestep) # map to prediction (batch_size, output size)
        return finalOutput

# INSTANTIATE MODEL
model = ForexRNN(
    input_size=features_train.shape[1],
    hidden_size=hiddenSize,
    num_layers=numLayers,
    dropout=dropOut,
    output_size=3
).to(device)

# LOSS FUNCTION AND OPTIMISER
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)