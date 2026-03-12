import torch
import torch.nn as nn
import joblib
from custom_modules import datafetcher, dataparser
import numpy as np
import os
import json

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
version = 3

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BUILD MODEL
class ForexHybrid(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, lstm_dropout, output_size,
                 num_filters, kernel_size):
        super(ForexHybrid, self).__init__()
        # CNN layers: takes 3D tensor as input (batch_size, channels, length)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                        kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters), # normalise before passing to LSTM
            nn.Dropout(dropout)
        )
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_filters, # takes CNN output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        # Output layer (maps final pattern produced by LSTM to actual prediction) (fully connected)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x is 3D tensor (batch_size, timesteps, features)
        # CNN
        x = x.permute(0, 2, 1) # (batch, features, timesteps)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # (batch, timesteps, num_filters)
        # LSTM
        lstmOutput, (hidden, cell) = self.lstm(x)
        # FC
        lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
        return self.fc(lastTimestep) # map to prediction (batch_size, output size)

# DEFINE FEATURES (copy-paste from the model training features exactly)
directory = "models"
filename = f"features_v{version}.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    featureList = json.load(file)["features"]

# DEFINE PARAMETERS (copy-paste from the model training hyperparameters exactly)
filename = f"hyperparameters_v{version}.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    hyperparams = json.load(file)
    params = hyperparams["modelParams"]
    lookback = hyperparams["lookback"]

# LOAD MODEL AND SCALER
model = ForexHybrid(
    **params,
    input_size=len(featureList),
    output_size=3,
    lstm_dropout=params["dropout"]
).to(device)

directory = "models"
filename = f"NN_{instrument}_{granularity}_{yearNow}_v{version}.pth"
filepath = os.path.join(directory, filename)
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

filename = f"scaler_v{version}.pkl"
filepath = os.path.join(directory, filename)
scaler = joblib.load(filepath)

# FETCH AND PARSE CURRENT DATA
jsonPath = datafetcher.getData(instrument, granularity, 200, "live")
df = dataparser.parseData(jsonPath)

# CONSTRUCT SEQUENCES
features = scaler.transform(df[featureList])
X = []
for i in range(len(features) - lookback + 1):
    X.append(features[i : i + lookback])
X = torch.tensor(np.array(X), dtype=torch.float32, device=device)

# RUN INFERENCES
def predictByThreshold(probs, threshold=0.35):
    preds = []
    for p in probs:
        if p[1] > max(threshold, p[0], p[2]):
            preds.append(1)
        else:
            preds.append(0 if p[0] > p[2] else 2)
    return preds

with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = predictByThreshold(probs)

# DISPLAY RESULTS
def getLabel(num):
    match num:
        case 0:
            return "DOWN"
        case 1:
            return "FLAT"
        case 2:
            return "UP"

finalPred = getLabel(preds[-1])

print("")
finalProbs = probs[-1]  # most recent candle
for idx, prob in enumerate(finalProbs):
    print(f"{getLabel(idx)}: {prob*100:.2f}%")

print(f"\nFinal prediction: {finalPred}")

input("\nPress Enter to exit")