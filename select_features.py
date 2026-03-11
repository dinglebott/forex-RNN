from custom_modules import dataparser
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from custom_modules.xgboost_trainer import xgbSignals
import copy
from sklearn.metrics import f1_score
import os

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
arch = 1 # 0 for LSTM, 1 for CNN/LSTM
# hyperparameters
hiddenSize = 128 # no. of neurons in hidden state
numLayers = 2 # no. of layers in the LSTM
dropOut = 0.25 # equivalent of subsample for RNN
lookback = 20
optimiserName = "Adam"
learningRate = 1.5e-4
weightDecay = 5e-4
batchSize = 1024
clipGradNorm = 5.1
# CNN params
numFilters = 64
kernelSize = 5
# other
epochs = 80 # early stopping implemented
deadzone = 0.0015
featureList = ["return", "return_4", "log_return", "log_return_4",
               "atr_14", "volatility_regime",
               "bb_width", "bb_position",
               "hl_spread", "oc_spread", "upper_wick", "lower_wick",
               "normalised_ema15", "normalised_ema50", "ema_cross",
               "rsi_14", "macd_hist", "vol_ratio", "vol_momentum",
               "open_return", "high_return", "low_return", "close_return"]

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENSURE REPRODUCIBILITY
torch.manual_seed(16) # ensure reproducible results
np.random.seed(16)

# LOAD DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-01-01.json")
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
features = pd.concat([features, xgbSignals], axis=1) # shape (samples, features)
featureList.extend(["xgb_0", "xgb_1", "xgb_2"])
features.dropna(inplace=True) # drop rows with no xgbSignals (warmup rows)

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA
splitDate = f"{yearNow - 1}-01-01"
valSplitDate = f"{yearNow - 2}-01-01"

splitIdx = timestamps[timestamps > splitDate].index[0] # get index of first row after split date
splitIdx = timestamps.index.get_loc(splitIdx) # realign indexes
valSplitIdx = timestamps[timestamps > valSplitDate].index[0]
valSplitIdx = timestamps.index.get_loc(valSplitIdx)

features_train = features.iloc[:valSplitIdx]
features_val = features.iloc[valSplitIdx:splitIdx]
features_test = features.iloc[splitIdx:]

labels_train = labels.iloc[:valSplitIdx].values # convert to numpy array for correct indexing in createSequences()
labels_val = labels.iloc[valSplitIdx:splitIdx].values
labels_test = labels.iloc[splitIdx:].values

# SCALE FEATURES
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_val = scaler.transform(features_val) # avoid fitting on val/test to prevent data leakage
features_test = scaler.transform(features_test)

# CREATE DATA SEQUENCES
def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

X_train, y_train = createSequences(features_train, labels_train, lookback)
X_val, y_val = createSequences(features_val, labels_val, lookback)
X_test, y_test = createSequences(features_test, labels_test, lookback)

# CONVERT TO TENSORS
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
y_val = torch.tensor(y_val, dtype=torch.long, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)
# shape of X: (samples, timesteps, features)
# shape of y: (samples, output_classes)
valTrue = y_val.cpu().numpy() # for f1 score later
testTrue = y_test.cpu().numpy()

# BUILD MODELS
class ForexRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(ForexRNN, self).__init__()
        # LSTM layer: takes 3D tensor as input (batch_size, timesteps, features)
        self.lstm = nn.LSTM(
            input_size=input_size, # no. of features per datapoint
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # set batch size as first dimension of input tensor
            dropout=dropout if num_layers > 1 else 0
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
        return self.fc(lastTimestep) # map to prediction (batch_size, output size)

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
        # lstmOutput: hidden state of EVERY timestep for the LAST layer only (batch_size, timesteps, hidden_size)
        # hidden: final hidden state (LAST timestep) for EVERY layer (layers, batch_size, hidden_size)
        # cell: similar to hidden but contains cell state instead of hidden state
        # FC
        lastTimestep = lstmOutput[:, -1, :] # slice out last timestep across all samples and neurons
        return self.fc(lastTimestep) # map to prediction (batch_size, output size)

# INSTANTIATE MODEL
match arch:
    case 0:
        model = ForexRNN(
            input_size=features_train.shape[1],
            hidden_size=hiddenSize,
            num_layers=numLayers,
            dropout=dropOut,
            output_size=3
        ).to(device)
    case 1:
        model = ForexHybrid(
            input_size=features_train.shape[1],
            hidden_size=hiddenSize,
            num_layers=numLayers,
            dropout=dropOut,
            lstm_dropout=dropOut,
            output_size=3,
            num_filters=numFilters,
            kernel_size=kernelSize
        ).to(device)

# LOSS FUNCTION AND OPTIMISER
classCounts = np.bincount(labels_train.astype(int)) # no. of each class
classWeights = 1.0 / classCounts # majority class => smaller weight and vice versa
classWeights = (classWeights / classWeights.sum()) * len(classWeights)  # normalise
weightsTensor = torch.tensor(classWeights, dtype=torch.float32, device=device) # penalise mistakes on minority classes more

criterion = nn.CrossEntropyLoss(weight=weightsTensor) # function to minimise
optimiserClass = {"Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop}[optimiserName]
optimiser = optimiserClass(model.parameters(), lr=learningRate, weight_decay=weightDecay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, "max", factor=0.5, patience=10)

# TRAIN MODEL
dataset = torch.utils.data.TensorDataset(X_train, y_train) # Dataset object is a wrapper to keep tensors aligned
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
# DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

# for early stopping and saving best model
bestValF1 = 0
badEpochs = 0
bestModelState = None

# training loop
for epoch in range(epochs):
    # train
    model.train() # set model to training mode
    epochLoss = 0 # initialise cumulative loss for this epoch
    for X_batch, y_batch in dataloader:
        optimiser.zero_grad() # clear gradients from previous batch
        predictions = model(X_batch) # run the __call__ method which calls forward(X_batch)
        loss = criterion(predictions, y_batch) # evaluate loss, returns loss tensor (1 element: the average loss across the batch)
        loss.backward() # backpropagation, compute loss with respect to each parameter
        nn.utils.clip_grad_norm_(model.parameters(), clipGradNorm) # prevent exploding gradients
        optimiser.step() # adjust gradients to minimise loss
        epochLoss += loss.item() # convert tensor to normal number and add to cumulative loss
    avgLoss = epochLoss / len(dataloader)
    
    # validate (check for overfitting while training)
    model.eval() # disable dropout
    with torch.no_grad(): # disable gradient tracking to save memory
        valLogits = model(X_val) # raw output of model => tensor of shape (samples, 3)
        valLoss = criterion(valLogits, y_val).item()
        valPreds = torch.argmax(valLogits, dim=1).cpu().numpy() # convert to predictions, shift to cpu
    valF1Score = f1_score(valTrue, valPreds, average="macro", zero_division=0)
    print(f"EPOCH {epoch + 1} | Train loss: {avgLoss:.5f} | Val loss: {valLoss:.5f} | Val F1: {valF1Score:.5f}")

    # check for early stopping
    if valF1Score >= bestValF1:
        bestValF1 = valF1Score
        badEpochs = 0
        bestModelState = copy.deepcopy(model.state_dict()) # shallow copy retains references to original tensors
    else:
        badEpochs += 1
        if badEpochs >= 20:
            print("EARLY STOPPING NOW")
            break
    
    # tune learning rate down if plateauing
    scheduler.step(valF1Score)
# restore best model
if bestModelState is not None:
    model.load_state_dict(bestModelState)

# TEST MODEL
model.eval() # disable dropout
with torch.no_grad(): # disable gradient tracking to save memory
    testLogits = model(X_test) # raw output of model => tensor of shape (samples, 3)
    testPreds = torch.argmax(testLogits, dim=1).cpu().numpy() # convert to predictions, shift to cpu
    
# EVALUATE MODEL
f1Score = f1_score(testTrue, testPreds, average="macro", zero_division=0)

# GET PERMUTATION IMPORTANCES
model.eval()
importances = []
numFeatures = X_test.shape[2]

print("Testing features...")
for featureIdx in range(numFeatures):
    scores = []
    for _ in range(50):
        X_perm = X_test.clone()
        perm = torch.randperm(X_test.shape[0]) # get random permutation of integers from 0 to no. of samples
        X_perm[:, :, featureIdx] = X_perm[:, :, featureIdx][perm] # shuffle all values of this particular feature
        # predict with messed up feature column
        with torch.no_grad():
            logits = model(X_perm)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        # record score for this iteration
        score = f1_score(testTrue, preds, average="macro", zero_division=0)
        scores.append(score)
    # average score for this feature
    avgScore = np.mean(scores)
    # calculate how much the model was hurt by shuffling this feature
    importances.append(f1Score - avgScore)

# display results
importances = pd.DataFrame({
    "Feature": featureList,
    "Importance": importances
}).sort_values("Importance", ascending=False)
print(importances.to_string(index=False))

# save to JSON
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "features.json"
filepath = os.path.join(directory, filename)
importances.set_index("Feature")["Importance"].to_json(filepath, indent=4)