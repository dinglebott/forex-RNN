import optuna
from custom_modules import dataparser
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from custom_modules.xgboost_trainer import xgbSignals
import copy
from sklearn.metrics import f1_score
import os
import json

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
# other
epochs = 50 # early stopping implemented
deadzone = 0.0015
featureList = ["return", "return_4", "log_return", "log_return_4",
               "atr_14", "volatility_regime",
               "bb_width", "bb_position",
               "hl_spread", "oc_spread", "upper_wick", "lower_wick",
               "normalised_ema15", "normalised_ema50", "ema_cross",
               "rsi_14", "macd_hist", "vol_ratio", "vol_momentum"]

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SUPPRESS OPTUNA LOGS (set to INFO for normal per-trial logs, or DEBUG for more logs)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ENSURE REPRODUCIBILITY
torch.manual_seed(16)
np.random.seed(16)

# LOAD DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 16}-01-01_{yearNow}-01-01.json")
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
features = df[featureList]
labels = df["target"]

# INTEGRATE XGBOOST SIGNALS
features = pd.concat([features, xgbSignals], axis=1) # shape (samples, features)
features.dropna(inplace=True) # drop rows with no xgbSignals (warmup rows)

# prune features
features.drop(columns=["xgb_0", "bb_position", "vol_ratio", "lower_wick", "vol_momentum"], inplace=True)

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA (remove 2025 test data)
splitDate = f"{yearNow - 1}-01-01"

splitIdx = timestamps[timestamps > splitDate].index[0] # get index of first row after split date
splitIdx = timestamps.index.get_loc(splitIdx) # realign indexes

features_train = features.iloc[:splitIdx]
features_test = features.iloc[splitIdx:]

labels_train = labels.iloc[:splitIdx].values # convert to numpy array for correct indexing in createSequences()
labels_test = labels.iloc[splitIdx:].values

# SCALE FEATURES
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test) # dont fit on test data

# DATA SEQUENCES (created within objective)
def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

# cache sequences to avoid rebuilding each trial (also converts to tensors)
sequenceCache = {}
def getSequences(fts, lbls, lookback, key): # key to differentiate train and test with same lookback
    cacheKey = (lookback, key)
    if cacheKey not in sequenceCache:
        X, y = createSequences(fts, lbls, lookback)
        sequenceCache[cacheKey] = (
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.long, device=device)
        )
    return sequenceCache[cacheKey]

# TRAIN TEST SPLIT (created within objective)
trainValSplit = TimeSeriesSplit(n_splits=3)

# BUILD MODEL (instantiated within objective)
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

# HELPER FUNCTIONS: predict in batches to prevent vram overflow
def batchPredict(model, X, batchSize=1024):
    allPreds = []
    for i in range(0, len(X), batchSize):
        batch = X[i : i + batchSize]
        logits = model(batch) # raw output of model => tensor of shape (samples, 3)
        preds = torch.argmax(logits, dim=1).cpu().numpy() # convert to predictions, shift to cpu
        allPreds.append(preds)
    return np.concatenate(allPreds)
def batchLoss(model, X, y, criterion, batchSize=1024):
    totalLoss = 0
    for i in range(0, len(X), batchSize):
        Xb = X[i : i + batchSize]
        yb = y[i : i + batchSize]
        totalLoss += criterion(model(Xb), yb).item() * len(Xb) # criterion returns avg loss, multiply to get total
    return totalLoss / len(X) # divide out to get overall avg loss

# OPTUNA MAGIC
def objective(trial):
    # PARAMS TO TUNE
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [512, 768]),
        "num_layers": 2
    }
    dropout = trial.suggest_float("dropout", 0.15, 0.45) if params["num_layers"] > 1 else 0.0 # dropout only works for >1 layers
    lookback = trial.suggest_int("lookback", 20, 25)
    optimiserName = trial.suggest_categorical("optimiser", ["Adam", "RMSprop"])
    learningRate = trial.suggest_float("lr", 3e-4, 1e-3, log=True)
    weightDecay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batchSize = trial.suggest_categorical("batch_size", [256, 512])
    clipGradNorm = trial.suggest_float("clip_grad_norm", 4.0, 6.0)

    # CREATE SEQUENCES (already converted to tensors by function)
    X_train, y_train = getSequences(features_train, labels_train, lookback, key="train")
    X_test, y_test = getSequences(features_test, labels_test, lookback, key="test")
    # shape of X: (samples, timesteps, features)
    # shape of y: (samples, output_classes)
    testTrue = y_test.cpu().numpy() # transfer to cpu for F1 score later

    # SUBFOLD LOOP
    scores = []
    trainScores = []
    for trainIdxs, valIdxs in trainValSplit.split(X_train):
        # SPLIT TRAIN AND VALIDATION SETS
        purgedTrainIndexes = trainIdxs[:-4] # prevent leaking from candlesAhead
        X_fold_train = X_train[purgedTrainIndexes]
        X_fold_val = X_train[valIdxs]
        y_fold_train = y_train[purgedTrainIndexes]
        y_fold_val = y_train[valIdxs]
        foldTrainTrue = y_fold_train.cpu().numpy() # for F1 score later

        # INSTANTIATE MODEL
        model = ForexRNN(
            **params,
            dropout=dropout,
            input_size=features.shape[1],
            output_size=3
        ).to(device)

        # LOSS FUNCTION AND OPTIMISER
        classCounts = np.bincount(labels_train.astype(int)) # no. of each class
        classWeights = 1.0 / classCounts # majority class => smaller weight and vice versa
        classWeights = (classWeights / classWeights.sum()) * len(classWeights)  # normalise
        weightsTensor = torch.tensor(classWeights, dtype=torch.float32, device=device) # penalise mistakes on minority classes more

        criterion = nn.CrossEntropyLoss(weight=weightsTensor) # function to minimise
        optimiserClass = {"Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop}[optimiserName]
        optimiser = optimiserClass(model.parameters(), lr=learningRate, weight_decay=weightDecay)

        # TRAIN MODEL
        dataset = torch.utils.data.TensorDataset(X_fold_train, y_fold_train) # Dataset object is a wrapper to keep tensors aligned
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
        # DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

        # for early stopping
        bestValLoss = float("inf")
        badEpochs = 0
        bestModelState = None

        # training loop
        for _ in range(epochs):
            # train
            model.train() # set model to training mode
            for X_batch, y_batch in dataloader:
                optimiser.zero_grad() # clear gradients from previous batch
                predictions = model(X_batch) # run the __call__ method which calls forward(X_batch)
                loss = criterion(predictions, y_batch) # evaluate loss, returns loss tensor (1 element: avg loss across the batch)
                loss.backward() # backpropagation, compute loss with respect to each parameter
                nn.utils.clip_grad_norm_(model.parameters(), clipGradNorm) # prevent exploding gradients
                optimiser.step() # adjust gradients to minimise loss
            
            # validate (check for overfitting while training)
            model.eval() # disable dropout
            with torch.no_grad(): # disable gradient tracking to save memory
                valLoss = batchLoss(model, X_fold_val, y_fold_val, criterion)

            # check for early stopping
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                badEpochs = 0
                bestModelState = copy.deepcopy(model.state_dict()) # shallow copy retains references to original tensors
            else:
                badEpochs += 1
                if badEpochs >= 15:
                    break
        # restore best model
        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        # TEST MODEL
        model.eval() # disable dropout
        with torch.no_grad(): # disable gradient tracking to save memory
            testPreds = batchPredict(model, X_test)
            foldTrainPreds = batchPredict(model, X_fold_train) # check for overfitting

        # EVALUATE MODEL
        f1Score = f1_score(testTrue, testPreds, average="macro", zero_division=0)
        scores.append(f1Score)
        trainF1Score = f1_score(foldTrainTrue, foldTrainPreds, average="macro", zero_division=0)
        trainScores.append(trainF1Score)
    
    # print train and test F1 for overfitting check
    print(f"Trial {trial.number} | Train F1: {np.mean(trainScores):.5f} | Test F1: {np.mean(scores):.5f}")
    # return score to study object
    return np.mean(scores)

# MAIN OPTUNA MAGIC
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# PRINT AND SAVE RESULTS
print(study.best_params) # a python dict
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "hyperparameters.json"
filepath = os.path.join(directory, filename)
with open(filepath, "w") as f:
    json.dump(study.best_params, f, indent=4)