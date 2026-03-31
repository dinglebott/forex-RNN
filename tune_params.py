import optuna
from custom_modules import dataparser, lstm
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import copy
from sklearn.metrics import log_loss
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, arch, _ = globalVars.values()
# other
epochs = 80 # early stopping implemented
earlyStoppingPatience = 10
featureList = [
    "open_return", "high_return", "low_return", "close_return", "vol_return", "smooth_return",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "oc_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "dist_ema100", "ema_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum", "adx_direction"
]

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SUPPRESS OPTUNA LOGS (set to INFO for normal per-trial logs, or DEBUG for more logs)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ENSURE REPRODUCIBILITY
torch.manual_seed(42)
np.random.seed(42)

# LOAD DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-01-01.json")
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

# GET FEATURES AND LABELS (input and output)
filepath = os.path.join("results", "features.json")
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a python dict
# extract positive features into list
featureList = [key for key in rawFeatures if rawFeatures[key] >= 0] # -1 for all features, 0 for positive only
'''featureList = [
    "high_return", "low_return", "vol_return", "smooth_return",
    "atr_14", "volatility_regime",
    "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "ema_cross",
    "rsi_14", "macd_hist",
    "vol_ratio", "vol_momentum",
    "adx_direction"
]''' # for manual feature setting (comment out when not needed)
print(f"Best {len(featureList)} features:", featureList)
features = df[featureList]
labels = df["target"]

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA
splitDate = f"{yearNow - 1}-01-01"

splitIdx = timestamps[timestamps > splitDate].index[0] # get index of first row after split date
splitIdx = timestamps.index.get_loc(splitIdx) # realign indexes

features_train = features.iloc[:splitIdx]
labels_train = labels.iloc[:splitIdx].values # convert to numpy array for correct indexing in createSequences()

# SCALE FEATURES
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)

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
trainValSplit = TimeSeriesSplit(n_splits=2)

# BUILD MODEL (instantiated within objective)
ForexRNN, ForexHybrid = lstm.classBuilder()

# HELPER FUNCTIONS: predict in batches to prevent vram overflow
def batchPredict(model, X, batchSize=1024):
    allPreds = []
    for i in range(0, len(X), batchSize):
        batch = X[i : i + batchSize]
        logits = model(batch) # raw output of model => tensor of shape (samples, 3)
        preds = torch.argmax(logits, dim=1).cpu().numpy() # convert to predictions
        allPreds.append(preds)
    return np.concatenate(allPreds)
def batchProbs(model, X, batchSize=1024):
    allProbs = []
    for i in range(0, len(X), batchSize):
        batch = X[i : i + batchSize]
        logits = model(batch) # raw output of model => tensor of shape (samples, 3)
        probs = torch.softmax(logits, dim=1).cpu().numpy() # convert to predictions
        allProbs.append(probs)
    return np.concatenate(allProbs)
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
        "hidden_size": trial.suggest_categorical("hidden_size", [350, 400, 450, 500, 550]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2])
    }
    dropout = trial.suggest_float("dropout", 0.1, 0.4) # for CNN
    lookback = trial.suggest_categorical("lookback", [20])
    optimiserName = trial.suggest_categorical("optimiser", ["RMSprop"])
    learningRate = trial.suggest_float("lr", 1e-5, 1e-4)
    weightDecay = trial.suggest_float("weight_decay", 1e-5, 1e-4)
    batchSize = trial.suggest_categorical("batch_size", [384, 512, 768])
    clipGradNorm = trial.suggest_float("clip_grad_norm", 5.0, 7.0)
    if arch == 1:
        numFilters = trial.suggest_categorical("num_filters", [16, 24, 32])
        kernelSize = trial.suggest_categorical("kernel_size", [3, 5])
        lstmDropout = dropout if params["num_layers"] > 1 else 0.0 # dropout only works for >1 layers

    # CREATE SEQUENCES (already converted to tensors by function)
    X_train, y_train = getSequences(features_train, labels_train, lookback, key="train")
    # shape of X: (samples, timesteps, features)
    # shape of y: (samples, output_classes)

    # SUBFOLD LOOP
    testScores = []
    for trainIdxs, valIdxs in trainValSplit.split(X_train):
        # SPLIT TRAIN AND VALIDATION SETS
        X_fold_train = X_train[trainIdxs]
        X_fold_val = X_train[valIdxs]
        y_fold_train = y_train[trainIdxs]
        y_fold_val = y_train[valIdxs]

        # INSTANTIATE MODEL
        match arch:
            case 0:
                model = ForexRNN(
                    **params,
                    dropout=dropout,
                    input_size=features.shape[1],
                    output_size=3
                ).to(device)
            case 1:
                model = ForexHybrid(
                    **params,
                    dropout=dropout,
                    lstm_dropout=lstmDropout,
                    input_size=features.shape[1],
                    output_size=3,
                    num_filters=numFilters,
                    kernel_size=kernelSize
                ).to(device)

        # LOSS FUNCTION AND OPTIMISER
        criterion, optimiser, scheduler, _ = lstm.optimiserBundle(
            model, labels_train, device,
            optimiserName, learningRate, weightDecay
        )

        # TRAIN MODEL
        dataset = torch.utils.data.TensorDataset(X_fold_train, y_fold_train) # Dataset object is a wrapper to keep tensors aligned
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
        # DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

        # for early stopping
        bestValLoss = 100
        badEpochs = 0
        bestModelState = None
        valLosses = []

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
                valProbs = batchProbs(model, X_fold_val)
                valLoss = log_loss(y_fold_val.cpu().numpy(), valProbs)
                valLosses.append(valLoss)

            # check for early stopping
            if valLoss <= bestValLoss:
                bestValLoss = valLoss
                badEpochs = 0
                bestModelState = copy.deepcopy(model.state_dict()) # shallow copy retains references to original tensors
            else:
                badEpochs += 1
                if badEpochs >= earlyStoppingPatience:
                    break
            
            # tune learning rate down if plateauing
            scheduler.step(valLoss)
        
        # restore best model
        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        # EVALUATE MODEL
        # report stable score of last 5 epochs instead of lucky best epoch
        stableScore = np.mean(valLosses[-5:]) if len(valLosses) >= 5 else np.mean(valLosses)
        testScores.append(stableScore)
    
    # print train and test F1 for overfitting check
    print(f"Trial {trial.number} | Loss: {np.mean(testScores):.5f}")
    
    return np.mean(testScores)

# MAIN OPTUNA MAGIC
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# PRINT AND SAVE RESULTS
print(study.best_params) # a python dict
hyperparameters = {
    "modelParams": {
        "hidden_size": study.best_params["hidden_size"],
        "num_layers": study.best_params["num_layers"],
        "dropout": study.best_params["dropout"],
        "num_filters": study.best_params["num_filters"] if arch == 1 else None,
        "kernel_size": study.best_params["kernel_size"] if arch == 1 else None
    },
    "lookback": study.best_params["lookback"],
    "allParams": study.best_params
}
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filepath = os.path.join(directory, "hyperparameters.json")
with open(filepath, "w") as file:
    json.dump(hyperparameters, file, indent=4)