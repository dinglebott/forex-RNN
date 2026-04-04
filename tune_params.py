import optuna
from custom_modules import dataparser, lstm
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    "open_return", "high_return", "low_return", "close_return", "vol_return", "smooth_return", "dist_smooth",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "oc_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "dist_ema100", "ema_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum", "adx_direction",
    "dist_high", "dist_low"
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
df = dataparser.addTarget(df)
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

# GET FEATURES AND LABELS (input and output)
filepath = os.path.join("results", "features.json")
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a python dict
# extract positive features into list
featureList = [key for key in rawFeatures if rawFeatures[key] >= 0] # -1 for all features, 0 for positive only
featureList = [
    "adx_direction", "ema_cross", "bb_position", "macd_hist",
    "upper_wick", "lower_wick", "dist_high", "dist_low", "dist_ema15", "rsi_14",
    "volatility_regime", "bb_width", "atr_14",
    "vol_ratio", "vol_momentum",
    "smooth_return", "dist_smooth"
] # for manual feature setting (comment out when not needed)
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

# DATA SEQUENCES (created within objective)
def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

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
        "hidden_size": trial.suggest_categorical("hidden_size", [512]),
        "num_layers": trial.suggest_categorical("num_layers", [1])
    }
    dropout = trial.suggest_float("dropout", 0.05, 0.15) # for CNN
    lookback = trial.suggest_categorical("lookback", [15, 20, 25])
    optimiserName = trial.suggest_categorical("optimiser", ["RMSprop"])
    learningRate = trial.suggest_float("lr", 9e-5, 1e-4)
    weightDecay = trial.suggest_float("weight_decay", 9.5e-5, 1.5e-4)
    batchSize = trial.suggest_categorical("batch_size", [512])
    clipGradNorm = trial.suggest_float("clip_grad_norm", 5.5, 6.5)
    if arch == 1:
        numFilters = trial.suggest_categorical("num_filters", [16, 24])
        kernelSize = trial.suggest_categorical("kernel_size", [3, 5])
        lstmDropout = dropout if params["num_layers"] > 1 else 0.0 # dropout only works for >1 layers

    # TRAIN TEST SPLIT
    def expandingWindowSplit(n_samples, n_splits, val_ratio):
        splits = []
        for f in range(n_splits):
            trainEnd = int((1 - (n_splits - f) * val_ratio) * n_samples)
            valEnd = int(trainEnd + (val_ratio * n_samples))
            trainIdx = range(trainEnd)
            valIdx = range(trainEnd, valEnd)
            splits.append((trainIdx, valIdx))
        return splits

    trainValSplit = expandingWindowSplit(len(features_train), 3, 0.1)

    # SUBFOLD LOOP
    testScores = []
    for trainIdxs, valIdxs in trainValSplit:
        # SPLIT TRAIN AND VALIDATION FEATURES/LABELS
        features_fold_train = features_train.iloc[trainIdxs]
        labels_fold_train = labels_train[trainIdxs]
        features_fold_val = features_train.iloc[valIdxs]
        labels_fold_val = labels_train[valIdxs]
        # SCALE FEATURES
        scaler = StandardScaler()
        features_fold_train = scaler.fit_transform(features_fold_train)
        features_fold_val = scaler.transform(features_fold_val)
        # CREATE SEQUENCES
        X_fold_train, y_fold_train = createSequences(features_fold_train, labels_fold_train, lookback)
        X_fold_val, y_fold_val = createSequences(features_fold_val, labels_fold_val, lookback)
        # CONVERT TO TENSORS
        X_fold_train = torch.tensor(X_fold_train, dtype=torch.float32, device=device)
        y_fold_train = torch.tensor(y_fold_train, dtype=torch.long, device=device)
        X_fold_val = torch.tensor(X_fold_val, dtype=torch.float32, device=device)
        y_fold_val = torch.tensor(y_fold_val, dtype=torch.long, device=device)
        
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
            model, labels_fold_train, device,
            optimiserName, learningRate, weightDecay
        )

        # TRAIN MODEL
        dataset = torch.utils.data.TensorDataset(X_fold_train, y_fold_train) # Dataset object is a wrapper to keep tensors aligned
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
        # DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

        # for early stopping
        bestValLoss = 100
        badEpochs = 0
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
                valLoss = batchLoss(model, X_fold_val, y_fold_val, criterion) # cost loss
                valLosses.append(valLoss)

            # check for early stopping
            if valLoss <= bestValLoss:
                bestValLoss = valLoss
                badEpochs = 0
            else:
                badEpochs += 1
                if badEpochs >= earlyStoppingPatience:
                    break
            
            # tune learning rate down if plateauing
            scheduler.step(valLoss)

        # EVALUATE MODEL
        # report stable score of last 5 epochs instead of lucky best epoch
        stableScore = np.mean(valLosses[-5:]) if len(valLosses) >= 5 else np.mean(valLosses)
        testScores.append(stableScore)
    
    # RETURN TO OPTUNA
    return np.mean(testScores)

# MAIN OPTUNA MAGIC
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

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