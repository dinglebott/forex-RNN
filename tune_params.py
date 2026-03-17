import optuna
from custom_modules import dataparser, lstm
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import copy
from sklearn.metrics import f1_score
import os
import json

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
arch = 1 # 0 for LSTM, 1 for CNN/LSTM
# other
epochs = 80 # early stopping implemented
schedulerPatience = 5
earlyStoppingPatience = 20
featureList = ["return", "return_4", "log_return", "log_return_4",
               "atr_14", "volatility_regime",
               "bb_width", "bb_position",
               "hl_spread", "oc_spread", "upper_wick", "lower_wick",
               "normalised_ema15", "normalised_ema50", "ema_cross",
               "rsi_14", "macd_hist", "vol_ratio", "vol_momentum",
               "open_return", "high_return", "low_return", "close_return"]

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
numFeatures = 12
directory = "results"
filename = "features.json"
filepath = os.path.join(directory, filename)
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a python dict
# extract top n features into list
featureList = list(rawFeatures.keys())[:numFeatures]
print(f"Best {numFeatures} features:", featureList)

features = df[featureList]
labels = df["target"]

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA
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
        "hidden_size": trial.suggest_categorical("hidden_size", [128, 200, 256, 300, 384, 512]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3])
    }
    dropout = trial.suggest_float("dropout", 0.3, 0.6) # for CNN
    lstmDropout = dropout if params["num_layers"] > 1 else 0.0 # dropout only works for >1 layers
    lookback = trial.suggest_categorical("lookback", [15, 20, 25, 30])
    optimiserName = trial.suggest_categorical("optimiser", ["Adam", "RMSprop"])
    learningRate = trial.suggest_float("lr", 1e-5, 1e-3)
    weightDecay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    batchSize = trial.suggest_categorical("batch_size", [256, 512, 768, 1024])
    clipGradNorm = trial.suggest_float("clip_grad_norm", 4.0, 6.0)
    if arch == 1:
        numFilters = trial.suggest_categorical("num_filters", [32, 64, 96, 128])
        kernelSize = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])

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
        foldValTrue = y_fold_val.cpu().numpy()

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
        criterion, optimiser, scheduler, _ = lstm.optimiserBundle(model, labels_train, device,
                                                                  optimiserName, learningRate, weightDecay)

        # TRAIN MODEL
        dataset = torch.utils.data.TensorDataset(X_fold_train, y_fold_train) # Dataset object is a wrapper to keep tensors aligned
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
        # DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

        # for early stopping
        bestValLoss = 100
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
hyperparameters = {
    "modelParams": {
        "hidden_size": study.best_params["hidden_size"],
        "num_layers": study.best_params["num_layers"],
        "dropout": study.best_params["dropout"],
        "num_filters": study.best_params["num_filters"],
        "kernel_size": study.best_params["kernel_size"]
    },

    "lookback": study.best_params["lookback"],

    "allParams": study.best_params
}
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "hyperparameters.json"
filepath = os.path.join(directory, filename)
with open(filepath, "w") as file:
    json.dump(hyperparameters, file, indent=4)