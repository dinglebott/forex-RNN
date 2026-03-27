from custom_modules import dataparser, lstm
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import f1_score, log_loss, roc_auc_score, confusion_matrix
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, arch, _ = globalVars.values()
# GET HYPERPARAMETERS
filepath = os.path.join("results", "hyperparameters.json")
with open(filepath, "r") as file:
    hyperparameters = json.load(file) # hyperparameters is a python dict
print(f"Hyperparameters: {hyperparameters['allParams']}")
hiddenSize, numLayers, dropOut, lookback, optimiserName, learningRate, weightDecay, batchSize, clipGradNorm, numFilters, kernelSize = hyperparameters["allParams"].values()
# other
epochs = 100 # early stopping implemented
earlyStoppingPatience = 20
featureList = [
    "return", "return_4", "log_return", "log_return_4",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "oc_spread", "upper_wick", "lower_wick",
    "normalised_ema15", "normalised_ema50", "ema_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum",
    "open_return", "high_return", "low_return", "close_return"
]

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# extract top n features into list
featureList = [key for key in rawFeatures if rawFeatures[key] >= 0] # -1 for all features, 0 for positive only
print(f"Best {len(featureList)} features:", featureList)
features = df[featureList]
labels = df["target"]

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
scaler = StandardScaler()
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
trainTrue = y_train.cpu().numpy() # for f1 score later
testTrue = y_test.cpu().numpy()

# BUILD MODEL
ForexRNN, ForexHybrid = lstm.classBuilder()

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
criterion, optimiser, scheduler, classWeights = lstm.optimiserBundle(
    model, labels_train, device,
    optimiserName, learningRate, weightDecay
)

# TRAIN MODEL
dataset = torch.utils.data.TensorDataset(X_train, y_train) # Dataset object is a wrapper to keep tensors aligned
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
# DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

def batchPredict(model, X, batchSize=1024):
    allPreds = []
    for i in range(0, len(X), batchSize):
        batch = X[i : i + batchSize]
        logits = model(batch) # raw output of model => tensor of shape (samples, 3)
        preds = torch.argmax(logits, dim=1).cpu().numpy() # convert to predictions
        allPreds.append(preds)
    return np.concatenate(allPreds)

# for early stopping and saving best model
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
        loss = criterion(predictions, y_batch) # evaluate loss, returns loss tensor (1 element: the average loss across the batch)
        loss.backward() # backpropagation, compute loss with respect to each parameter
        nn.utils.clip_grad_norm_(model.parameters(), clipGradNorm) # prevent exploding gradients
        optimiser.step() # adjust gradients to minimise loss
    
    # validate (check for overfitting while training)
    model.eval() # disable dropout
    with torch.no_grad(): # disable gradient tracking to save memory
        valLogits = model(X_val) # raw output of model => tensor of shape (samples, 3)
        valLoss = criterion(valLogits, y_val)

    # check for early stopping
    if valLoss <= bestValLoss:
        bestValLoss = valLoss
        badEpochs = 0
        bestModelState = copy.deepcopy(model.state_dict()) # shallow copy retains references to original tensors
    else:
        badEpochs += 1
        if badEpochs >= earlyStoppingPatience:
            break
    
    # tune learning rate down
    scheduler.step(valLoss)
# restore best model
if bestModelState is not None:
    model.load_state_dict(bestModelState)

# TEST MODEL
model.eval() # disable dropout
with torch.no_grad(): # disable gradient tracking to save memory
    testLogits = model(X_test) # raw output of model => tensor of shape (samples, 3)
    testProbs = torch.softmax(testLogits, dim=1).cpu().numpy() # convert to probabilities for each class (first dimension sums to 1)
    testPreds = torch.argmax(testLogits, dim=1).cpu().numpy()
    trainPreds = batchPredict(model, X_train)
    
# EVALUATE MODEL
costScore = lstm.costScore(testTrue, testPreds)
f1Score = f1_score(testTrue, testPreds, average="macro", zero_division=0)
trainF1Score = f1_score(trainTrue, trainPreds, average="macro", zero_division=0)
lossScore = criterion(testLogits, y_test)
rocAucScore = roc_auc_score(testTrue, testProbs, multi_class="ovr", average="macro")
total, trainable = lstm.numParams(model)

# CONFUSION MATRIX
cmatrix = confusion_matrix(testTrue, testPreds)
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real ~", "Real +"], columns=["Pred -", "Pred ~", "Pred +"])
cmatrixDf["Count"] = cmatrixDf.sum(axis=1)
cmatrixDf.loc["Count"] = cmatrixDf.sum(axis=0)
print(f"Cost score: {costScore:.5f}")
print(f"F1 score (macro-averaged): {f1Score:.5f}")
print(f"Train F1 score: {trainF1Score:.5f}")
print(f"Log loss: {lossScore:.5f}")
print(f"ROC-AUC score: {rocAucScore:.5f}")
print(f"Confusion matrix:\n{cmatrixDf}")
print(f"\nModel size: {trainable}")
print(testProbs)

# SAVE MODEL
directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)
filepath = os.path.join(directory, f"NN_{instrument}_{granularity}_{yearNow}.pth")
torch.save(model.state_dict(), filepath)
print("\nModel saved to: " + f"NN_{instrument}_{granularity}_{yearNow}.pth")

# SAVE SCALER
filepath = os.path.join(directory, f"scaler.pkl")
joblib.dump(scaler, filepath)