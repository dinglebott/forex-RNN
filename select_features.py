from custom_modules import dataparser, lstm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import f1_score
import os

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
arch = 1 # 0 for LSTM, 1 for CNN/LSTM
# hyperparameters
hiddenSize = 192 # no. of neurons in hidden state
numLayers = 1 # no. of layers in the LSTM
dropOut = 0.45 # equivalent of subsample for RNN
lookback = 20
optimiserName = "RMSprop"
learningRate = 2.36e-06
weightDecay = 0.000867
batchSize = 256
clipGradNorm = 4.46
# CNN params
numFilters = 32
kernelSize = 5
# other
epochs = 150 # early stopping implemented
earlyStoppingPatience = 100
featureList = [
    "open_return", "high_return", "low_return", "close_return", "vol_return", "smooth_return",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "oc_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "ema_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum",
]

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENSURE REPRODUCIBILITY
torch.manual_seed(42) # ensure reproducible results
np.random.seed(42)

# LOAD DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-01-01.json")
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

# SEPARATE FEATURES AND LABELS (input and output)
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
valTrue = y_val.cpu().numpy() # for f1 score later
testTrue = y_test.cpu().numpy()

# BUILD MODELS
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
criterion, optimiser, scheduler, _ = lstm.optimiserBundle(
    model, labels_train, device,
    optimiserName, learningRate, weightDecay
)

# TRAIN MODEL
dataset = torch.utils.data.TensorDataset(X_train, y_train) # Dataset object is a wrapper to keep tensors aligned
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
# DataLoader returns an iterator that yields batches as a tuple of tensors (X_batch, y_batch)

# for early stopping and saving best model
bestCostScore = 0
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
        valPreds = torch.argmax(valLogits, dim=1).cpu().numpy() # convert to predictions
    valF1Score = f1_score(valTrue, valPreds, average="macro", zero_division=0)
    valCostScore = lstm.costScore(valTrue, valPreds)
    print(f"EPOCH {epoch + 1} | Train loss: {avgLoss:.4f} | Val loss: {valLoss:.4f} | F1: {valF1Score:.4f} | Cost score: {valCostScore:.4f}")

    # check for early stopping
    if valCostScore >= bestCostScore:
        bestCostScore = valCostScore
        badEpochs = 0
        bestModelState = copy.deepcopy(model.state_dict()) # shallow copy retains references to original tensors
    else:
        badEpochs += 1
        if badEpochs >= earlyStoppingPatience:
            print("EARLY STOPPING NOW")
            break
    
    # tune learning rate down
    scheduler.step(valCostScore)
# restore best model
if bestModelState is not None:
    model.load_state_dict(bestModelState)

# TEST MODEL
model.eval() # disable dropout
with torch.no_grad(): # disable gradient tracking to save memory
    testLogits = model(X_test) # raw output of model => tensor of shape (samples, 3)
    testProbs = torch.softmax(testLogits, dim=1).cpu().numpy()
    testPreds = torch.argmax(testLogits, dim=1).cpu().numpy() # convert to predictions
    
# EVALUATE MODEL
costScore = lstm.costScore(testTrue, testPreds)

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
        score = lstm.costScore(testTrue, preds)
        scores.append(score)
    # average score for this feature
    avgScore = np.mean(scores)
    # calculate how much the model was hurt by shuffling this feature
    importances.append(costScore - avgScore)

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