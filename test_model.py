import torch
import joblib
from custom_modules import dataparser, lstm
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import f1_score, log_loss, roc_auc_score, confusion_matrix

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, arch, version = globalVars.values()

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BUILD MODEL
ForexRNN, ForexHybrid = lstm.classBuilder()

# DEFINE FEATURES (copy-paste from the model training features exactly)
filepath = os.path.join("results", f"features_v{version}.json")
# deserialise json data
with open(filepath, "r") as file:
    featureList = json.load(file)["features"]

# DEFINE PARAMETERS (copy-paste from the model training hyperparameters exactly)
filepath = os.path.join("results", f"hyperparameters_v{version}.json")
# deserialise json data
with open(filepath, "r") as file:
    hyperparams = json.load(file)
    params = hyperparams["modelParams"]
    lookback = hyperparams["lookback"]

# LOAD MODEL AND SCALER
match arch:
    case 0:
        del params["num_filters"]
        del params["kernel_size"]
        model = ForexRNN(
            **params,
            input_size=len(featureList),
            output_size=3
        ).to(device)
    case 1:
        model = ForexHybrid(
            **params,
            input_size=len(featureList),
            output_size=3,
            lstm_dropout=params["dropout"]
        ).to(device)

filepath = os.path.join("models", f"NN_{instrument}_{granularity}_{yearNow}_v{version}.pth")
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

filepath = os.path.join("models", f"scaler_v{version}.pkl")
scaler = joblib.load(filepath)

# PARSE TEST DATA
df = dataparser.parseData(os.path.join("json_data", f"{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-01-01.json"))
timestamps = df["time"] # separate timestamps to avoid scaling
df.drop(columns=["time"], inplace=True)

# GET FEATURES AND LABELS (input and output)
filepath = os.path.join("results", f"features_v{version}.json")
with open(filepath, "r") as file:
    featureList = json.load(file)["features"]

features = df[featureList]
labels = df["target"]

labels = labels[features.index] # align indexes
timestamps = timestamps[features.index]

# SPLIT DATA
splitDate = f"{yearNow - 1}-01-01"

splitIdx = timestamps[timestamps > splitDate].index[0] # get index of first row after split date
splitIdx = timestamps.index.get_loc(splitIdx) # realign indexes

features_test = features.iloc[splitIdx:]
labels_test = labels.iloc[splitIdx:].values # convert to numpy array for correct indexing in createSequences()

# SCALE FEATURES, CREATE DATA SEQUENCES
features_test = scaler.transform(features_test)

def createSequences(fts, lbls, lookback):
    X, y = [], []
    for i in range(len(fts) - lookback):
        X.append(fts[i : i + lookback])
        y.append(lbls[i + lookback])
    return np.array(X), np.array(y)

X_test, y_test = createSequences(features_test, labels_test, lookback)

# CONVERT TO TENSORS
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)
# shape of X: (samples, timesteps, features)
# shape of y: (samples, output_classes)
testTrue = y_test.cpu().numpy() # for f1 score

# RUN INFERENCES
def predictByThreshold(probs, preds, threshold=0.335):
    filteredPreds = np.full(len(probs), 1) # initialise all flat
    # binary mask: returns array of True and False that is used to index preds
    filteredPreds[(preds == 0) & (probs[:, 0] > threshold)] = 0 # only if confidence exceeds threshold
    filteredPreds[(preds == 2) & (probs[:, 2] > threshold)] = 2
    return filteredPreds

with torch.no_grad():
    logits = model(X_test)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# EVALUATE MODEL
costScore = lstm.costScore(testTrue, preds)
f1Score = f1_score(testTrue, preds, average="macro", zero_division=0)
logLossScore = log_loss(testTrue, probs)
rocAucScore = roc_auc_score(testTrue, probs, multi_class="ovr", average="macro")
total, trainable = lstm.numParams(model)

# CONFUSION MATRIX
cmatrix = confusion_matrix(testTrue, preds)
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real ~", "Real +"], columns=["Pred -", "Pred ~", "Pred +"])
cmatrixDf["Count"] = cmatrixDf.sum(axis=1)
cmatrixDf.loc["Count"] = cmatrixDf.sum(axis=0)
print(f"Model version: {version}")
print(f"Cost score: {costScore:.5f}")
print(f"F1 score (macro-averaged): {f1Score:.5f}")
print(f"Log loss: {logLossScore:.5f}")
print(f"ROC-AUC score: {rocAucScore:.5f}")
print(f"Confusion matrix:\n{cmatrixDf}")
print(f"\nModel size: {trainable}")
with np.printoptions(threshold=50):
    print(probs)