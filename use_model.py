import torch
import joblib
from custom_modules import datafetcher, dataparser, lstm
import numpy as np
import os
import json

# GLOBAL VARIABLES
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, arch, version = globalVars.values()

# use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BUILD MODEL
ForexRNN, ForexHybrid = lstm.classBuilder()

# DEFINE FEATURES (copy-paste from the model training features exactly)
directory = "results"
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
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = torch.argmax(logits, dim=1).cpu().numpy()

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