# EXPORTS:
# xgbSignals => dataframe containing xgb predictions for 2011-2024
import xgboost as xgb
from . import dataparser
from datetime import datetime
import numpy as np
import pandas as pd
import os
import json

yearNow = 2026
version = 7
# LOAD DATA
df = dataparser.parseData(f"json_data/EUR_USD_H4_{yearNow - 21}-01-01_{yearNow}-01-01.json")
directory = "custom_modules"
filename = f"XGBoost_EUR_USD_H4_2026_v{version}.json"
trainedXgbFilepath = os.path.join(directory, filename)

# FEATURES AND HYPERPARAMETERS (from trading-trees project)
filename = f"featues_v{version}.json"
filepath = os.path.join(directory, filename)
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # python dict
features = list(rawFeatures.keys())

filename = f"hyperparameters_v{version}.json"
filepath = os.path.join(directory, filename)
with open(filepath, "r") as file:
    params = json.load(file)
params["max_depth"] = int(params["max_depth"])
params["min_child_weight"] = int(params["min_child_weight"])

# TARGET VARIABLE: next 4 candles net return => positive (2) or negative (0)
df["forward_return"] = (df["close"].shift(-4) / df["close"]) - 1
conditions = [
    df["forward_return"] < -0.0015, # downward move
    df["forward_return"] > 0.0015 # upward move
]
choices = [0, 2]
df["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
df.dropna(inplace=True)

# SPLIT DATA
warmupFold = dataparser.splitByDate(df, datetime(2005, 1, 1), datetime(2005, 6, 1))
warmupFold2 = dataparser.splitByDate(df, datetime(2005, 6, 1), datetime(2006, 1, 1))
fold1 = dataparser.splitByDate(df, datetime(2006, 1, 1), datetime(2011, 1, 1))
fold2 = dataparser.splitByDate(df, datetime(2011, 1, 1), datetime(2016, 1, 1))
fold3 = dataparser.splitByDate(df, datetime(2016, 1, 1), datetime(2021, 1, 1))
fold4 = dataparser.splitByDate(df, datetime(2021, 1, 1), datetime(2025, 1, 1))
testFold = dataparser.splitByDate(df, datetime(2025, 1, 1), datetime(2026, 1, 1))

# DEFINE DATASETS
X_warmup, y_warmup = warmupFold[features], warmupFold["target"]
X_warmup2, y_warmup2 = warmupFold2[features], warmupFold2["target"]
X_1, y_1 = fold1[features], fold1["target"]
X_2, y_2 = fold2[features], fold2["target"]
X_3, y_3 = fold3[features], fold3["target"]
X_4, y_4 = fold4[features], fold4["target"]
X_test, y_test = testFold[features], testFold["target"]

X_folds = [X_warmup, X_warmup2, X_1, X_2, X_3, X_4]
y_folds = [y_warmup, y_warmup2, y_1, y_2, y_3, y_4]

# CREATE MODEL AND GET PREDICTIONS
xgbSignals = pd.DataFrame(columns=[0, 1, 2])
for x in range(1, 5):
    # reset model
    model = xgb.XGBClassifier(**params, eval_metric="mlogloss",
                            n_estimators=1000, # high ceiling
                            early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
                            random_state=42,
                            device="cpu", # avoid transferring data to gpu
                            tree_method="hist")
    # define datasets
    X_train = pd.concat(X_folds[:x])
    y_train = pd.concat(y_folds[:x])
    X_val = X_folds[x]
    y_val = y_folds[x]
    X_pred = X_folds[x+1]
    # train
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    # predict and record
    output = model.predict_proba(X_pred)
    xgbSignals = pd.concat([xgbSignals, pd.DataFrame(output)], ignore_index=False)

# LOAD AND USE FULLY-TRAINED MODEL ON TEST FOLD (2025)
model = xgb.XGBClassifier()
model.load_model(trainedXgbFilepath)
testPreds = model.predict_proba(X_test)
xgbSignals = pd.concat([xgbSignals, pd.DataFrame(testPreds)], ignore_index=False)

# REFORMAT FINAL OUTPUT
rowsPredicted = pd.concat([X_1, X_2, X_3, X_4, X_test], ignore_index=False) # get rows with a prediction with original indexes
xgbSignals.index = rowsPredicted.index # align indexes to original
xgbSignals.rename(columns={
                           0: "xgb_0",
                           1: "xgb_1",
                           2: "xgb_2"
                           }, inplace=True)