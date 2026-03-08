# EXPORTS:
# xgbSignals => dataframe containing xgb predictions for 2011-2024
import xgboost as xgb
from . import dataparser
from datetime import datetime
import numpy as np
import pandas as pd

# LOAD DATA
df = dataparser.parseData("json_data/EUR_USD_H4_2010-01-01_2026-01-01.json")

# FEATURES AND HYPERPARAMETERS (obtained from trading-trees project)
features = ["atr_14", "volatility_momentum", "vol_ratio_lag3", "volatility_regime", "vol_ratio_lag4", "hl_spread", "normalised_ema50",
            "vol_ratio_lag1", "trend_strength", "bb_width", "vol_ratio", "vol_momentum", "dist_ema15", "macd_hist", "vol_trend"]
params = {
    "max_depth": 3,
    "learning_rate": 0.0737224128,
    "subsample": 0.4283120798,
    "colsample_bytree": 0.5598602095,
    "min_child_weight": 82,
    "reg_alpha": 4.9376459424,
    "reg_lambda": 17.8068765017
}

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
warmupFold = dataparser.splitByDate(df, datetime(2010, 1, 1), datetime(2010, 6, 1))
warmupFold2 = dataparser.splitByDate(df, datetime(2010, 6, 1), datetime(2011, 1, 1))
fold1 = dataparser.splitByDate(df, datetime(2011, 1, 1), datetime(2014, 1, 1))
fold2 = dataparser.splitByDate(df, datetime(2014, 1, 1), datetime(2018, 1, 1))
fold3 = dataparser.splitByDate(df, datetime(2018, 1, 1), datetime(2022, 1, 1))
fold4 = dataparser.splitByDate(df, datetime(2022, 1, 1), datetime(2025, 1, 1))

# DEFINE DATASETS
X_warmup, y_warmup = warmupFold[features], warmupFold["target"]
X_warmup2, y_warmup2 = warmupFold2[features], warmupFold2["target"]
X_1, y_1 = fold1[features], fold1["target"]
X_2, y_2 = fold2[features], fold2["target"]
X_3, y_3 = fold3[features], fold3["target"]
X_4, y_4 = fold4[features], fold4["target"]

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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    # predict and record
    output = model.predict_proba(X_pred)
    xgbSignals = pd.concat([xgbSignals, pd.DataFrame(output)], ignore_index=True)

# REFORMAT FINAL OUTPUT
rowsPredicted = pd.concat(X_folds[2:], ignore_index=False) # get predicted rows with original indexes
xgbSignals.index = rowsPredicted.index # align indexes to original
xgbSignals.rename(columns={
                           0: "xgb_0",
                           1: "xgb_1",
                           2: "xgb_2"
                           }, inplace=True)