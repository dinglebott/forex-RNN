# EXPORTS:
# parseData() adds features and returns DataFrame
# splitByDate() returns specified slice of DataFrame by date
import json
import pandas as pd
import numpy as np

def ultimateSmoother(series, period=5):
    values = series.values.astype(float)
    result = np.zeros(len(values))
    # coefficients
    f = (1.41421 * np.pi) / period
    a1 = np.exp(-f)
    c2 = 2 * a1 * np.cos(f)
    c3 = -(a1 ** 2)
    c1 = (1 - c2 - c3) / 4
    # initialise
    result[0] = values[0]
    result[1] = values[1]
    # recurrence relation
    for i in range(2, len(values)):
        result[i] = (
            (1 - c1) * values[i] # weighted input
            + (2 * c1 - c2) * values[i-1] # weighted previous input
            - (c1 + c3) * values[i-2] # weighted input 2 bars ago
            + c2 * result[i-1] # feedback from previous output
            + c3 * result[i-2] # feedback from output 2 bars ago
        )
    return result

def parseData(jsonPath):
    # deserialise json data
    with open(jsonPath, "r") as file:
        rawData = json.load(file) # rawData is a Python dict
    
    # unpack dict into DataFrame
    records = []
    for c in rawData["candles"]:
        if c["complete"]:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]), # convert from string
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })
    df = pd.DataFrame(records)

    # denoise
    df["close_smooth"] = ultimateSmoother(df["close"])[:len(df)]
    df["volume"] = ultimateSmoother(df["volume"])[:len(df)]
    # rebuild candle around smoothed close
    df["open"] = df["close_smooth"] + (df["open"] - df["close"])
    df["high"] = df["close_smooth"] + (df["high"] - df["close"])
    df["low"] = df["close_smooth"] + (df["low"] - df["close"])
    #df["close_smooth"] = df["close"]

    # ADD FEATURES
    # helper
    def getEma(period):
        return df["close_smooth"].ewm(span=period, adjust=False).mean()
    # Raw
    df["open_return"] = (df["open"] / df["close_smooth"].shift(1)) - 1
    df["high_return"] = (df["high"] / df["close_smooth"].shift(1)) - 1
    df["low_return"] = (df["low"] / df["close_smooth"].shift(1)) - 1
    df["close_return"] = (df["close_smooth"] / df["close_smooth"].shift(1)) - 1
    # Returns
    df["return"] = df["close_smooth"].pct_change()
    df["return_4"] = df["close_smooth"].pct_change(4)
    df["log_return"] = np.log(df["close_smooth"] / df["close_smooth"].shift(1)) # natural log (base e)
    df["log_return_4"] = np.log(df["close_smooth"] / df["close_smooth"].shift(4))
    # ATR
    trueRange = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close_smooth"].shift(1)).abs(),
        (df["low"]  - df["close_smooth"].shift(1)).abs()
    ], axis=1).max(axis=1) # greatest of 3 values
    df["atr_14"] = trueRange.rolling(14).mean() / df["close_smooth"]
    df["raw_atr"] = trueRange.rolling(14).mean() / df["close"]
    df["volatility_regime"] = df["atr_14"] / df["atr_14"].rolling(50).mean()
    # Bollinger bands
    bb_mid = df["close_smooth"].rolling(20).mean()
    bb_std = df["close_smooth"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close_smooth"] - bb_lower) / (bb_upper - bb_lower)
    # Structure
    df["hl_spread"] = (df["high"] - df["low"]) / df["close_smooth"]
    df["oc_spread"] = (df["close_smooth"] - df["open"]) / df["close_smooth"]
    df["upper_wick"] = (df["high"] - df[["open", "close_smooth"]].max(axis=1)) / df["atr_14"]
    df["lower_wick"] = (df[["open", "close_smooth"]].min(axis=1) - df["low"]) / df["atr_14"]
    # EMAs
    for period in (15, 50):
        df[f"raw_ema{period}"] = getEma(period)
        df[f"normalised_ema{period}"] = (df["close_smooth"] / df[f"raw_ema{period}"]) - 1
    df["ema_cross"] = df["normalised_ema15"] - df["normalised_ema50"]
    # RSI
    def rsi(series, n=14):
        delta = series.diff()
        avgGain = delta.clip(lower=0).rolling(n).mean()
        avgLoss = (-delta.clip(upper=0)).rolling(n).mean()
        relativeStrength = avgGain / avgLoss
        return 100 - (100 / (1 + relativeStrength))
    df["rsi_14"] = rsi(df["close_smooth"])
    # MACD histogram
    macd = getEma(12) - getEma(26)
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    # Volume
    vol_sma30 = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma30
    df["vol_momentum"] = df["vol_ratio"] - df["vol_ratio"].rolling(5).mean()
    # TODO: for the xgboost
    
    # TARGET VARIABLE
    df["forward_return"] = (df["close"].shift(-4) / df["close"]) - 1
    conditions = [
        df["forward_return"] < -0.0015, # downward move
        df["forward_return"] > 0.0015 # upward move
    ]
    choices = [0, 2]
    df["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)

    # drop empty rows and return
    df.dropna(inplace=True)
    return df

def splitByDate(df, start, end):
    times = pd.to_datetime(df["time"].str.split(".").str[0], format="%Y-%m-%dT%H:%M:%S") # convert timestamps to datetime objects
    # .str applies operation to entire series cellwise
    mask = (times >= start) & (times < end)
    return df[mask]
    # df[boolean-mask] filters out values according to the mask