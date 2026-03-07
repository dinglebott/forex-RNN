# EXPORTS:
# parseData() adds features and returns DataFrame
# splitByDate() returns specified slice of DataFrame by date
import json
import pandas as pd
import numpy as np

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

    # ADD FEATURES
    # helper
    def getEma(period):
        return df["close"].ewm(span=period, adjust=False).mean()
    # returns
    df["return"] = df["close"].pct_change() # removed v5+
    # spreads
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_spread"] = (df["close"] - df["open"]) / df["close"] # removed v5+
    df["body_ratio"] = (df["oc_spread"] / df["hl_spread"]).clip(-1, 1) # prevent infinity values
    # EMAs
    for period in (15, 50):
        df[f"raw_ema{period}"] = getEma(period)
        df[f"normalised_ema{period}"] = (df["close"] / df[f"raw_ema{period}"]) - 1
    # RSI
    def rsi(series, n=14):
        delta = series.diff()
        avgGain = delta.clip(lower=0).rolling(n).mean()
        avgLoss = (-delta.clip(upper=0)).rolling(n).mean()
        relativeStrength = avgGain / avgLoss
        return 100 - (100 / (1 + relativeStrength))
    df["rsi_14"] = rsi(df["close"])
    # MACD histogram
    macd = getEma(12) - getEma(26)
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    # ATR
    trueRange = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1) # greatest of 3 values
    df["atr_14"] = trueRange.rolling(14).mean() / df["close"]
    # Bollinger bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    # volume ratio
    vol_sma30 = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma30
    # lagged returns and volumes
    for lag in range(1, 6):
        df[f"return_lag{lag}"] = df["return"].shift(lag) # lag5 removed v5+
        df[f"vol_ratio_lag{lag}"] = df["vol_ratio"].shift(lag) # lag5 removed v5+
    
    # new features (v5+)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["atr_14"] # removed v5.3+
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["atr_14"] # removed v5.3+
    df["direction"] = np.sign(df["close"] - df["open"])
    df["volatility_momentum"] = df["rsi_14"] * df["atr_14"]
    df["vol_trend"] = df["vol_ratio"] * df["normalised_ema50"]
    df["trend_strength"] = abs(df["normalised_ema15"] - df["normalised_ema50"])
    df["volatility_regime"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

    # new features (v5.1+)
    df["atr_adjusted_return"] = df["return"] / df["atr_14"]
    df["return_accel"] = df["return_lag1"] - df["return_lag2"] # removed v5.3+
    df["vol_momentum"] = df["vol_ratio"] - df["vol_ratio"].rolling(5).mean()
    df["dist_ema15"] = (df["close"] - df["raw_ema15"]) / df["atr_14"]
    
    # drop empty rows and return
    df.dropna(inplace=True)
    return df

def splitByDate(df, start, end):
    times = pd.to_datetime(df["time"].str.split(".").str[0], format="%Y-%m-%dT%H:%M:%S") # convert timestamps to datetime objects
    # .str applies operation to entire series cellwise
    mask = (times >= start) & (times < end)
    return df[mask]
    # df[boolean-mask] filters out values according to the mask