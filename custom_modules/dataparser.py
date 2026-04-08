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
    df["volume"] = ultimateSmoother(df["volume"])[:len(df)]
    df["volume"] = df["volume"].clip(lower=1e-10)
    df["close_smooth"] = ultimateSmoother(df["close"])[:len(df)]
    # smoothed features
    df["smooth_return"] = np.log(df["close_smooth"] / df["close_smooth"].shift(1))
    df["dist_smooth"] = np.log(df["close"] / df["close_smooth"])
    
    # ADD FEATURES
    # helper
    def getEma(period):
        return df["close"].ewm(span=period, adjust=False).mean()
    # Raw
    df["open_return"] = np.log(df["open"] / df["close"].shift(1))
    df["high_return"] = np.log(df["high"] / df["close"].shift(1))
    df["low_return"] = np.log(df["low"] / df["close"].shift(1))
    df["close_return"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_return"] = np.log(df["volume"] / df["volume"].shift(1))
    # ATR
    trueRange = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1) # greatest of 3 values
    df["atr_14"] = trueRange.rolling(14).mean() / df["close"]
    df["volatility_regime"] = df["atr_14"] / df["atr_14"].rolling(50).mean()
    # Bollinger bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    # Structure
    raw_atr = trueRange.rolling(14).mean()
    df["hl_spread"] = np.log(df["high"] / df["low"])
    df["oc_spread"] = np.log(df["close"] / df["open"])
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / raw_atr
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / raw_atr
    # EMAs
    df["dist_ema15"] = np.log(df["close"] / getEma(15))
    df["dist_ema50"] = np.log(df["close"] / getEma(50))
    df["dist_ema100"] = np.log(df["close"] / getEma(100))
    df["ema_cross"] = np.log(getEma(15) / getEma(50))
    # RSI
    def rsi(series, n=14):
        delta = series.diff()
        avgGain = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        avgLoss = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        relativeStrength = avgGain / avgLoss
        return 100 - (100 / (1 + relativeStrength))
    df["rsi_14"] = rsi(df["close"]) - 50
    # MACD histogram
    macd = getEma(12) - getEma(26)
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - macd_signal) / df["close"]
    # Volume
    vol_sma30 = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma30
    df["vol_momentum"] = df["vol_ratio"] - df["vol_ratio"].rolling(5).mean()
    # ADX, DIs
    def adx(df, period=14):
        high = df["high"]
        low = df["low"]
        # directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        # +DM: up move is greater than down move and positive
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # smooth with wilder moving average (equivalent to EWM with alpha=1/period)
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        atr_smooth = trueRange.ewm(alpha=1/period, adjust=False).mean()
        # directional indicators
        plus_di = 100 * plus_dm_smooth / atr_smooth
        minus_di = 100 * minus_dm_smooth / atr_smooth
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_line = dx.ewm(alpha=1/period, adjust=False).mean()

        return plus_di, minus_di, adx_line
    
    df["plus_di"], df["minus_di"], df["adx"] = adx(df, period=14)
    df["di_diff"] = df["plus_di"] - df["minus_di"]
    df["adx_direction"] = df["di_diff"] * df["adx"] / 100
    # Support/resistance detection
    def detectSwingPoints(close: pd.Series, n: int = 5) -> tuple[pd.Series, pd.Series]:
        roll_max = close.rolling(window=2 * n + 1, center=True).max()
        roll_min = close.rolling(window=2 * n + 1, center=True).min()

        swing_highs = close.where(close == roll_max)
        swing_lows  = close.where(close == roll_min)

        # A swing at bar t is only confirmed n bars later
        swing_highs = swing_highs.shift(n)
        swing_lows  = swing_lows.shift(n)

        return swing_highs, swing_lows

    def computeSwingDistances(
        close: pd.Series,
        atr: pd.Series,
        swing_highs: pd.Series,
        swing_lows: pd.Series,
        lookback: int = 100,
    ) -> pd.DataFrame:
        n = len(close)
        dist_high = np.full(n, np.nan)
        dist_low  = np.full(n, np.nan)

        swing_high_vals = swing_highs.to_numpy()
        swing_low_vals  = swing_lows.to_numpy()
        close_vals      = close.to_numpy()
        atr_vals        = atr.to_numpy()

        for t in range(n):
            start = max(0, t - lookback)

            # Slice the lookback window, excluding t itself (causal)
            window_highs = swing_high_vals[start:t]
            window_lows  = swing_low_vals[start:t]

            # Extract confirmed swing prices (non-NaN entries)
            prior_highs = window_highs[~np.isnan(window_highs)]
            prior_lows  = window_lows[~np.isnan(window_lows)]

            if atr_vals[t] == 0 or np.isnan(atr_vals[t]):
                continue

            if len(prior_highs) > 0:
                nearest_high   = prior_highs[np.argmin(np.abs(prior_highs - close_vals[t]))]
                dist_high[t]   = (nearest_high - close_vals[t]) / atr_vals[t]

            if len(prior_lows) > 0:
                nearest_low   = prior_lows[np.argmin(np.abs(prior_lows - close_vals[t]))]
                dist_low[t]   = (close_vals[t] - nearest_low) / atr_vals[t]

        return pd.DataFrame(
            {"dist_high": dist_high, "dist_low": dist_low},
            index=close.index,
        )
    
    swing_highs, swing_lows = detectSwingPoints(df["close"], n=5)
    swing_distances = computeSwingDistances(
        close=df["close"],
        atr=df["atr_14"],
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        lookback=100
    )

    df = pd.concat([df, swing_distances], axis=1)

    # lagged features
    for lag in range(1, 5):
        df[f"close_lag{lag}"] = df["close_return"].shift(lag)
        df[f"vol_lag{lag}"] = df["vol_return"].shift(lag)

    # drop empty rows and return
    df.dropna(inplace=True)
    return df

def addTarget(df):
    df["forward_return"] = df["close"].shift(-2) - df["close"]
    conditions = [
        df["forward_return"] < -0.0004, # downward move
        df["forward_return"] > 0.0004 # upward move
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