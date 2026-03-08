## RAW DATA COLLECTION
Instrument: EUR_USD at 4-hourly timeframe\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2010-01-01 to 2026-01-01\
Pulled from OANDA REST-v20 API, stored in JSON format\
<br/>

## DATASETS
**Feature selection ():**\

**Hyperparameter tuning (Optuna):**\

**Final model training:**\
Train: 2010-2024, Test: 2025\
<br/>

## INITIAL FEATURE ENGINEERING
**Scoring metric:** SHAP values\
**Price:**\
Return => Percentage change from previous close\
4-period return => Percentage change from close 4 candles prior\
Log return => ln(close / previous close)
4-period log return => ln(close / close 4 candles prior)
**Structure:**\
High-low spread (normalised) => (H - L) / C\
Open-close spread (normalised) => (C - O) / C\
Upper wick => (High - candle top) / atr_14\
Lower wick => (Candle bottom - low) / atr_14\
**Trend:**\
15-period EMA (normalised) => (C / EMA) - 1\
50-period EMA (normalised) => (C / EMA) - 1\
EMA cross => ema15 - ema50\
**Momentum:**\
14-period RSI\
12/26/9-period MACD histogram => (ema12 - ema26) - signal\
**Volatility:**\
14-period ATR\
Volatility regime => atr_14 / atr_14 50-period mean\
Bollinger band width => (upperband - lowerband) / midband\
**Volume:**\
Volume ratio => volume / volume sma30\
Volume momentum => vol_ratio - vol_ratio 5-period mean\
**Mean reversion:**\
Bollinger band position => (C - lowerband) / (upperband - lowerband)\
**XGBoost signals:**\
Probability of each class as predicted by an XGBoost model
<br/>

## HYPERPARAMETER TUNING
**Scoring metric:** Macro-adjusted F1 score (see Explanation of metrics below)\
**Hyperparameters tested:**\

<br/>

## MODEL EVALUATION
**Explanation of metrics:**\
Positive = 1 (upward move), Negative = 0 (downward move)\
Accuracy (0-100) => (Correct predictions / Total predictions) * 100%\
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's\
<br/>