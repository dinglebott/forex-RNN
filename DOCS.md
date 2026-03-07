## RAW DATA COLLECTION
Instrument: EUR_USD at 4-hourly timeframe\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2010-01-01 to 2026-01-01\
Pulled from OANDA REST-v20 API, stored in JSON format\
<br/>

## DATASETS
**Feature selection (SHAP):**\
10 folds of 7-year rolling windows\
Train: 2010-2015, Test: 2016\
Train: 2011-2016, Test: 2017\
...\
Train: 2019-2024, Test: 2025\
**Hyperparameter tuning (Optuna):**\
5 folds of 7-year rolling windows (advance by 2 years per fold)\
2011-2017, 2013-2019, ..., 2019-2025\
Cross-validation performed by dividing into subfolds\
**Final model training:**\
Train: 2010-2024, Test: 2025\
<br/>

## INITIAL FEATURE ENGINEERING
**Scoring metric:** SHAP values\
**Price:**\
Return => Percentage change from previous close *(Removed in v5+)*\
High-low spread (normalised) => (H - L) / C\
Open-close spread (normalised) => (C - O) / C *(Removed in v5+)*\
Body ratio => OC spread / HL spread\
**Trend:**\
12-period EMA (normalised) => (C / EMA) - 1\
50-period EMA (normalised) => (C / EMA) - 1\
**Momentum:**\
14-period RSI\
12/26/9-period MACD histogram => (ema12 - ema26) - signal\
**Volatility:**\
14-period ATR\
Bollinger band width (normalised) => (upperband - lowerband) / midband\
**Volume:**\
Volume ratio => volume / volumesma30\
**Mean reversion:**\
Bollinger band position => (C - lowerband) / (upperband - lowerband)\
**Lagged features:**\
1/2/3/4/5-period lagged returns => Return values of previous 5 candles *(Removed lag5 in v5+)*\
1/2/3/4/5-period lagged volume => Volume values of previous 5 candles *(Removed lag5 in v5+)*\
**Hybrid features (added in v5+):**\
Volatility-adjusted momentum => atr_14 * rsi_14\
Volume-trend confluence => vol_ratio * ema50\
Trend strength => abs(ema15 - ema50)\
Volatility regime => atr_14 / atr_14 50-period mean\
Candle direction => Sign of close - open\
Upper wick => (High - candle top) / atr_14\
Lower wick => (Candle bottom - low) / atr_14\
**More features (added in v5.1+):**\
Volatility-adjusted return => return / atr_14\
Return acceleration => return_lag1 - return_lag2\
Volume momentum => vol_ratio - vol_ratio 5-period mean\
Distance from EMA (volatility-adjusted) => (C - ema15) / atr_14\
<br/>

## HYPERPARAMETER TUNING
**Scoring metric:** Macro-adjusted F1 score (see Explanation of metrics below)\
**Hyperparameters tested:**\
n_estimators: No. of decision trees to build\
max_depth: Maximum number of levels each tree can grow its depth to - lower values can reduce overfitting by limiting complexity\
learning_rate: Lower value reduces the contribution of each tree and prevents overfitting\
subsample: Fraction of data sampled per tree\
colsample_bytree: Fraction of features sampled per tree\
min_child_weight: Higher values make model require more evidence to make a split\
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