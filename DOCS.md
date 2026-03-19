## RAW DATA COLLECTION
Instrument: EUR_USD at 4-hourly timeframe\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2005-01-01 to 2026-01-01\
Pulled from OANDA REST-v20 API, stored in JSON format\
<br/>

## DATASETS
**Feature selection (Permutation importance):**\
Train: 2005-2023\
Validate: 2024 (used for early stopping of epochs)\
Test: 2025 (used to produce baseline F1 score, and for permuting)\
**Hyperparameter tuning (Optuna):**\
Train/Validate: 2005-2024 (split handled by TimeSeriesSplit cross-validation with 3 folds)\
Test: 2025 (used to produce F1 score for Optuna to maximise)\
**Final model training:**\
Train: 2010-2024\
Test: 2025\
<br/>

## FEATURE ENGINEERING
**Scoring metric:** Permutation importances\
Each feature is shuffled randomly across all the samples of the test set, effectively turning it into garbage data. The impact on the model's F1 score when predicting the test set is then measured. The process is repeated for every feature and the scores are compared.\
**Price:**\
Open return => ln(O / previous C)\
High return => ln(H / previous C)\
Low return => ln(L / previous C)\
Close return => ln(C / previous C)\
Volume return => ln(V / previous V)\
Smooth return => ln(smoothed C / previous smoothed C)
*Volume and smoothed C have been smoothed by Ehler's UltimateSmoother algorithm*\
**Structure:**\
High-low spread => ln(H - L)\
Open-close spread => ln(C - O)\
Upper wick => (High - candle top) / atr_14\
Lower wick => (Candle bottom - low) / atr_14\
**Trend:**\
15-period EMA => ln(C / EMA)\
50-period EMA => ln(C / EMA)\
EMA cross => ln(ema15 / ema50)\
**Momentum:**\
14-period RSI\
12/26/9-period MACD histogram => ((ema12 - ema26) - signal) / C\
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
Probability of each class as predicted by an XGBoost model\
<br/>

## HYPERPARAMETER TUNING
**Scoring metric:** Macro-adjusted F1 score (see Explanation of metrics below)\
**Hyperparameters tested:**\
`hidden_size`: No. of neurons per layer\
`num_layers`: No. of stacked LSTM layers\
`dropout`: Fraction of neurons randomly deactivated for training\
`lookback`: Length of lookback window\
`optimiserName`: Type of optimiser used\
`learningRate`: Rate of weight change\
`weightDecay`: Regularisation, penalises large weights\
`batchSize`: Size of batches loaded by dataloader\
`clipGradNorm`: Clips gradients to prevent infinities\
`num_filters`: No. of kernels applied by CNN\
`kernel_size`: Size of pattern CNN recognises\
**Initial search spaces:**\
`hidden_size`: [16, 32, 64, 128, 256, 512, 768]\
`num_layers`: [1, 2, 3]\
`dropout`: 0.1 - 0.5\
`lookback`: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]\
`optimiserName`: ["AdamW", "RMSprop"]\
`learningRate`: 1e-4 - 1e-2\
`weightDecay`: 1e-6 - 1e-3\
`batchSize`: [64, 128, 256, 512]\
`clipGradNorm`: 2.0 - 8.0\
`num_filters`: [16, 32, 64, 128, 512]\
`kernel_size`: [3, 5, 7]\
<br/>

## MODEL EVALUATION
**Explanation of metrics:**\
Negative = 0, Flat = 1, Positive = 2\
Accuracy (0-100) => (Correct predictions / Total predictions) * 100%\
Cost score (0-1) => Cost-weighted score (penalises wrong-direction predictions more)
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's\
<br/>

### Model 4
*Changes from v3: Implement cost-sensitive score and loss function, reduce model complexity*\
**Train:** 2005 - 2024\
**Test:** 2025\
**Features:** ["close_return", "dist_ema50", "macd_hist", "smooth_return", "upper_wick", "lower_wick", "atr_14", "rsi_14", "low_return", "dist_ema15"]\
**Hyperparameters:**\
hidden_size: 32\
num_layers: 1\
dropout: 0.36370\
lookback: 15\
optimiser: RMSprop\
lr: 2.9658e-5\
weight_decay: 5.9564e-4\
batch_size: 64\
clip_grad_norm: 3.9427\
num_filters: 32\
kernel_size: 7\
**Cost matrix parameters:** 1.3, 0.1\
**Accuracy:** 36.808%\
**Cost score:** 0.5540\
**F1 score (macro-averaged):** 0.36635\
**Train F1 score:** 0.35241\
**ROC-AUC score:** 0.54647\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 154 | 172 | 140 |
| Real ~ | 155 | 222 | 172 |
| Real + |121 | 210 | 189 |
<br/>

### Model 3 (invalid: data leaked in wavelet deconstruction)
*Changes from v2: Implement wavelet de-noising, minimum probability threshold for flat class, and class weighting*\
**Train:** 2005 - 2024\
**Test:** 2025\
**Features:** ["atr_14",\
"bb_width", "bb_position",\
"hl_spread", "lower_wick",\
"normalised_ema15", "normalised_ema50", "ema_cross",\
"rsi_14", "macd_hist", "vol_ratio", "vol_momentum"]\
**Hyperparameters:**\
hidden_size: 128\
num_layers: 2\
dropout: 0.26\
lookback: 20\
optimiser: Adam\
lr: 0.001\
weight_decay: 0.0006\
batch_size: 1024\
clip_grad_norm: 5.1\
num_filters: 128\
kernel_size: 5\
**Accuracy:** 46.601%\
**F1 score (macro-averaged):** 0.46928\
**ROC-AUC score:** 0.64912\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 251 | 133 | 62 |
| Real ~ | 216 | 235 | 141 |
| Real + | 84 | 181 | 227 |
<br/>

### Model 2
*Changes from v1: Implemented learning rate scheduler, added CNN layers, expanded training data to start from 2005*\
**Train:** 2005 - 2024\
**Test:** 2025\
**Features:** ["atr_14", "volatility_regime",\
"bb_position",\
"hl_spread", "upper_wick"\
"normalised_ema50", "ema_cross",\
"rsi_14", "macd_hist", "vol_ratio", "vol_momentum",\
"xgb_1", "xgb_2"]\
**Hyperparameters:**\
hidden_size: 128\
num_layers: 2\
dropout: 0.26\
lookback: 20\
optimiser: Adam\
lr: 0.0004\
weight_decay: 0.0006\
batch_size: 1024\
clip_grad_norm: 5.1\
num_filters: 128\
kernel_size: 5\
**Accuracy:** 42.092%\
**F1 score (macro-averaged):** 0.41553\
**ROC-AUC score:** 0.58021\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 166 | 167 | 133 |
| Real ~ | 129 | 283 | 135 |
| Real + | 152 | 170 | 195 |
<br/>

### Model 1
**Target variable:** Net movement direction of next 4 candles (down/flat/up)\
**Train:** 2010 - 2024\
**Test:** 2025\
**Features:** ["return", "return_4", "log_return", "log_return_4",\
"atr_14", "volatility_regime",\
"bb_width",\
"hl_spread", "oc_spread", "upper_wick",\
"normalised_ema15", "normalised_ema50", "ema_cross",\
"rsi_14", "macd_hist",\
"xgb_1", "xgb_2"]\
**Hyperparameters:**\
hidden_size: 768\
num_layers: 2\
dropout: 0.18\
lookback: 20\
optimiser: Adam\
lr: 0.0005\
weight_decay: 1.5e-5\
batch_size: 256\
clip_grad_norm: 5.1\
**Accuracy:** 40.719%\
**F1 score (macro-averaged):** 0.39607\
**ROC-AUC score:** 0.57066\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 117 | 154 | 195 |
| Real ~ | 122 | 267 | 158 |
| Real + | 110 | 168 | 239 |