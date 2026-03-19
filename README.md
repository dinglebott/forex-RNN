## About project
Goal: Predict the direction of future price movements in forex markets by training a neural network\
This is part 2 of the whole forex prediction project, where I experiment with neural network architectures such as CNN and LSTM. See below for the rest of the project.\
CNN (Convolutional Neural Network) is a model that uses kernels to filter out local patterns in the data (~ 3 candles), and abstracts it into inputs for the LSTM to train on. It performs well in pattern recognition and spatial data.\
LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network which performs well in sequential data like time series, due to its "memory" of previous time windows.\
*See DOCS.md for detailed results and workflow*\
<br/>
Part 1: [trading-trees](https://github.com/dinglebott/trading-trees), using a tree-based architecture (XGBoost)\
Part 3: [money-meta](https://github.com/dinglebott/money-meta), ensembling models from parts 1 and 2

## Outline of methodology
Phase 1: Fetch historical data in `fetch_data.py`\
Phase 2: Feature engineering and selection in `select_features.py`\
Phase 3: Tune hyperparameters (using features from Phase 2) in `tune_params.py`\
Phase 4: Train final model and evaluate (using features and hyperparameters from Phases 2 and 3) in `train_model.py`\
*See DOCS.md for detailed testing methodology*

## Project structure
The main code is all in the top-level scripts.\
The `custom_modules` folder contains helper functions to fetch and manipulate the data.

## How to build a model
The top-level scripts contain global variables for the current year, the desired instrument, and granularity. I built my model for 2026 EUR/USD at H4 granularity. For other options, set these to the appropriate values.\
Once the model is trained, rewrite `features.json` in the `results` folder to match the format you see (sorry about that). Also, rename `features.json` and `hyperparameters.json` with the version number like you see too.\
Output for all phases is printed to the terminal, and the final model is automatically saved as a JSON file.
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already).\
If you have a key, set it as an environment variable `API_KEY` in a local `.env` file.\
The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com.

## How to use a model
Run `use_model.py`, the prediction and confidence will be printed to the terminal.\
Remember to set the correct global variables. Obviously make sure you have the correct model trained for your use case first.

## Why?????
why am i still trying this is the hardest shit i've ever done