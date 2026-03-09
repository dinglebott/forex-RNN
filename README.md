## About project
Goal: Predict the direction of future price movements in forex markets by training a neural network\
This is the next stage of the [trading-trees](https://github.com/dinglebott/trading-trees) project, which uses a tree-based model architecture\
In this project I will integrate the model from [trading-trees](https://github.com/dinglebott/trading-trees) into a larger LSTM architecture\
LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network which performs well in time series like the forex market, due to its "memory" of previous time windows\
*See DOCS.md for details on data collection, tuning, training, and testing methodology*

## Outline of methodology
Phase 1: Fetch historical data => fetch_data.py\
Phase 2: Feature engineering and selection => select_features.py\
Phase 3: Tune hyperparameters (using features from Phase 2) => tune_params.py\
Phase 4: Train final model and evaluate (using features and hyperparameters from Phases 2 and 3) => train_model.py
*See DOCS.md for detailed testing methodology*

## Project structure
The main code is all in the top-level scripts\
The custom_modules folder contains helper functions to fetch and manipulate the data, as well as a fully-trained XGBoost model and a script to produce leak-free XGBoost predictions for the LSTM

## How to build a model
The top-level scripts contain global variables for the current year and the desired instrument and granularity. Set these to the appropriate values\
For a model other than EUR_USD_H4_2026, first train an appropriate XGBoost model using the framework in [trading-trees](https://github.com/dinglebott/trading-trees)\
Then paste the JSON file into the custom_modules folder, and edit the filepath accordingly in the xgboost_trainer.py module\
In Phase 4, double-check the final features and hyperparameters in their respective JSON files (features.json and hyperparameters.json)\
*Rename them with their version names once the model is trained to prevent future models overwriting them*\
Output for all phases is printed to the terminal, and the final model is automatically saved as a JSON file
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already)\
If you have a key, set it as an environment variable API_KEY in a local .env file\
The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com

## How to use a model
Run use_model.py\
Prediction and confidence are printed to the terminal\
Remember to set the correct global variables\
Also set the version number of the model you want (in the version variable)\
Obviously make sure you have the correct model trained for your use case first

## Why?????
why am i still trying