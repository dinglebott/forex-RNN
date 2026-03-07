# EXPORTS:
# getData() saves a single batch of data and returns the file path
# getDataLoop() saves many batches of data in 1 file
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
import time

# get oanda api key (from .env file)
load_dotenv()
apiKey = os.getenv("API_KEY")

# global variables
headers = {"Authorization": f"Bearer {apiKey}"}
baseUrl = "https://api-fxtrade.oanda.com" # access token generated from live account (don't use fxpractice)

# NOTES AND EXPLANATION
# instrument: EUR_USD, USD_JPY etc
# granularity values: S5/10/15/30, M1/2/4/5/10/15/30, H1/2/3/4/6/8/12, D, W, M
# count: default 500, maximum 5000
# price: mid(M), bid(B), ask(A)
# LOOP variant:
# start, end in datetime format: e.g. datetime(2000, 1, 1) (year, month, day)


# FETCH DATA ONCE (saves locally as JSON file)
def getData(instr="EUR_USD", gran="H1", count=500, subfolder=""):
    # get response
    params = {
        "granularity": gran,
        "count": count,
        "price": "M"
    }
    endpoint = f"/v3/instruments/{instr}/candles"
    response = requests.get(baseUrl + endpoint, headers=headers, params=params)
    
    # inspect response
    if response.status_code != 200:
        raise Exception(response.text)
    
    # save response
    data = response.json()
    directory = f"json_data/{subfolder}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%m%d-%H%M") # extract month date hour minute from datetime
    filename = f"{instr}_{gran}_{count}x_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
    print("Saved to: " + filename)
    
    # expose file path
    return filepath


# FETCH DATA LOOP (saves locally as JSON file)
def getDataLoop(
    start,
    end,
    instr="EUR_USD",
    gran="H1",
    subfolder=""
):
    # helper
    def getOneCandle():
        match gran[0]:
            case "S":
                return timedelta(seconds=int(gran[1:]))
            case "M":
                if len(gran) == 1:
                    return timedelta(days=31)
                else:
                    return timedelta(minutes=int(gran[1:]))
            case "H":
                return timedelta(hours=int(gran[1:]))
            case "D":
                return timedelta(days=1)
            case "W":
                return timedelta(weeks=1)

    # get response
    endpoint = f"/v3/instruments/{instr}/candles"
    allCandles = []
    currentStart = start

    while currentStart < end:
        chunkEnd = min(currentStart + 5000 * getOneCandle(), end) # prevent overshooting
        params = {
            "from": currentStart.isoformat() + "Z", # zero offset from UTC
            "to": chunkEnd.isoformat() + "Z",
            "granularity": gran,
            "price": "M"
        }
        response = requests.get(baseUrl + endpoint, headers=headers, params=params)

        # inspect response
        if response.status_code != 200:
            raise Exception(response.text)
        
        # add response to final list
        data = response.json()
        candles = data["candles"]
        if not candles:
            currentStart += getOneCandle()
            continue
        allCandles += candles

        # move start time forward
        lastCandleTime = candles[-1]["time"] # in isoformat
        currentStart = datetime.strptime(
            lastCandleTime.split(".")[0], # split to remove fractional seconds
            "%Y-%m-%dT%H:%M:%S" # format
        ) + getOneCandle() # avoid duplicating last candle
        print("Downloaded until: " + lastCandleTime)

        # rate limit
        time.sleep(0.5)
    
    # save to file
    directory = f"json_data/{subfolder}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{instr}_{gran}_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as file:
        json.dump({"candles": allCandles}, file, indent=4)
    print("Saved to: " + filename)