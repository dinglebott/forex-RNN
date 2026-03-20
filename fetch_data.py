from custom_modules import datafetcher
from datetime import datetime
import json

# PHASE 1: FETCH HISTORICAL DATA
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, _, _ = globalVars.values()

datafetcher.getDataLoop(datetime(yearNow - 21, 1, 1), datetime(yearNow, 1, 1), instrument, granularity)