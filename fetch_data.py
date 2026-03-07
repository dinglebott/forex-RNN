from custom_modules import datafetcher
from datetime import datetime

# PHASE 1: FETCH HISTORICAL DATA
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"

datafetcher.getDataLoop(datetime(yearNow - 16, 1, 1), datetime(yearNow, 1, 1), instrument, granularity)