# %% Import Libraries
import pandas as pd
import requests
import json

# %%

url = "https://intraday.worldtradingdata.com/api/v1/intraday"
params = {
    "symbol": "SNAP",
    "interval": "1",
    "range": "2",
    "api_token": "demo",
}
response = requests.request("GET", url, params=params)
response.json()
