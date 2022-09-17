import nasdaqdatalink as ndl
import logging
import os

logging.basicConfig()
ndl_logs = logging.getLogger("nasdaqdatalink")
ndl_logs.setLevel(logging.DEBUG)

ndl.ApiConfig.api_key = os.getenv('NASDAQ_API_KEY')

def getDataset(index, start_date, end_date):
    data = ndl.get(index, start_date=start_date, end_date=end_date, collapse="daily")
    return data
