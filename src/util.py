from datetime import datetime
import numpy as np 
import pandas as pd
import os

def getWeekDayValueSin(index):
    date_object = pd.to_datetime(index)
    return np.sin(date_object.isoweekday() * (2* np.pi / 5))

def getWeekDayValueCos(index):
    date_object = pd.to_datetime(index)
    return np.cos(date_object.isoweekday() * (2* np.pi / 5))

def splitDataframe(dataframe):
    first_70days = dataframe.iloc[:70, :]
    last_days = dataframe.iloc[70:, :]
    return first_70days, last_days

def getDataset():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/nasdaq_index.csv')

    dataset = pd.read_csv(filename, parse_dates=[0])
    dataset = dataset.rename(columns={"Close/Last": "Close"})
    filtered_dataset = dataset.query("Date >= '2015-01-28' and Date <= '2015-06-18'" )
    sorted_dataset = filtered_dataset[::-1]
    sorted_dataset = sorted_dataset.reset_index()
    sorted_dataset = sorted_dataset.drop(columns=['Volume'])
    sorted_dataset['Open'] = np.where(sorted_dataset['Open']==0, sorted_dataset['Close'], sorted_dataset['Open'])
    sorted_dataset['High'] = np.where(sorted_dataset['High']==0, sorted_dataset['Close'], sorted_dataset['High'])
    sorted_dataset['Low'] = np.where(sorted_dataset['Low']==0, sorted_dataset['Close'], sorted_dataset['Low'])
    sorted_dataset['weekDayValueSin'] = sorted_dataset.apply(lambda row: getWeekDayValueSin(row['Date']), axis=1)
    sorted_dataset['weekDayValueCos'] = sorted_dataset.apply(lambda row: getWeekDayValueCos(row['Date']), axis=1)

    return sorted_dataset