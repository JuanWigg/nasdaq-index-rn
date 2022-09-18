from datetime import datetime
import numpy as np 
import pandas as pd

def getWeekDayValueSin(index):
    date_object = pd.to_datetime(index)
    return np.sin(date_object.isoweekday() * (2* np.pi / 7))

def getWeekDayValueCos(index):
    date_object = pd.to_datetime(index)
    return np.cos(date_object.isoweekday() * (2* np.pi / 7))

def splitDataframe(dataframe):
    first_70days = dataframe.iloc[:70, :]
    last_days = dataframe.iloc[70:, :]
    return first_70days, last_days
