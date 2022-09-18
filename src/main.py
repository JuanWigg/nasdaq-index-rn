import nasdaq as nq 
import pandas as pd
import util

# From https://data.nasdaq.com/
dataset = nq.getDataset('LBMA/SILVER', start_date='2022-01-31', end_date='2022-06-06')

print("============== Dataset ================\n")
print(dataset)

print("=========== Adding Weekday ============\n")
dataset['weekDayValue'] = dataset.apply(lambda row: util.getWeekDayValue(row.name), axis=1)
dataset['weekDay'] = dataset.apply(lambda row: pd.to_datetime(row.name).day_name(), axis=1)
print(dataset)

print("========== Splitting dataset ==========\n")
first_70days, last_days = util.splitDataframe(dataset)

print("First portion: \n")
print(first_70days)

print("Last portion: \n")
print(last_days)

