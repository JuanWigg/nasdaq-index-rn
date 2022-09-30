from ast import If
from optparse import Option
import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

dataset = util.getDataset()
sorted_dataset = util.getDataset()

print("============== Dataset ================\n")
print(dataset)

plt.plot(sorted_dataset['Close'], marker='s')
plt.xlabel("Date")
plt.ylabel("Algo")

print("=========== Adding Weekday ============\n")
sorted_dataset['weekDayValueSin'] = sorted_dataset.apply(lambda row: util.getWeekDayValueSin(row['Date']), axis=1)
sorted_dataset['weekDayValueCos'] = sorted_dataset.apply(lambda row: util.getWeekDayValueCos(row['Date']), axis=1)
sorted_dataset['weekDay'] = sorted_dataset.apply(lambda row: pd.to_datetime(row['Date']).day_name(), axis=1)
print(sorted_dataset)

print("========== Splitting Dataset ==========\n")
first_70days, last_days = util.splitDataset(sorted_dataset)

print("First portion: \n")
print(first_70days)

print("Last portion: \n")
print(last_days)

predicted_last_days = last_days.copy()
predicted_last_days['Close'] = predicted_last_days['Close'].apply(lambda x: x+random.randint(-50,50))


print("====================")
print(last_days['Close'])
print("====================")
print(predicted_last_days['Close'])
print("====================")

print(r2_score(last_days['Close'], predicted_last_days['Close']))

plt.plot(predicted_last_days['Close'], marker="s")
plt.show()