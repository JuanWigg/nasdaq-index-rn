import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/nasdaq_index.csv', parse_dates=[0])
dataset = dataset.rename(columns={"Close/Last": "Close"})
filtered_dataset = dataset.query("Date >= '2015-01-28' and Date <= '2015-06-18'" )
sorted_dataset = filtered_dataset[::-1]
sorted_dataset = sorted_dataset.reset_index()
sorted_dataset = sorted_dataset.drop(columns=['Volume'])
sorted_dataset['Open'] = np.where(sorted_dataset['Open']==0, sorted_dataset['Close'], sorted_dataset['Open'])
sorted_dataset['High'] = np.where(sorted_dataset['High']==0, sorted_dataset['Close'], sorted_dataset['High'])
sorted_dataset['Low'] = np.where(sorted_dataset['Low']==0, sorted_dataset['Close'], sorted_dataset['Low'])
test_dataset = sorted_dataset.query("Open == 0")



print("============== Dataset ================\n")
print(dataset)

print("============== Test Dataset ================\n")
print(test_dataset)


plt.plot(sorted_dataset['Close'], marker='s')
plt.xlabel("Date")
plt.ylabel("Algo")
plt.show()


print("=========== Adding Weekday ============\n")
sorted_dataset['weekDayValueSin'] = sorted_dataset.apply(lambda row: util.getWeekDayValueSin(row['Date']), axis=1)
sorted_dataset['weekDayValueCos'] = sorted_dataset.apply(lambda row: util.getWeekDayValueCos(row['Date']), axis=1)
sorted_dataset['weekDay'] = sorted_dataset.apply(lambda row: pd.to_datetime(row['Date']).day_name(), axis=1)
print(sorted_dataset)

print("========== Splitting Dataset ==========\n")
first_70days, last_days = util.splitDataframe(sorted_dataset)

print("First portion: \n")
print(first_70days)

print("Last portion: \n")
print(last_days)

