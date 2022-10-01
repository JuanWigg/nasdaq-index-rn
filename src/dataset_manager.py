from datetime import datetime
import imp
import numpy as np 
import pandas as pd
from pandas import DataFrame
import os
from trainset import Trainset

def getWeekDayValueSin(index):
    date_object = pd.to_datetime(index)
    return np.sin(date_object.isoweekday() * (2* np.pi / 5))

def getWeekDayValueCos(index):
    date_object = pd.to_datetime(index)
    return np.cos(date_object.isoweekday() * (2* np.pi / 5))

def getDataset(trainset: Trainset):
    dataset = getFile()
    filtered_dataset = filterDataset(dataset, trainset)
    sorted_dataset = invertDataset(filtered_dataset)
    cleaned_sorted_dataset = cleanDataset(sorted_dataset)
    cleaned_sorted_dataset['weekDayValueSin'] = cleaned_sorted_dataset.apply(lambda row: getWeekDayValueSin(row['Date']), axis=1)
    cleaned_sorted_dataset['weekDayValueCos'] = cleaned_sorted_dataset.apply(lambda row: getWeekDayValueCos(row['Date']), axis=1)
    return splitDataset(cleaned_sorted_dataset, trainset)

def getFile():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/nasdaq_index.csv')
    dataset = pd.read_csv(filename, parse_dates=[0])
    return dataset

def filterDataset(dataset: DataFrame, trainset: Trainset):
    initDate = trainset.fecha_inicio_dataset
    endDate = trainset.fecha_fin_dataset
    days_before_first_day = dataset.query(f"Date < '{initDate}'" ).head(trainset.dias_hacia_atras)
    filtered_dataset = dataset.query(f"Date >= '{initDate}' and Date <= '{endDate}'" )
    return filtered_dataset.append(days_before_first_day)

def invertDataset(filtered_dataset):
    sorted_dataset = filtered_dataset[::-1]
    sorted_dataset = sorted_dataset.reset_index()
    return sorted_dataset

def cleanDataset(sorted_dataset):
    return sorted_dataset.drop(columns=['Volume', 'Open', 'High', 'Low'])

def splitDataset(dataframe: DataFrame,  trainset: Trainset):

    totalData = len(dataframe) - trainset.dias_hacia_atras
    total_training_days = round(totalData * trainset.porcentaje_entrenamiento)
    total_validation_days = round(totalData * trainset.porcentaje_validacion)
    upper_limit_training_days = total_training_days + trainset.dias_hacia_atras
    lower_limit_validation_days = upper_limit_training_days - trainset.dias_hacia_atras 
    upper_limit_validation_days = upper_limit_training_days + total_validation_days
    lower_limit_training_days = upper_limit_validation_days - trainset.dias_hacia_atras

    training_days = dataframe.iloc[:upper_limit_training_days, :]
    validation_days = dataframe.iloc[lower_limit_validation_days:upper_limit_validation_days, :]
    test_days = dataframe.iloc[lower_limit_training_days:, :]
    return training_days, validation_days, test_days

def generateTrainingIOs(training_days: DataFrame, dias_hacia_atras: int):
    training_close_values = training_days['Close'].to_numpy()
    training_weekDay_sin_values = training_days['weekDayValueSin'].to_numpy()
    training_weekDay_cos_values = training_days['weekDayValueCos'].to_numpy()
    training_inputs             = []
    training_outputs            = []

    for i in range(dias_hacia_atras, len(training_days)):
        training_input = []
        for j in range(dias_hacia_atras, 0, -1):  
            training_input.append(training_close_values[i-j])

        training_input.append(training_weekDay_sin_values[i])
        training_input.append(training_weekDay_cos_values[i])
        training_inputs.append(training_input)
        training_outputs.append([training_close_values[i] - training_input[3]])

    return training_inputs, training_outputs
