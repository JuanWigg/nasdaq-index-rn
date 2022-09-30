from keras.models import load_model
from fileinput import close
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.metrics import r2_score

def toArrayOfArrays(value):
    return [value]


dataset = util.getDataset()
training_set, validation_set, prediction_set = util.splitDataset(dataset)

training_close_values = training_set['Close'].to_numpy()
training_weekDay_sin_values = training_set['weekDayValueSin'].to_numpy()
training_weekDay_cos_values = training_set['weekDayValueCos'].to_numpy()

validation_close_values = validation_set['Close'].to_numpy()
validation_weekDay_sin_values = validation_set['weekDayValueSin'].to_numpy()
validation_weekDay_cos_values = validation_set['weekDayValueCos'].to_numpy()

prediction_close_values = prediction_set['Close'].to_numpy()
prediction_weekDay_sin_values = prediction_set['weekDayValueSin'].to_numpy()
prediction_weekDay_cos_values = prediction_set['weekDayValueCos'].to_numpy()

validation_inputs           = []
validation_outputs          = []
prediction_inputs           = []
prediction_outputs          = []

for index in range(0,len(validation_close_values)-4):
    validation_inputs.append([validation_close_values[index], validation_close_values[index+1], validation_close_values[index+2],validation_close_values[index+3], validation_weekDay_sin_values[index+4], validation_weekDay_cos_values[index+4]])
    validation_outputs.append([validation_close_values[index+4]])

for index in range(0,len(prediction_close_values)-4):
    prediction_inputs.append([prediction_close_values[index], prediction_close_values[index+1], prediction_close_values[index+2],prediction_close_values[index+3], prediction_weekDay_sin_values[index+4], prediction_weekDay_cos_values[index+4]])
    prediction_outputs.append([prediction_close_values[index+4]])

print("Cargando modelo...")
model = load_model('model.h5')

resultados = model.predict(prediction_inputs)

flat_results = [item for sublist in resultados for item in sublist]
flat_results2 = flat_results.copy()
flat_results2.append(flat_results[-1])
flat_results2.pop(0)

print(f'R2: {r2_score(prediction_outputs, flat_results2)}')
plt.plot(flat_results2, marker='o', color='green')
plt.xlabel("Fecha")
plt.ylabel("Valor de cierre")
plt.plot(prediction_outputs, marker="s", color='red')
plt.show()