from fileinput import close
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util
from trainset import Trainset
from sklearn.metrics import r2_score

def toArrayOfArrays(value):
    return [value]

training_days, validation_days, test_days = util.getDataset(Trainset())

print("--------------------------------TRAINING DAYS--------------------------------")
print(training_days)
print("--------------------------------VALIDATION DAYS--------------------------------")
print(validation_days)
print("--------------------------------TEST DAYS--------------------------------")
print(test_days)

# training_set, validation_set, prediction_set = util.splitDataframe(dataset)

# training_close_values = training_set['Close'].to_numpy()
# training_weekDay_sin_values = training_set['weekDayValueSin'].to_numpy()
# training_weekDay_cos_values = training_set['weekDayValueCos'].to_numpy()

# validation_close_values = validation_set['Close'].to_numpy()
# validation_weekDay_sin_values = validation_set['weekDayValueSin'].to_numpy()
# validation_weekDay_cos_values = validation_set['weekDayValueCos'].to_numpy()

# prediction_close_values = prediction_set['Close'].to_numpy()
# prediction_weekDay_sin_values = prediction_set['weekDayValueSin'].to_numpy()
# prediction_weekDay_cos_values = prediction_set['weekDayValueCos'].to_numpy()

# training_inputs             = []
# training_outputs            = []
# validation_inputs           = []
# validation_outputs          = []
# prediction_inputs           = []
# prediction_outputs          = []



# for index in range(0,len(training_close_values)-4):
#     training_inputs.append([training_close_values[index], training_close_values[index+1 ], training_close_values[index+2],training_close_values[index+3], training_weekDay_sin_values[index+4], training_weekDay_cos_values[index+4]])
#     training_outputs.append([training_close_values[index+4]])

# for index in range(0,len(validation_close_values)-4):
#     validation_inputs.append([validation_close_values[index], validation_close_values[index+1], validation_close_values[index+2],validation_close_values[index+3], validation_weekDay_sin_values[index+4], validation_weekDay_cos_values[index+4]])
#     validation_outputs.append([validation_close_values[index+4]])

# for index in range(0,len(prediction_close_values)-4):
#     prediction_inputs.append([prediction_close_values[index], prediction_close_values[index+1], prediction_close_values[index+2],prediction_close_values[index+3], prediction_weekDay_sin_values[index+4], prediction_weekDay_cos_values[index+4]])
#     prediction_outputs.append([prediction_close_values[index+4]])

# ##training_outputs = training_outputs[:10]

# ## NUEVA SECCION

# # totalDataset = util.getTotalDataframe(dataset)

# # result_inputs           = []
# # result_outputs          = []

# # result_close_values = totalDataset['Close'].to_numpy()
# # result_weekDay_sin_values = totalDataset['weekDayValueSin'].to_numpy()
# # result_weekDay_cos_values = totalDataset['weekDayValueCos'].to_numpy()

# # for index in range(0,len(result_close_values)-4):
# #     result_inputs.append([result_close_values[index], result_close_values[index+1 ], result_close_values[index+2],result_close_values[index+3], result_weekDay_sin_values[index+4], result_weekDay_cos_values[index+4]])
# #     result_outputs.append([result_close_values[index+4]])



# print("==================================== INPUTS ===============")
# print(training_inputs)
# print("===========================================================")

# print("================================= Training Outputs ========")
# print(training_outputs)
# print("===========================================================")



# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(50, input_dim=6, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(50, activation=tf.keras.activations.relu))

# # model.add(tf.keras.layers.Dense(50, activation=tf.keras.activations.relu))
# model.add(tf.keras.layers.Dense(1))


# #
# #inputs = tf.keras.layers.Input(shape=(4,))
# #x = tf.keras.layers.Dense(50, activation="relu")(inputs)
# #x = tf.keras.layers.Dense(100, activation="relu")(x)
# #modelo = tf.keras.layers.Model(inputs, x)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
#     loss='mean_squared_error',
#     metrics=['mae']
# )

# print(model.summary())
# print("Comenzando entrenamiento...")
# historial = model.fit(training_inputs, training_outputs, epochs=3000 ,verbose=1, validation_data=(validation_inputs, validation_outputs))
# print("Modelo entrenado!")

# model.save('model.h5', historial)

# print("Hagamos una predicción!")
# print(len(validation_inputs))
# print(validation_inputs)

# #
# #print("Hagamos una predicción!")
# #
# ## print(len(validation_inputs))
# ## print(validation_inputs)
# #
# #print(model.predict([[4140.38, 4181.35, 4148.43, 4188.59, 0.5877852522924732, -0.8090169943749473]]))
# #
# #
# #print("EL REAL ERA 4229.15")
# #
# #
# #print("Hagamos otra predicción!")
# #print(model.predict([[4488.15, 4414.25, 4479.06, 4482.61, 0.5877852522924732, -0.8090169943749473]]))
# #
# #print("EL REAL ERA 4426.82")
# resultados = model.predict(prediction_inputs)

# #print ()
# #for index in range(0,len(resultados)):
# #    print(f"Valor obtenido: {resultados[index]} - Valor real: {validation_close_values[index]}")
# #


# flat_results = [item for sublist in resultados for item in sublist]
# flat_results2 = flat_results.copy()
# flat_results2.append(flat_results[-1])
# flat_results2.pop(0)

# print(f'R2: {r2_score(prediction_outputs, flat_results2)}')
# plt.plot(flat_results2, marker='o', color='green')
# plt.xlabel("Fecha")
# plt.ylabel("Valor de cierre")
# plt.plot(prediction_outputs, marker="s", color='red')
# plt.show()



