from fileinput import close
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util

dataset = util.getDataset()
taining_set, validation_set = util.splitDataframe(dataset)

training_close_values = taining_set['Close'].to_numpy()
training_weekDay_sin_values = taining_set['weekDayValueSin'].to_numpy()
training_weekDay_cos_values = taining_set['weekDayValueCos'].to_numpy()

training_inputs  = []
training_outputs = []
expected_output  = []

validation_inputs = []
validation_outputs = []
validation_expected_output  = []

validation_close_values = validation_set['Close'].to_numpy()
validation_weekDay_sin_values = validation_set['weekDayValueSin'].to_numpy()
validation_weekDay_cos_values = validation_set['weekDayValueCos'].to_numpy()

for index in range(0,len(training_close_values)-4):
    expected_output.append(training_close_values[index+4])
    training_inputs.append([training_close_values[index], training_close_values[index+1], training_close_values[index+2],training_close_values[index+3], training_weekDay_sin_values[index+4], training_weekDay_cos_values[index+4]])
    training_outputs.append(expected_output)

for index in range(0,len(validation_close_values)-4):
    validation_expected_output.append(validation_close_values[index+4])
    validation_inputs.append([validation_close_values[index], validation_close_values[index+1], validation_close_values[index+2],validation_close_values[index+3], validation_weekDay_sin_values[index+4], validation_weekDay_cos_values[index+4]])
    validation_outputs.append(validation_expected_output)


print("==================================== INPUTS ===============")
print(training_inputs)
print("===========================================================")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, input_dim=6, activation=tf.keras.activations.tanh))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.tanh))
model.add(tf.keras.layers.Dense(1))


#
#inputs = tf.keras.layers.Input(shape=(4,))
#x = tf.keras.layers.Dense(50, activation="relu")(inputs)
#x = tf.keras.layers.Dense(100, activation="relu")(x)
#modelo = tf.keras.layers.Model(inputs, x)

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=1.0),
    loss='mean_squared_error'
)

print(model.summary())
print("Comenzando entrenamiento...")
historial = model.fit(training_inputs, training_outputs, epochs=10, verbose=1)
print("Modelo entrenado!")

print("Hagamos una predicción!")
print(len(validation_inputs))
print(validation_inputs)

resultados = model.predict(validation_inputs)

print ()
for index in range(0,len(resultados)):
    print(f"Valor obtenido: {resultados[index]} - Valor real: {validation_close_values[index]}")

plt.plot(resultados, marker='s')
plt.xlabel("Date")
plt.ylabel("Cierre")
plt.plot(validation_close_values, marker="s")
plt.show()