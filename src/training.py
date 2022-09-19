from fileinput import close
from turtle import shape
import tensorflow as tf
import numpy as np
import util

dataset = util.getDataset()
taining_set, validation_set = util.splitDataframe(dataset)

training_close_values = taining_set['Close'].to_numpy()

training_inputs= []
training_outputs= []

for index in range(0,len(training_close_values)-4):
    values = [training_close_values[index], training_close_values[index+1],training_close_values[index+2],training_close_values[index+3]]
    expected_output = training_close_values[index+4]

    training_inputs.append(values)
    training_outputs.append(expected_output)

inputs = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Dense(50, activation="relu")(inputs)
x = tf.keras.layers.Dense(100, activation="relu")(x)
modelo = tf.keras.layers.Model(inputs, x)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(training_inputs, training_outputs, epochs=100, verbose=True)
print("Modelo entrenado!")

print("Hagamos una predicción!")
resultado = modelo.predict([4453.79, 4432.92, 4455.9, 4468.98])
print("El resultado es " + str(resultado) + ". El real es 4531.85")