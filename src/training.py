from fileinput import close
import tensorflow as tf
import numpy as np
import util

dataset = util.getDataset()
taining_set, validation_set = util.splitDataframe(dataset)

training_close_values = taining_set['Close'].to_numpy()
training_weekDay_sin_values = taining_set['weekDayValueSin'].to_numpy()
training_weekDay_cos_values = taining_set['weekDayValueCos'].to_numpy()

training_inputs  = []
training_outputs = []
expected_output  = []



for index in range(0,len(training_close_values)-4):
    expected_output.append(training_close_values[index+4])


    training_inputs.append([training_close_values[index], training_close_values[index+1], training_close_values[index+2],training_close_values[index+3], training_weekDay_sin_values[index+4], training_weekDay_cos_values[index+4]])
    training_outputs.append(expected_output)






model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, input_dim=6, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.relu))


#
#inputs = tf.keras.layers.Input(shape=(4,))
#x = tf.keras.layers.Dense(50, activation="relu")(inputs)
#x = tf.keras.layers.Dense(100, activation="relu")(x)
#modelo = tf.keras.layers.Model(inputs, x)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print(model.summary())
print("Comenzando entrenamiento...")
historial = model.fit(training_inputs, training_outputs, epochs=500, verbose=1)
print("Modelo entrenado!")

print("Hagamos una predicción!")
resultado = model.predict([[4453.79, 4432.92, 4455.9, 4468.98, -0.433884, -0.900969]])
print("El resultado es " + str(resultado) + ". El real es 4531.85")