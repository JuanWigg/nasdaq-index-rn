from trainset import Trainset
import dataset_manager
from fileinput import close
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from datetime import datetime

myTrainset = Trainset()

myTrainset.fecha_inicio_dataset= "2013-04-02"
myTrainset.fecha_fin_dataset= "2015-06-18"
myTrainset.dias_hacia_atras = 4
myTrainset.porcentaje_entrenamiento = 0.7
myTrainset.porcentaje_validacion = 0.15
myTrainset.porcentaje_test = 0.15

training_days, validation_days, test_days = dataset_manager.getDataset(myTrainset)

training_inputs, training_outputs = dataset_manager.generateTrainingIOs(training_days, myTrainset.dias_hacia_atras)
validation_inputs, validation_outputs = dataset_manager.generateTrainingIOs(validation_days, myTrainset.dias_hacia_atras)
test_inputs, test_outputs = dataset_manager.generateTrainingIOs(test_days, myTrainset.dias_hacia_atras)

print("Cargando modelo...")
model = tf.keras.models.load_model("models/model-paper.h5")

predictions = [item for sublist in model.predict(test_inputs) for item in sublist]
results_df = pd.DataFrame(columns=['Date', 'Prediction', 'RealValue'])
results_df["Date"] = test_days["Date"].iloc[myTrainset.dias_hacia_atras:].apply(lambda row: (pd.to_datetime(row)).strftime("%Y-%m-%d"))
results_df["RealValue"] = test_days["Close"].iloc[myTrainset.dias_hacia_atras:]
results_df["Prediction"] = predictions
results_df.reset_index()


plt.plot(results_df["Prediction"], marker='o', color='green')
plt.xlabel("Fecha")
plt.ylabel("Valor de cierre")
plt.plot(results_df["RealValue"], marker="s", color='red')
plt.xticks(ticks=range(results_df.index[0], results_df.index[-1]+1), labels=results_df["Date"], rotation=90)
plt.grid()
print(f'R2: {r2_score(results_df["RealValue"], results_df["Prediction"])}')

plt.show()