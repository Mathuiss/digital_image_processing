import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model

x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")

model = load_model("rbc_model.h5")
model2 = load_model("rbc_model2.h5")

arr_metrics = model.evaluate(x_val, y_val)
arr_metrics2 = model2.evaluate(x_val, y_val)

# plt.plot(arr_metrics)
print(f"Model 1 val loss: {arr_metrics[0]}, Model 1 val accuracy: {arr_metrics[1]}")
print(f"Model 2 val loss: {arr_metrics2[0]}, Model 2 val accuracy: {arr_metrics2[1]}")
