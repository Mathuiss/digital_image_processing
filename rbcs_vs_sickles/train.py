
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")

# Defining CNN model
model = Sequential()

model.add(Conv2D(32, kernel_size=(25, 25), padding="same", activation="relu", input_shape=(255, 255, 1)))
model.add(MaxPooling2D(pool_size=(25, 25)))

model.add(Conv2D(64, kernel_size=(20, 20), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(10, 10)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(1500, activation="relu"))
model.add(Dense(750, activation="relu"))
model.add(Dense(350, activation="relu"))
model.add(Dropout(0.5))
# 1 for firing or not firing, sigmoid because
model.add(Dense(1, activation="sigmoid"))
# binary classification


# Compiling the model using binary_crossentropy and the adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model using training data x and saving the metrics to the arr_metrics variable
arr_metrics = model.fit(x_train, y_train, batch_size=25, epochs=25, validation_data=(x_val, y_val))

# Now we evaluate the model using the validation data y
model.evaluate(x_val, y_val)

# Saving the model in project root directory
model.save("rbc_model3.h5")

print(arr_metrics.history.keys())

# Showing metrics saved in arr_metrics
plt.plot(arr_metrics.history["accuracy"])
plt.plot(arr_metrics.history["val_accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"])
plt.show()

plt.plot(arr_metrics.history["loss"])
plt.plot(arr_metrics.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend("Traing", "Val")
plt.show()
