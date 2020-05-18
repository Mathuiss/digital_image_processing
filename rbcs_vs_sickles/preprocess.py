import cv2
import os
import random
import numpy as np

# Declaring constants
TRAIN_VAL_FACTOR = 0.2


def load_img_src(folder):
    arr_img_src = []

    for item in os.listdir(folder):
        if item.endswith(".jpg"):
            arr_img_src.append(f"{folder}/{item}")

    return arr_img_src


def box_cell(arr_img):
    # plt.imshow(arr_img, cmap="gray")
    # plt.show()
    return cv2.Canny(arr_img, 30, 80)
    # plt.imshow(arr_canny, cmap="gray")
    # plt.show()


def preprocess(dict_img_src):
    pile = []

    # We create a big pile of dicts with x and y
    for key in ["Healthy", "Sick"]:
        for img_src in dict_img_src[key]:
            print(f"Preprocessing {img_src}")

            if key == "Healthy":
                label = int(1)
            else:
                label = int(0)

            arr_img = cv2.imread(img_src, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            arr_img = cv2.resize(arr_img, (255, 255))
            arr_img = box_cell(arr_img)
            arr_img = arr_img / 255
            pile.append(np.array([arr_img, label]))

    # We shuffle the pile to create a random order of healthy and sick cells
    random.shuffle(pile)

    # We create and empty np array for x (data) and y (label)
    # x_pile = np.empty((len(pile), 255, 255))
    x_pile = np.empty((0, 255, 255))
    y_pile = np.array([])

    # The pile is an array of dicts
    # We need to unpack this and place in the 3d np array for x and y
    print("Formatting data to tensor matrix...")
    for i in pile:
        x_pile = np.append(x_pile, [i[0]], axis=0)
        y_pile = np.append(y_pile, i[1])

    # Now that we seperated the x and y, we create the training and validation sets
    print("Creating train and test batches...")

    # # Training validation data and labels
    x_train = x_pile[:int((1 - TRAIN_VAL_FACTOR) * len(x_pile))]  # Images
    y_train = y_pile[:int((1 - TRAIN_VAL_FACTOR) * len(y_pile))]  # Labels

    # Validation data and labels
    x_val = x_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Images
    y_val = y_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Labels

    print("Done")

    return ((x_train, x_val), (y_train, y_val))


# Prepocessing all data into usable chunks for CNN
dict_img_src = {"Healthy": load_img_src("healthy"), "Sick": load_img_src("sick")}
(x_train, x_val), (y_train, y_val) = preprocess(dict_img_src)

# Reshaping to (length_data_set, width_img, height_img, channels)
x_train = x_train.reshape((-1, 255, 255, 1))
x_val = x_val.reshape((-1, 255, 255, 1))

np.save("x_train", x_train)
np.save("x_val", x_val)
np.save("y_train", y_train)
np.save("y_val", y_val)


print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_val: {x_val.shape}")
print(f"y_val: {y_val.shape}")
