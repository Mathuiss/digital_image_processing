import numpy as np
import cv2
import os
from keras.models import load_model
from keras.models import Sequential


def preprocess(img_src):
    arr_img = cv2.imread(img_src, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    arr_img = cv2.resize(arr_img, (255, 255))
    arr_img = cv2.Canny(arr_img, 30, 80)
    return arr_img / 255


model = load_model("rbc_model.h5")
arr_img_src = os.listdir("predict")

for img_src in arr_img_src:
    print(f"Predicting: {img_src}")
    arr_img = preprocess(f"predict/{img_src}")
    # arr_img = np.array([arr_img])
    arr_img = arr_img.reshape((-1, 255, 255, 1))
    y = model.predict(arr_img)
    print(y)
