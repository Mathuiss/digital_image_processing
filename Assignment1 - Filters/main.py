import math
import numpy as np
import cv2


def poisson(src, dest):
    src = cv2.imread(src)
    dest = cv2.imread(dest)

    src_mask = np.zeros(src.shape, src.dtype)
    center = (math.floor(len(dest[0]) / 2), math.floor(len(dest) / 2))
    print(center)
    return cv2.seamlessClone(src, dest, src_mask, center, cv2.NORMAL_CLONE)


for i in range(1, 5):
    target_a = f"rocket{i}.jpeg"
    target_b = f"rocket{5 - i}.jpeg"
    output_location = f"poisson/{target_a}+{target_b}"
    print(output_location)
    cv2.imwrite(output_location, poisson(target_a, target_b))

print("done")
