# digital_image_processing
DIP Assignments

## Assiggment 1:

### Filters:

We have 4 images of rockets, named rocket1 to rocket 4. We simplt loop through this file and apply all filters with ther respective functions.

```
for i in range(1, 5):
    img_name = f"rocket{i}.jpeg"
    salt_pepper(img_name, 0.08)
    gaussian(img_name)
    poisson(img_name)
    speckle(img_name)
```

The implementation for this can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/test_tubes/Assignment1%20-%20Filters/filters.ipynb)

### Test Tubes

We have 2 sets of images of test tubes. We must succesfully turn the image so that the test tube stands up straight. We must also crop out at least 10 test tubes in each set.

The approach is that we use the ```np.Canny()``` functionb to detect lines in the image. This way we see what the important parts are for each picture. This works pretty well but we still see the edges of the standard, which is holding the test tube. I have written an algorithm that detects gaps. We know the gap between the test tube and the standard is more than 50 pixels. If we measure the gap we gan measure where the test tube begins and where the test tube ends. This work is all done in the function ```def find_edges(img_src):``` which returns a dictionary such as ```{'Upper': 322, 'Lower': 136, 'Left': 134, 'Right': 185}```.

We can then use this dictionary to cut out the image using array slicing: ```return img_gray[boundaries["Lower"]:boundaries["Upper"] + 1, boundaries["Left"]:boundaries["Right"] + 1]```

The implementation can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/test_tubes/Assignment2%20-%20TestTubes/test_tubes.ipynb)

## Assignment 2

### RBCs vs Sickles





