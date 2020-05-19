# Digital Image Processing
DIP Assignments
- Filters
- Test Tubes
- RBCs vs Sickles

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

The implementation for this can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/test_tubes/Assignment1%20-%20Filters/filters.ipynb).

### Test Tubes

We have 2 sets of images of test tubes. We must turn the image so that the test tube stands up straight. We must also succesfully crop out at least 10 test tubes in each set.

The approach is that we use the ```np.Canny()``` function to detect lines in the image. This way we see what the important parts are for each picture. This works pretty well but we still see the edges of the standard, which is holding the test tube.

I have written an algorithm that detects gaps. We know the gap between the test tube and the standard is more than 50 pixels. If we measure the gap we gan measure where the test tube begins and where the test tube ends.

This work is all done in the function ```def find_edges(img_src):``` which returns a dictionary such as ```{'Upper': 322, 'Lower': 136, 'Left': 134, 'Right': 185}```.

We can then use this dictionary to cut out the image using array slicing: ```return img_gray[boundaries["Lower"]:boundaries["Upper"] + 1, boundaries["Left"]:boundaries["Right"] + 1]```

The implementation can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/test_tubes/Assignment2%20-%20TestTubes/test_tubes.ipynb).

## Assignment 2

### RBCs vs Sickles

To run assignment 2 you must execute the following commands:
```
cd rbcs_vs_sickles
python preprocess.py
python train.py
python evaluate.py
python predict.py
```

#### Preprocessing:

In this assignment we have a data set consisting of simages of healthy and sick red blood cells. These images are in their respective folders: ```healthy/``` and ```sick/```. In order to train an AI with keras we must preprocess the data so that we have a tupe of images and labels like so: ```(x_train, y_train), (x_test, y_test) = preprocess()```.

During the preprocessing we are going to to do the following things:
1. We load all healthy and sick images in a dictionary
2. We use the Canny edge detection algorithm to detect the cells in the image
3. We create a big pile with all the images and their assigned labels
4. We shuffle the pile so that there is a random order of healthy and sick images and labels
5. We create a new pile for the images and a new pile for the labels
6. We place the images on the image pile and the labels on the labels pile
7. We slice both piles with a ratio of 80/20 to create x_train, y_train, x_val and y_val
8. We reshape the numpy array so that the input layer of the neural network can process the images
9. We save the data set to the hard drive

This procedure is all fairly straight forward. The images are loaded using ```cv2.imread(img_src, cv2.IMREAD_REDUCED_GRAYSCALE_8)```, to filter out the background noise. The images are then resized to ```cv2.resize(arr_img, (255, 255))```, in order to create a lighter load on the network. Furthermore we use ```cv2.Canny(arr_img, 30, 80)```, to detect the cells in the image. Lastly we normalize the image so that the network can handle this array with ```arr_img = arr_img / 255```.

After we shuffle the pile, we load the images onto 2 new piles, named: ```x_pile``` and ```y_pile```. This work is done like so:
```
for i in pile:
        x_pile = np.append(x_pile, [i[0]], axis=0)
        y_pile = np.append(y_pile, i[1])
```

Throughout the preprocessing script we use a constant to tell the program how much validation data we want. This is done like so:
```TRAIN_VAL_FACTOR = 0.2```

With the above instructions completed we can continue to divide the data in training and test data, using array slicing:
```
 # # Training validation data and labels
    x_train = x_pile[:int((1 - TRAIN_VAL_FACTOR) * len(x_pile))]  # Images
    y_train = y_pile[:int((1 - TRAIN_VAL_FACTOR) * len(y_pile))]  # Labels

    # Validation data and labels
    x_val = x_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Images
    y_val = y_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Labels
```

Finnaly we reshape using the ```.reshape((-1, 255, 255, 1))``` command on both ```x_train``` and ```x_val```

The implementation can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/rbcs_vs_sickles/preprocess.py).

#### Training:


