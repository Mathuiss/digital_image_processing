# Digital Image Processing
DIP Assignments
- Filters
- Test Tubes
- RBCs vs Sickles

## Assiggment 1:

### Filters:

We have 4 images of rockets, named rocket1 to rocket 4. We simplt loop through this file and apply all filters with ther respective functions.

```python
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
```bash
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
```python
for i in pile:
        x_pile = np.append(x_pile, [i[0]], axis=0)
        y_pile = np.append(y_pile, i[1])
```

Throughout the preprocessing script we use a constant to tell the program how much validation data we want. This is done like so:
```TRAIN_VAL_FACTOR = 0.2```

With the above instructions completed we can continue to divide the data in training and test data, using array slicing:
```python
    # Training validation data and labels
    x_train = x_pile[:int((1 - TRAIN_VAL_FACTOR) * len(x_pile))]  # Images
    y_train = y_pile[:int((1 - TRAIN_VAL_FACTOR) * len(y_pile))]  # Labels

    # Validation data and labels
    x_val = x_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Images
    y_val = y_pile[int((1 - TRAIN_VAL_FACTOR) * len(y_pile)):]  # Labels
```

Finnaly we reshape using the ```.reshape((-1, 255, 255, 1))``` command on both ```x_train``` and ```x_val```

The implementation can be found [here](https://github.com/Mathuiss/digital_image_processing/blob/master/rbcs_vs_sickles/preprocess.py).

#### Training:

In the ```train.py``` script we are going to take the following steps:
1. We load the training data, training labels, test data and test labels
2. We are going to build a sequential model
3. We are going to compile the model
4. We are going to train the model
5. We are going to evaluate the model
6. We are going to plot the results and metrics

Because we saved the data set to the hard drive in the ```preprocess.py``` script we can now load the data set with:
```python
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")
```

We initialize the model as: ```model = Sequential()```

We are going to build a model with a 2d convolutional input layer. We use 32 filters and a kernel size of ```25 x 25```. We do this because the cells in the image are about this size. We hope the convolutional layer will pick up on the change between healthy and sick cells in the image. The padding we use is ```padding="same"```, because we want the entire picture in the convolutional layer. We use ```activation="relu"```, because this is a good activation function if you want to capture the most important features. The input shape of the image is ```input_shape=(255, 255, 1)```. The image is ```255 x 255``` times ```1``` channel.
```python
model.add(Conv2D(32, kernel_size=(25, 25), padding="same", activation="relu", input_shape=(255, 255, 1)))
```

After this we will use a pooling layer with a pool size of ```25 x 25```.
```python
model.add(MaxPooling2D(pool_size=(25, 25)))
```

We continue by simply adding the larges possible convolutional and pooling layer to it.
```python
model.add(Conv2D(64, kernel_size=(10, 10), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(10, 10)))
```

After we have done convolution we want to let some of the data go, in order to avoid overfitting. We also want to flatten the output from the convolutional layers, because nex we are going to apply fully connected layers.
```python
model.add(Dropout(0.2))
model.add(Flatten())
```

We start with a fully connected layer with ```1500``` outputs. We gradually work our way down to ```350``` inputs to reduce the models complexity. We use ```activation="rely"``` because wewant to activate the neurons where the dotproduct of ```i``` and ```w``` is positive. Finally we add a dropout layer with a dropout factor of ```0.5```. This is a large dropout, but I read it is recommended for convolutional neural networks. I tried ```0.3``` and ```0.4``` as well but I got the best results with ```0.5```.
```python
model.add(Dense(1500, activation="relu"))
model.add(Dense(750, activation="relu"))
model.add(Dense(350, activation="relu"))
model.add(Dropout(0.5))
```

We finnish by adding our output layer which is a fully connected layer with 1 output, and has an activation function of ```activation="sigmoid"``` because we are trying to solve a classification problem with a binary crossentropy.
```python
model.add(Dense(1, activation="sigmoid"))
```

We continue to compile the model with the ```adam``` optimizer and we are going to calculate loss for ```binary_crossentropy```.
```python```
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=\["accuracy"])
```
