
import pandas as pd

# this file runs the keras model on the collected images from the webcam network and returns a prediction for if the image contains a sunrise or not
# the model can be found here: models/buoy_model/keras_model.h5
# the images to be tested can be found here: images/webcam_captures where we will find locations such as 'tokyo_bay_japan' and 'shibuyaku_tokyo'
# we want to chase the sunrise around the world and only show those golden hours of the morning when the world is waking up. A romantic way to unite the world, around the sun.


# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras import initializers
from keras import constraints
from keras import activations
from keras import metrics
from keras import losses
from keras import models
from keras import utils
from keras import callbacks


# load the model
model = load_model('models/buoy_model/keras_model.h5')

# load the images
# we will load the images from the webcam_captures folder
# we will then run the model on each image and return a prediction for if the image contains a sunrise or not
# we will then save the predictions to a csv file

# get the list of locations
locations = os.listdir('images/webcam_captures')

# create a dataframe to store the predictions
predictions = pd.DataFrame(columns=['location', 'prediction'])

# loop through the locations
for location in locations:
    # get the list of images
    images = os.listdir('images/webcam_captures/' + location)
    # loop through the images
    for image in images:
        # load the image
        img = image.load_img('images/webcam_captures/' + location + '/' + image, target_size=(256, 256))
        # convert the image to an array
        img = image.img_to_array(img)
        # reshape the image
        img = img.reshape(1, 256, 256, 3)
        # predict the image
        prediction = model.predict(img)
        # add the prediction to the dataframe
        predictions = predictions.append({'location': location, 'prediction': prediction}, ignore_index=True)
        if prediction == 4: # if the prediction is a sunrise (4) then we will save the image to the batches folder
            image.save_img('images/batches/' + location + '/' + image, img)
            print('Sunrise detected in ' + location + ' at ' + image)
        else:
            pass

# save the predictions to a csv file
predictions.to_csv('predictions.csv')

# we can now use the predictions to find the sunrise in each location

# to move your conda environment to an external drive, follow these steps:
# 1. create a new environment on the external drive
# 2. activate the new environment
# 3. run the following commands:
# conda list --explicit > spec-file.txt

# 4. deactivate the environment

# 5. move the spec-file.txt file to the external drive