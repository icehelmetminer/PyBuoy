# Goal:
# Using the pretrained Keras Model in models/converted_keras/keras_model.h5
# to predict the class of the image as it is saved in the panels folders

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import random
import pickle
model_filepath = 'models/converted_keras/keras_model.h5'
import tensorflow as tf

try:
    model = tf.keras.models.load_model(model_filepath)
except:
    print('Model not found')

tf.keras.models.load_model(
    model_filepath, custom_objects=None, compile=True, options=None
)
print('Model loaded')
# Loading the image
image_path = 'images/panels/41001/2022_11_5_19_27.jpg_panel_1.jpg'
img = cv2.imread(image_path) # Reading the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image to RGB
img = cv2.resize(img, (224, 224)) # Resizing the image
img = img / 255 # Normalizing the image
img = np.array(img).reshape(-1, 224, 224, 3) # Reshaping the image
print('image loaded')

# Predicting the class
prediction = model.predict(img)
print(prediction)
