import requests
from PIL import Image
from io import BytesIO
import datetime
import pytz
import numpy as np
import tensorflow as tf

# Step 1: Use NOAA API to get buoy location data
buoy_id = '46219' # example buoy ID
url = f'https://www.ndbc.noaa.gov/data/stations/station_history_{buoy_id}.txt'
response = requests.get(url)
lines = response.content.decode().split('\n')
lat, lon = [float(line.split()[-1]) for line in lines if 'Lat' in line or 'Lon' in line]

# Step 2: Use sunrise-sunset.org to calculate sunset time
date = datetime.datetime.now(pytz.timezone('US/Pacific'))
url = f'https://api.sunrise-sunset.org/json?lat={lat}&lng={lon}&date={date.strftime("%Y-%m-%d")}'
response = requests.get(url)
sunset_time_str = response.json()['results']['sunset']
sunset_time = datetime.datetime.strptime(sunset_time_str, '%I:%M:%S %p').time()

# Step 3: Download the image at sunset time
webcam_url = f'https://www.ndbc.noaa.gov/images/buoycam/{buoy_id}.jpg'
response = requests.get(webcam_url)
img = Image.open(BytesIO(response.content))

# Step 4: Preprocess the image data
img = img.resize((224, 224)) # example size for VGG16
img_arr = np.array(img.convert('RGB'))
img_arr = img_arr / 255.0 # normalize pixel values

# Step 5: Train a model to classify sunset images
# example code using VGG16 architecture
base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ... train the model on a dataset of sunset and non-sunset images ...

# Step 6: Use Flask to create a web app
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        buoy_id = request.form['buoy_id']
        webcam_url = f'https://www.ndbc.noaa.gov/images/buoycam/{buoy_id}.jpg'
        response = requests.get(webcam_url)


        # get the current time and check if it's after sunset
        date = datetime.datetime.now(pytz.timezone('US/Pacific'))
        url = f'https://api.sunrise-sunset.org/json?lat={lat}&lng={lon}&date={date.strftime("%Y-%m-%d")}'
        response = requests.get(url)
        sunset_time_str = response.json()['results']['sunset']
        sunset_time = datetime.datetime.strptime(sunset_time_str, '%I:%M:%S %p').time()
        if date.time() < sunset_time:
            return 'Sunset has not occurred yet at this location.'

        # preprocess the image and run it through the trained model
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224)) # example size for VGG16
        img_arr = np.array(img.convert('RGB'))
        img_arr = img_arr / 255.0 # normalize pixel values
        pred = model.predict(np.expand_dims(img_arr, axis=0))[0][0]
        is_sunset = pred > 0.5 # classify as sunset if model prediction > 0.5

        # render the template with the result
        return render_template('result.html', is_sunset=is_sunset)

    return render_template('index.html')


if __name__ == '__main__':
    print('Starting Flask app...', flush=True)
    app.run(debug=True)
