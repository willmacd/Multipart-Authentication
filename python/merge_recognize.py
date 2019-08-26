import os
import io
import sys
import cv2
import json
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

# ignore warning thrown by tensorflow interfering with login process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set threshold for authentication
threshold = 45   # subject to change later in development

# fetch data passed through PythonShell from app.js
lines = sys.stdin.readline()
data = json.loads(lines)
name = str(data['name'])
model = str(data['model'])


# function to recognize user using multimodal visual and audio models
def recognize():
    # check if there was an existing model passed through python shell otherwise return to app.js
    if data['model'] is not None:
        model = tf.keras.models.load_model(str(data['model']))
    else:
        print("There is no model for this specified user")
        return

    # encode the image data passed through python shell
    img64 = str.encode(data['image'])
    spectro64 = str.encode(data['spectro'])

    # if img64/spectro64 are missing padding, add padding to base64 file
    missing_padding = len(data) % 4
    if missing_padding:
        img64 += b'=' * (4 - missing_padding)
        spectro64 += b'=' * (4 - missing_padding)

    # decode the image data with base 64 encoding
    decodeImg = base64.b64decode(img64)
    decodeSpectro = base64.b64decode(spectro64)

    # create an image object of the decoded image data using PIL library
    imgObj = Image.open(io.BytesIO(decodeImg))
    spectroObj = Image.open(io.BytesIO(decodeSpectro))
    img = cv2.cvtColor(np.array(imgObj), cv2.COLOR_BGR2RGB)
    spectro = cv2.cvtColor(np.array(spectroObj), cv2.COLOR_BGR2RGB)

    # resize input data and image data array to fit the output of MobileNetV2 (required a newaxis to be added)
    resizeImg = cv2.resize(img, (160, 160))
    image = resizeImg[np.newaxis, :, :, :]
    resizeSpectro = cv2.resize(spectro, (240, 240))
    spectrogram = resizeSpectro[np.newaxis, :, :, :]

    # output a prediction percentage for the spectrogram matching the user specified
    prediction = model.predict([image, spectrogram], steps=5)
    percentage = prediction[0][0] * float(100)

    # if the percentage is greater than the specified threshold allow access, otherwise deny access
    if percentage >= threshold:
        print("[MATCH] Face and voice detected is predicted to match " + name + "'s ==> " + str(percentage))
    else:
        print("[CONFLICT] Face and voice detected does not match " + name + "'s ==> " + str(percentage))


if __name__ == '__main__':
    recognize()
