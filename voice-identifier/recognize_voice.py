import pyaudio
import os
import io
import sys
import cv2
import json
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

from recognize_spectro_processing import process_spectro

# HYPERPARAMETERS
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3.5
threshold = 0.20   # subject to change later in development

# fetch data passed through PythonShell from app.js
lines = sys.stdin.readline()
data = json.loads(lines)
name = str(data['name'])
model = str(data['model'])


def recognize_voice(name):
    # check if there was an existing model passed through python shell
    if data['model'] is not None:
        model = tf.keras.models.load_model(str(data['model']))
    else:
        print("There is no model for this specified user")
        return

    # encode the image data passed through python shell
    img64 = str.encode(data['image'])

    # if img64 missing padding, add padding to base64 file
    missing_padding = len(data) % 4
    if missing_padding:
        img64 += b'='* (4 - missing_padding)

    # decode the image data with base 64 encoding
    decode = base64.b64decode(img64)

    # create an image object of the decoded image data using PIL
    imgObj= Image.open(io.BytesIO(decode))
    img = cv2.cvtColor(np.array(imgObj), cv2.COLOR_BGR2RGB)

    # resize imput data and image data array to fit the output of MobileNetV2
    resize = cv2.resize(img, (240, 240))
    image = resize[np.newaxis, :, :, :]

    # output a prediction percentage for the spectrogram matching the user specified
    prediction = model.predict(image, steps=1)
    percentage = prediction[0][0] * 100

    # if the percentage is greater than the specified threshold allow access, otherwise deny access
    if percentage >= threshold:
        authentication = True
        print("[VOICE MATCH] Voice detected is predicted to match " + data['name'] + "'s ==> " + str(percentage))
    else:
        authentication = False
        print("[Voice CONFLICT] Voice detected does not match " + data['name'] + "'s ==> " + str(percentage))
    return authentication


if __name__ == '__main__':
    recognize_voice(name)


'''
    # ensure that loginAttempt audio file is in '.wav' format (will work for multiple audio files as well)
    for path in os.listdir(test_file):
        path = os.path.join(test_file, path)
        fname = os.path.basename(path)



# read the test files
sr, audio = read(test_file + "loginAttempt.wav")

# extract the mfcc features from the file
vector = extract_features(audio, sr)

# get the likelihood score that 'loginAttempt.wav' matches the GMM (outputs a log() value of the score)
prob = model.predict_proba(vector)[:, 1].mean()

# if log_likelihood is greater than threshold grant access
if prob >= threshold:
    authentication = True
    print("[VOICE MATCH] Voice matches the specified user ==> " + str(prob))
else:
    authentication = False
    print("[VOICE CONFLICT] Voice does not match the specified user ==> " + str(prob))
return authentication
'''