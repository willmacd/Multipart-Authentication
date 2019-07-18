import subprocess
import pyaudio
import os
import io
import sys
import cv2
import json
import base64
import filetype
import numpy as np
from PIL import Image
import tensorflow as tf

from data_processing import normalizeSoundRecognizing, eliminateAmbienceRecognizing, recognizeSpectrogram

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
    # setting paths to test file and specifying voice model
    test_file = DATABASE_DIR + name + '/audioComparison/'

    if data['model'] is not None:
        model = tf.keras.models.load_model(str(data['model']))
    else:
        print("There is no model for this specified user")
        return

    # ensure that loginAttempt audio file is in '.wav' format (will work for multiple audio files as well)
    for path in os.listdir(test_file):
        path = os.path.join(test_file, path)
        fname = os.path.basename(path)

        # check that the audio files are saved under the correct extension
        # if file extension is not '.wav' then convert to '.wav' format
        kind = filetype.guess(path)
        if kind.extension != "wav":
            command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + fname
            subprocess.call(command, shell=True)
            os.remove(path)
            os.rename('./' + fname, path)

    # data preprocessing
    normalizeSoundRecognizing(name)
    eliminateAmbienceRecognizing(name)
    recognizeSpectrogram(name)

    img64 = str.encode(data['image'])

    decode = base64.b64decode(img64)

    imgObj= Image.open(io.BytesIO(decode))

    img = cv2.cvtColor(np.array([imgObj]), cv2.COLOR_BGR2RGB)

    resize = cv2.resize(img, (240, 240))
    image = resize[np.newaxis, :, :, :]

    prediction = model.predict(image, steps=1)
    percentage = prediction[0][0] * 100

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