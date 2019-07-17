import pyaudio
import os
import time
import pickle
import numpy as np
import filetype
import subprocess
import sys
import json
import tensorflow as tf
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM

from data_processing import normalizeSoundTraining, eliminateAmbienceTraining, trainingSpectrogram
from feature_extraction import extract_features

# HYPERPARAMETERS
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3.5

spect_size = 240
batch_size = 3

# fetch data passed through PythonShell from app.js
lines = sys.stdin.readline()
data = json.loads(lines)
name = str(data['name'])
trainingDir = str(data['audioTrainingDir'])
validationDir = str(data['audioValidationDir'])


# todo remove references to gmm model
def train_gmm(name):
    # setting paths to database directory and .gmm files in models
    source = DATABASE_DIR + name + '/audio/'
    # destination = DATABASE_DIR + name + '/gmm-model/'

    # count = 1

    ###################
    # data processing #
    ###################

    for path in os.listdir(source):
        path = os.path.join(source, path)
        fname = os.path.basename(path)

        # check that the audio files are saved under the correct extension
        # if file extension is not '.wav' then convert to '.wav' format
        kind = filetype.guess(path)
        if kind.extension != "wav":
            command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + fname
            subprocess.call(command, shell=True)
            os.remove(path)
            os.rename('./' + name + fname, path)

    normalizeSoundTraining(name)
    eliminateAmbienceTraining(name)
    trainingSpectrogram(name)

    audioTrain_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    audioValidation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    audioTrain_generator = audioTrain_datagen.flow_from_directory(
        trainingDir,
        target_size=(spect_size, spect_size),
        batch_size=batch_size,
        class_mode='binary')

    audioValidation_generator = audioValidation_datagen.flow_from_directory(
        validationDir,
        target_size=(spect_size, spect_size),
        batch_size=batch_size,
        class_mode='binary')

    ##################
    # Building Model #
    ##################

    # check that a model does not yet exist
    if data['model'] is None:
        # specifying the shape of the input spectrograms
        SPECT_SHAPE = (spect_size, spect_size, 3)

        # creating a base model from pre-trained MobileNetV2 network
        base_model = tf.keras.applications.MobileNetV2(input_shape=SPECT_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        # freeze the base model
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                      loss='binary_crossentropy',
                      metric=['accuracy'])

    else:
        model = tf.keras.models.load_model(str(data['model']))

    ############
    # Training #
    ############

    if data['epochs'] is None:
        epochs = 10
    else:
        epochs = int(data['epochs'])

    steps_per_epoch = audioTrain_generator
    validation_steps = audioValidation_generator.n

    history = model.fit_generator(audioTrain_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  workers=4,
                                  validation_data=audioValidation_generator,
                                  validation_steps=validation_steps)

    # unfreeze the base model and tune by training again
    base_model.trainable = True

    # train from layer 100 onwards
    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    tuneHistory = model.fit_generator(audioTrain_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      workers=4,
                                      validation_data=audioValidation_generator,
                                      validation_steps=validation_steps)

    if data['model'] is None:
        date = time.time()
        print("Saving new audio model to: ../models/" + str(data['name']) + "/" + str(date) + ".h5")
        os.makedirs("./models/" + str(data['name']) + "/")
        model.save("./models/" + str(data['name']) + "/" + str(date) + ".h5")
    else:
        model.save(data['model'])


if __name__ == '__main__':
    train_gmm(name)


'''
    for path in os.listdir(source):
        features = np.array([])

        # reading audio files of speaker
        sr, audio = read(source + path)

        # extract 40 dim MFCC and delta MFCC features
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        # when features of the 5 speaker files are concatenated, then train the model
        if count == 5:
            gmm = GMM(n_components=18, max_iter=300, covariance_type='diag', n_init=5)
            gmm.fit(features)

            # save the trained Gaussian Model
            pickle.dump(gmm, open(destination + name + '.gmm', 'wb'))
            print("Model for " + name + "'s voice has successfully been trained")

            features = np.asarray(())
            count = 0
        count = count + 1
'''

