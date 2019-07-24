import pyaudio
import os
import time
import filetype
import subprocess
import sys
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Sequential, Model
from keras.layers import concatenate, Activation, Dense

from voice_identifier.data_processing import normalizeSoundTraining, eliminateAmbienceTraining, trainingSpectrogram

# HYPERPARAMETERS
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'

image_size = 160
spect_size = 240
img_batch_size = 5
audio_batch_size = 4

# fetch data passed through PythonShell from app.js
lines = sys.stdin.readline()
data = json.loads(lines)
name = str(data['name'])
audioTrainingDir = str(data['audioTrainDir'])
audioValidationDir = str(data['audioValidationDir'])
imageTrainingDir = str(data['imageTrainDir'])
imageValidationDir = str(data['imageValidationDir'])


def train():
    # processing the audio training and validation data
    for path in os.listdir(audioTrainingDir + 'user/'):
        path = os.path.join(audioTrainingDir + 'user/', path)
        fname = os.path.basename(path)

        # check that the audio files are saved under the correct extension
        # if file extension is not '.wav' then convert to '.wav' format
        kind = filetype.guess(path)
        if kind.extension != "wav":
            command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + fname
            subprocess.call(command, shell=True)
            os.remove(path)
            os.rename('/' + fname, path)

    for path in os.listdir(audioValidationDir + 'user/'):
        path = os.path.join(audioValidationDir + 'user/', path)
        fname = os.path.basename(path)

        # check that the audio files are saved under the correct extension
        # if file extension is not '.wav' then convert to '.wav' format
        kind = filetype.guess(path)
        if kind.extension != "wav":
            command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + fname
            subprocess.call(command, shell=True)
            os.remove(path)
            os.rename('./' + fname, path)

    # process data in the training files directory
    normalizeSoundTraining(name)
    eliminateAmbienceTraining(name)
    trainingSpectrogram(name)

    # Rescale all images by 1./255 and apply image augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    audioTrain_generator = train_datagen.flow_from_directory(
        audioTrainingDir,
        target_size=(spect_size, spect_size),
        batch_size=audio_batch_size,
        class_mode='binary')

    audioValidation_generator = validation_datagen.flow_from_directory(
        audioValidationDir,
        target_size=(spect_size, spect_size),
        batch_size=audio_batch_size,
        class_mode='binary')

    imgTrain_generator = train_datagen.flow_from_directory(
        imageTrainingDir,
        target_size=(image_size, image_size),
        batch_size=img_batch_size,
        class_mode='binary')

    imgValidation_generator = validation_datagen.flow_from_directory(
        imageValidationDir,
        target_size=(image_size, image_size),
        batch_size=img_batch_size,
        class_mode='binary')

    ##################
    # Model Building #
    ##################

    # check if input model exists
    if data['img_model'] is None and data['audio_model']:
        # create new model
        IMG_SHAPE = (image_size, image_size, 3)
        SPECT_SHAPE = (spect_size, spect_size, 3)

        # Create the base model from the pre-trained model MobileNet V2
        base_model_img = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                           include_top=False,
                                                           weights='imagenet')

        base_model_audio = tf.keras.applications.MobileNetV2(input_shape=SPECT_SHAPE,
                                                             include_top=False,
                                                             weights='imagenet')

        # freeze base model
        base_model_img.trainable = False
        base_model_audio.trainable = False

        # new model built from base model
        img_model = tf.keras.Sequential([
            base_model_img,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        audio_model = tf.keras.Sequential([
            base_model_audio,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

    else:
        # load input model
        img_model = tf.keras.models.load_model(str(data['img_model']))
        audio_model = tf.keras.models.load_model(str(data['audio_model']))

    # Start with training the model with the base model frozen
    if data["epochs"] is None:
        img_epochs = 10
        audio_epochs = 5
    else:
        img_epochs = int(data["epochs"])
        audio_epochs = int(data['epochs'])

    img_steps_per_epoch = imgTrain_generator.n
    img_validation_steps = imgValidation_generator.n
    audio_steps_per_epoch = audioTrain_generator.n
    audio_validation_steps = audioValidation_generator.n

    img_history = img_model.fit_generator(imgTrain_generator,
                                          steps_per_epoch=img_steps_per_epoch,
                                          epochs=img_epochs,
                                          workers=4,
                                          validation_data=imgValidation_generator,
                                          validation_steps=img_validation_steps)

    audio_history = audio_model.fit_generator(audioTrain_generator,
                                              steps_per_epoch=audio_steps_per_epoch,
                                              epochs=audio_epochs,
                                              workers=4,
                                              validation_data=audioValidation_generator,
                                              validation_steps=audio_validation_steps)

    # Tune the model by training with base model unfrozen
    base_model_img.trainable = True
    base_model_audio.trainable = True

    # Fine tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model_img.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model_audio.layers[:fine_tune_at]:
        layer.trainable = False

    img_model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                      metrics=['accuracy'])

    audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    ######################
    # Concatenate models #
    ######################

    # concatenate audio_model and img_model to create a multimodal network with spectrogram and face images as inputs
    concat_model = concatenate([img_model, audio_model])

    merged_model = Sequential()
    merged_model.add(Activation('sigmoid'))
    merged_model.add(Dense(256))
    merged_model.add(Activation('sigmoid'))
    merged_model.add(Dense(4))
    merged_model.add(Activation('sigmoid'))

    combined_model = Model([img_model.input, audio_model.input], merged_model(concat_model))
    combined_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                           loss='binary_crossentropy,',
                           metrics=['accuracy'])

    print("Finished training, saving model...")

    if data['img_model'] is None or data['audio_model'] is None:
        if os.path.exists("./models/" + str(data['name']) + "/"):
            date = time.time()
            print("Saving new model to: ../models/" + str(data['name']) + "/" + str(date) + ".h5")
            combined_model.save("./models/" + str(data['name']) + "/" + str(date) + ".h5")
        else:
            os.makedirs("./models/" + str(data['name']) + "/")
            date = time.time()
            print("Saving new model to: ../models/" + str(data['name']) + "/" + str(date) + ".h5")
            combined_model.save("./models/" + str(data['name']) + "/" + str(date) + ".h5")
    else:
        combined_model.save(data['img_model'])
        combined_model.save(data['audio_model'])

    print("done")


if __name__ == '__main__':
    train()
