import os
import time
import filetype
import subprocess
import sys
import json

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate, Activation, Dense, Dropout

from data_processing import normalizeSoundTraining, eliminateAmbienceTraining, trainingSpectrogram

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
            os.rename('./' + fname, path)

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
    print("Conversion from '.wav' to '.png' successfully")

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
    if data['img_model'] is None and data['audio_model'] is None:
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
        audio_epochs = int(data["epochs"])

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

    # unfreeze the lower levels of the audio and image networks
    base_model_img.trainable = True
    base_model_audio.trainable = True

    # define the layers at which we will unfreeze
    fine_tune_at = 100

    # keep all layers before the 'fine_tune_at' frozen for both audio and visual model
    for layers in base_model_img[:fine_tune_at]:
        layers.trainable = False
    for layers in base_model_audio[:fine_tune_at]:
        layers.trainable = False

    # recompile the img_model and audio_model after having unfrozen the lower levels
    img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    # train the recompiled im_model and audio_model
    img_tune_history = img_model.fit_generator(imgTrain_generator,
                                               steps_per_epoch=img_steps_per_epoch,
                                               epochs=img_epochs,
                                               workers=4,
                                               validation_data=imgValidation_generator,
                                               validation_steps=img_validation_steps)

    audio_tune_history = audio_model.fit_generator(audioTrain_generator,
                                                   steps_per_epoch=audio_steps_per_epoch,
                                                   epochs=audio_epochs,
                                                   workers=4,
                                                   validation_data=audioValidation_generator,
                                                   validation_steps=audio_validation_steps)

    ######################
    # Concatenate models #
    ######################

    # concatenate audio_model and img_model to create a multimodal network with spectrogram and face images as inputs
    merged_output = Concatenate(axis=-1)([img_model.output, audio_model.output])
    out = Dense(128, activation='sigmoid')(merged_output)
    out = Dropout(0.8)(out)
    out = Dense(32, activation='sigmoid')(out)
    out1 = Dense(1, activation='sigmoid')(out)

    # create a new model by concatenating the output tensors of the individual audio and img models
    concat_model = Model([img_model.input, audio_model.input], out)

    # compile the concatenated model
    concat_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

    print("Finished training, saving model...")

    # save the concatenated model containing the multi-modality of face and voice recognition
    if data['img_model'] is None or data['audio_model'] is None:
        if os.path.exists("./models/" + str(data['name']) + "/"):
            date = time.time()
            print("Saving new model to: ../models/" + str(data['name']) + "/" + str(date) + ".h5")
            concat_model.save("./models/" + str(data['name']) + "/" + str(date) + ".h5")
        else:
            os.makedirs("./models/" + str(data['name']) + "/")
            date = time.time()
            print("Saving new model to: ../models/" + str(data['name']) + "/" + str(date) + ".h5")
            concat_model.save("./models/" + str(data['name']) + "/" + str(date) + ".h5")
    else:
        concat_model.save(data['img_model'])
        concat_model.save(data['audio_model'])

    print("done")


if __name__ == '__main__':
    train()
