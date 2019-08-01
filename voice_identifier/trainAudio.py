import os
import time
import filetype
import subprocess
import sys
import json
import tensorflow as tf

from data_processing import normalizeSoundTraining, eliminateAmbienceTraining, trainingSpectrogram

spect_size = 240
batch_size = 4

# fetch data passed through PythonShell from app.js
lines = sys.stdin.readline()
data = json.loads(lines)
name = str(data['name'])
trainingDir = str(data['trainingDir'])
validationDir = str(data['validationDir'])


def trainAudio(name):
    # setting paths to training file directories
    trainFiles = trainingDir + 'user/'
    validationFiles = validationDir + 'user/'

    ###################
    # data processing #
    ###################

    for path in os.listdir(trainFiles):
        path = os.path.join(trainFiles, path)
        fname = os.path.basename(path)

        # check that the audio files are saved under the correct extension
        # if file extension is not '.wav' then convert to '.wav' format
        kind = filetype.guess(path)
        if kind.extension != "wav":
            command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + fname
            subprocess.call(command, shell=True)
            os.remove(path)
            os.rename('./' + fname, path)

    for path in os.listdir(validationFiles):
            path = os.path.join(validationFiles, path)
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

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    audioTrain_generator = train_datagen.flow_from_directory(
        trainingDir,
        target_size=(spect_size, spect_size),
        batch_size=batch_size,
        class_mode='binary')

    audioValidation_generator = validation_datagen.flow_from_directory(
        validationDir,
        target_size=(spect_size, spect_size),
        batch_size=batch_size,
        class_mode='binary')

    ##################
    # Building Model #
    ##################

    # check that a model does not yet exist
    if data['model'] is None:
        # specifying the shape of the input spectrogram
        SPECT_SHAPE = (spect_size, spect_size, 3)

        # creating a base model from pre-trained MobileNetV2 network
        base_model = tf.keras.applications.MobileNetV2(input_shape=SPECT_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        # freeze the base model
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    else:
        model = tf.keras.models.load_model(str(data['model']))

    ############
    # Training #
    ############

    if data['epochs'] is None:
        epochs = 5
    else:
        epochs = int(data['epochs'])

    steps_per_epoch = audioTrain_generator.n
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

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    tuneHistory = model.fit_generator(audioTrain_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      workers=4,
                                      validation_data=audioValidation_generator,
                                      validation_steps=validation_steps)

    print("Finished training, saving model...")

    if data['model'] is None:
        if os.path.exists("./models/" + str(data['name']) + "/"):
            date = time.time()
            print("Saving new audio model to: ../models/" + str(data['name']) + "/" + "voice" + ".h5")
            model.save("./models/" + str(data['name']) + "/" + "voice" + ".h5")    # str(date)
        else:
            os.makedirs("./models/" + str(data['name']) + "/")
            date = time.time()
            print("Saving new audio model to: ../models/" + str(data['name']) + "/" + "voice" + ".h5")
            model.save("./models/" + str(data['name']) + "/" + "voice" + ".h5")    # str(date)
    else:
        model.save(data['model'])
    model.summary()
    print("done")


if __name__ == "__main__":
    trainAudio(name)

