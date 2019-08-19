import os
import time
import filetype
import subprocess
import sys
import json

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import concatenate, Activation, Dense, Dropout, PReLU, Flatten

from data_processing import normalizeSoundTraining, eliminateAmbienceTraining, trainingSpectrogram

# setting size variables
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

    # loop through all files belonging to user in audioValidation directory
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

    # create a training and validation generators for voice recognition
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

    # create a training and validation generator for face recognition
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

    # initialize setps_per_epoch and validation_steps for both models
    img_steps_per_epoch = imgTrain_generator.n
    img_validation_steps = imgValidation_generator.n
    audio_steps_per_epoch = audioTrain_generator.n
    audio_validation_steps = audioValidation_generator.n

    # check if input model exists
    if data['model'] is None:
        # initialize shape parameters
        IMG_SHAPE = (image_size, image_size, 3)
        SPECT_SHAPE = (spect_size, spect_size, 3)

        if len(os.listdir("./models/")) is 0:
            # Create a base model for face recognition from the pre-trained model MobileNet V2
            base_model_img = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                               include_top=False,
                                                               weights='imagenet')

            # Create a base model for voice recognition from the pre-trained model MobileNetV2
            base_model_audio = tf.keras.applications.MobileNetV2(input_shape=SPECT_SHAPE,
                                                                 include_top=False,
                                                                 weights='imagenet')

            # freeze base models
            base_model_img.trainable = False
            base_model_audio.trainable = False

            # new model built off of the frozen base face model
            img_model = tf.keras.Sequential([
                base_model_img,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # new model built off of the frozen base voice model
            audio_model = tf.keras.Sequential([
                base_model_audio,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # compile the newly built face model
            img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            # compile the newly built voice model
            audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

            # Start with training the model with the base model frozen
            if data["epochs"] is None:
                # if no epochs were specified, set to arbitrary values
                img_epochs = 10
                audio_epochs = 5
            else:
                img_epochs = int(data["epochs"])
                audio_epochs = int(data["epochs"])

            # train the face recognition model with MobileNetV2 base still frozen
            img_history = img_model.fit_generator(imgTrain_generator,
                                                  steps_per_epoch=img_steps_per_epoch,
                                                  epochs=img_epochs,
                                                  workers=4,
                                                  validation_data=imgValidation_generator,
                                                  validation_steps=img_validation_steps)

            # train the voice recognition model with MobileNetV2 base still frozen
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
            for layer in base_model_img.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model_audio.layers[:fine_tune_at]:
                layer.trainable = False

            # recompile the img_model after having unfrozen the lower levels
            img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            # recompile the audio_model after having unfrozen the lower levels
            audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                                loss='binary_crossentropy',
                                metrics=['accuracy'])

            # train the recompiled img_model with unfrozen MobileNetV2 base model
            img_tune_history = img_model.fit_generator(imgTrain_generator,
                                                       steps_per_epoch=img_steps_per_epoch,
                                                       epochs=img_epochs,
                                                       workers=4,
                                                       validation_data=imgValidation_generator,
                                                       validation_steps=img_validation_steps)

            # train the recompiled audio_model with unfrozen MobileNetV2 base model
            audio_tune_history = audio_model.fit_generator(audioTrain_generator,
                                                           steps_per_epoch=audio_steps_per_epoch,
                                                           epochs=audio_epochs,
                                                           workers=4,
                                                           validation_data=audioValidation_generator,
                                                           validation_steps=audio_validation_steps)

        # if there is already one model existing freeze it as the base model and retrain the top layer for new user
        elif len(os.listdir("./models/")) is 1:
            # todo determine why global model can only be trained with 2 users
            # load the existing model
            model_list = os.listdir('./models/')
            existing_model = tf.keras.models.load_model("./models/" + model_list[0])

            # set a variable to the existing model for deletion once new model is being created
            del_existing_model = "./models/" + model_list[0]

            # cut the top concatenation layers out and break model down into its two base layers for retraining
            img_base_model = Model(existing_model.input[0], existing_model.layers[-8].output)
            audio_base_model = Model(existing_model.input[1], existing_model.layers[-7].output)

            # freeze the base models for both face and voice models
            img_base_model.trainable = False
            audio_base_model.trainable = False

            # create a new image model to build off the existing img_base_model
            img_model = Sequential([
                img_base_model,
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            # create a new audio mode to build off the existing audio_base_model
            audio_model = Sequential([
                audio_base_model,
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            # compile the newly built face model
            img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            # compile the newly built voice model
            audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                                loss='binary_crossentropy',
                                metrics=['accuracy'])

            # Start with training the model with the base model frozen
            if data["epochs"] is None:
                # if no epochs were specified, set to arbitrary values
                img_epochs = 10
                audio_epochs = 5
            else:
                img_epochs = int(data["epochs"])
                audio_epochs = int(data["epochs"])

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

            # unfreeze lower levels of face and voice base models
            img_model.trainable = True
            audio_model.trainable = True

            # define layer at which we will unfreeze the model
            fine_tune_at = 100

            # freeze all layers that are not included in the fine_tune_at parameter
            for layers in img_base_model.layers[:fine_tune_at]:
                layers.trainable = False
            for layers in audio_base_model.layers[:fine_tune_at]:
                layers.trainable = False

            # recompile the img_model after having unfrozen the lower levels
            img_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            # recompile the audio_model after having unfrozen the lower levels
            audio_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                                loss='binary_crossentropy',
                                metrics=['accuracy'])

            # retrain image model with unfrozen lower levels of the neural network
            img_history_tuned = img_model.fit_generator(imgTrain_generator,
                                                        steps_per_epoch=img_steps_per_epoch,
                                                        epochs=img_epochs,
                                                        workers=4,
                                                        validation_data=imgValidation_generator,
                                                        validation_steps=img_validation_steps)

            # retrain audio model with unfrozen lower levels of hte neural network
            audio_history_tuned = audio_model.fit_generator(audioTrain_generator,
                                                          steps_per_epoch=audio_steps_per_epoch,
                                                          epochs=audio_epochs,
                                                          workers=4,
                                                          validation_data=audioValidation_generator,
                                                          validation_steps=audio_validation_steps)

            # delete the existing model as the new global model will be saved at the end of training
            os.unlink(del_existing_model)

        # concatenate use img_model and audio_model output tensors as inputs to a concatenation layer
        merged_output = concatenate([img_model.output, audio_model.output], axis=-1)

        # pass concatenated outputs through series of layers
        layer = Flatten()(merged_output)
        layer = Dense(2, activation='relu')(layer)
        layer = Dense(1, activation='linear')(layer)
        layer = Flatten()(layer)
        out = Activation('sigmoid')(layer)

        # create a new model from the concatenated output tensors of the individual audio and img models
        concat_model = Model(inputs=[img_model.input, audio_model.input], outputs=[out])

        # compile the concatenated model
        concat_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

        print("Finished training, saving model...")
        # save the concatenated model containing the multi-modality of face and voice recognition
        if os.path.exists("./models/"):
            date = time.time()
            print("Saving new model to: ../models/" + str(date) + ".h5")
            concat_model.save("./models/" + str(date) + ".h5")
        else:
            os.makedirs("./models/")
            date = time.time()
            print("Saving new model to: ../models/" + str(date) + ".h5")
            concat_model.save("./models/" + str(date) + ".h5")
        print("done")

    # todo complete the logical case for PythonShell being passed an already existing model
    else:
        # load input model
        passed_model = tf.keras.models.load_model(str(data['model']))

        # if no models exist
        if os.listdir("./models/") is None:

            date = time.time()
            passed_model.save("./models/" + str(date) + ".h5")
        else:
            existing_model = os.listdir("./models/")
            existing_base_model = tf.keras.models.load_model("./models/" + str(existing_model[0]))


if __name__ == '__main__':
    train()